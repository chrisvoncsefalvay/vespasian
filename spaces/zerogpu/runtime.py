"""Session-isolated Rust controllers with request-context ZeroGPU dispatch."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

from bridge import Bridge
import hub

ROOT = Path(__file__).resolve().parent
RUNS: dict[str, "Run"] = {}
LOCK = threading.Lock()
MAX_ACTIVE = int(os.environ.get("VESPASIAN_SPACE_CONCURRENT_RUNS", "2"))
RETENTION_SECONDS = 2 * 60 * 60
CLIENT_TIMEOUT_SECONDS = 5 * 60
PUBLIC_FIELDS = {"model", "goal", "controls", "budget", "max_turns", "metric", "target_type",
                 "dataset_mode", "editor", "editor_model", "health_checks", "health_count", "max_new_tokens"}


def child_environment(root, bridge, *, token=False, api_key="", editor="openai"):
    allowed = {"PATH", "LANG", "LC_ALL", "TZ", "LD_LIBRARY_PATH", "SSL_CERT_FILE", "SSL_CERT_DIR",
               "REQUESTS_CA_BUNDLE", "CUDA_VISIBLE_DEVICES", "SYSTEMROOT"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    private_home = root / "home"
    private_home.mkdir(mode=0o700, exist_ok=True)
    env.update({
        "HOME": str(private_home), "HF_HOME": str(private_home / ".cache/huggingface"),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "0" if token else "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "VESPASIAN_REAL_PYTHON": sys.executable,
        "VESPASIAN_WORKER_SOCKET": str(bridge.path), "VESPASIAN_WORKER_SECRET": bridge.secret,
        "VESPASIAN_WORKER_SCRIPT": str(root / "run/worker.py"),
    })
    if token:
        env["HF_TOKEN"] = token
    if api_key:
        env["OPENAI_API_KEY" if editor == "openai" else "ANTHROPIC_API_KEY"] = api_key
    return env


def validate(config):
    if not str(config.get("goal") or "").strip():
        raise ValueError("Set an editing objective")
    if config.get("editor") not in {"openai", "anthropic"}:
        raise ValueError("Choose OpenAI or Anthropic as the editor")
    if not str(config.get("api_key") or "").strip():
        raise ValueError("Add an editor API key")
    if not 0.01 <= float(config.get("budget", 0)) <= 100:
        raise ValueError("Set an inference budget between $0.01 and $100")
    for name, minimum, maximum in (("example_limit", 5, 128), ("test_example_limit", 1, 128),
                                   ("max_turns", 1, 50), ("max_new_tokens", 1, 128), ("health_count", 1, 16)):
        value = config.get(name, minimum)
        if not isinstance(value, (int, float)) or int(value) != value or not minimum <= value <= maximum:
            raise ValueError(f"{name.replace('_', ' ').capitalize()} must be between {minimum} and {maximum}")
    if len(config.get("health_checks") or []) > 4:
        raise ValueError("Choose up to four health checks for this run")
    drop = config.get("health_max_drop")
    if drop is not None and not 0 <= float(drop) <= 1:
        raise ValueError("The health regression limit must be between 0 and 1")


def command(config, data, output, revision):
    binary = Path(os.environ.get("VESPASIAN_SPACE_BINARY", ROOT / "bin/vespasian"))
    shim = ROOT / "python_bridge.py"
    for path in (binary, shim):
        if not path.is_file():
            raise ValueError(f"The Space is missing {path.name}")
        path.chmod(0o755)
    args = [str(binary), str(data), "--json", "--no-tui", "--python", str(shim), "--device", "cuda",
            "--output", str(output), "--model", config["model"], "--revision", revision,
            "--goal", config["goal"], "--controls", config.get("controls") or "",
            "--target-type", config.get("target_type", "custom"),
            "--dataset-mode", config.get("dataset_mode", "paired"),
            "--metric", config.get("metric", "auto"),
            "--editor", config["editor"], "--editor-model", config["editor_model"],
            "--budget", str(config["budget"]), "--max-turns", str(int(config["max_turns"])),
            "--max-new-tokens", str(int(config["max_new_tokens"])), "--max-seq-length", "1024"]
    for check in config.get("health_checks") or []:
        args.extend(["--health-check", f"{check}:{int(config['health_count'])}"])
    if config.get("health_max_drop") is not None:
        args.extend(["--health-max-drop", str(config["health_max_drop"])])
    if config.get("save_trajectory"):
        args.append("--save-trajectory")
    return args


class Run:
    def __init__(self, owner, config, token=False):
        self.id = secrets.token_hex(24)
        self.owner = owner
        self.root = Path(tempfile.mkdtemp(prefix="vs-"))
        self.root.chmod(0o700)
        self.output = self.root / "run"
        self.config = {key: value for key, value in config.items() if key in PUBLIC_FIELDS}
        self.secrets = tuple(value for value in (token, config.get("api_key")) if value)
        self.started = time.monotonic()
        self.events = []
        self.status = "preparing"
        self.error = None
        self.process = None
        self.bridge = None
        self.cancelled = threading.Event()
        self.finished = False
        self.finished_at = None
        self.archive = None
        self.messages = queue.Queue()
        self.diagnostics = []
        self.controller = None
        self.advance_lock = threading.Lock()
        self.launch_lock = threading.Lock()
        self.state_lock = threading.RLock()
        self.last_seen = time.monotonic()
        self.pending_request = None
        self.pending_dispatch = None
        self.at_allocation_boundary = False

    def redact(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[redacted]")
            return value.replace(str(self.root), "[run]")
        if isinstance(value, dict):
            return {self.redact(key): self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        return value

    def state(self):
        with self.state_lock:
            return {"run_id": self.id, "events": list(self.events), "status": self.status,
                    "config": self.config, "error": self.error,
                    "elapsed": time.monotonic() - self.started, "finished": self.finished}

    def event(self, value):
        value = self.redact(value)
        with self.state_lock:
            self.events.append(value)
            if value.get("type") == "phase":
                self.status = value.get("message", "Running")
            elif value.get("type") == "complete":
                self.status = "Complete"
            elif value.get("type") == "error":
                self.error = value.get("message", "Run stopped")

    def _read(self, stream, errors=False):
        for line in stream:
            if errors:
                self.diagnostics.append(self.redact(line.strip()))
                self.diagnostics = self.diagnostics[-30:]
            else:
                try:
                    self.messages.put(json.loads(line))
                except ValueError:
                    self.diagnostics.append(self.redact(line.strip()))

    def drain(self):
        with self.state_lock:
            while True:
                try:
                    self.event(self.messages.get_nowait())
                except queue.Empty:
                    break

    def stop(self):
        self.cancelled.set()
        if self.process and self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

    def bundle(self):
        if not self.output.is_dir():
            return None
        for receipt in self.root.glob("*-source.json"):
            shutil.copy2(receipt, self.output / receipt.name)
        archive = self.root / f"vespasian-{self.id[:10]}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
            for path in sorted(self.output.rglob("*")):
                if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                    output.write(path, "vespasian/" + path.relative_to(self.output).as_posix())
        self.archive = archive
        return archive

    def start(self, config, token, availability):
        """Start CPU work without retaining a Gradio request or scheduling a GPU."""
        with self.launch_lock:
            if self.controller is not None:
                raise ValueError("This experiment has already started")
            self.controller = threading.Thread(
                target=self._control, args=(dict(config), token, availability),
                name=f"vespasian-{self.id[:10]}", daemon=True,
            )
            self.controller.start()

    def _control(self, config, token, availability):
        readers = []
        try:
            model = next((entry for entry in availability if entry["id"] == config["model"] and entry["available"]), None)
            if not model:
                raise ValueError("Choose an available model")
            self.status = "Retrieving your dataset"
            data = hub.prepare_data(config, self.root, token)
            if self.cancelled.is_set():
                raise InterruptedError("Run stopped")
            self.bridge = Bridge(self.root)
            env = child_environment(self.root, self.bridge, token=token,
                                    api_key=config["api_key"], editor=config["editor"])
            args = command(config, data, self.output, model["revision"])
            self.process = subprocess.Popen(args, env=env, cwd=self.root, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            start_new_session=True, bufsize=1)
            readers = [threading.Thread(target=self._read, args=(self.process.stdout,), daemon=True),
                       threading.Thread(target=self._read, args=(self.process.stderr, True), daemon=True)]
            for reader in readers:
                reader.start()
            stopping_since = None
            while self.process.poll() is None:
                self.drain()
                if not self.advance_lock.locked() and time.monotonic() - self.last_seen > CLIENT_TIMEOUT_SECONDS:
                    self.error = "The run stopped after five minutes without a browser connection"
                    self.stop()
                if self.cancelled.is_set():
                    stopping_since = stopping_since or time.monotonic()
                    if time.monotonic() - stopping_since > 5:
                        self._kill()
                        break
                time.sleep(0.05)
        except Exception as error:
            self.error = self.redact(str(error))
        finally:
            if self.process and self.process.poll() is None:
                self.stop()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._kill()
                    self.process.wait(timeout=5)
            for reader in readers:
                reader.join(timeout=2)
            self.drain()
            # An in-flight allocation must release its model state before cleanup.
            with self.advance_lock:
                self._clear_dispatch()
                if self.bridge:
                    self.bridge.close()
                if self.process and self.process.returncode and not self.error and not self.cancelled.is_set():
                    self.error = self.diagnostics[-1] if self.diagnostics else "The run stopped before completion"
                try:
                    self.bundle()
                except OSError:
                    self.error = "The run finished but its download could not be packaged"
                finally:
                    for name in ("engines", "home"):
                        path = self.root / name
                        if path.is_dir():
                            shutil.rmtree(path)
                    self.secrets = ()
                    self.status = "Stopped" if self.cancelled.is_set() else ("Complete" if not self.error else "Run stopped")
                    self.finished_at = time.monotonic()
                    self.finished = True

    def _kill(self):
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _clear_dispatch(self):
        iterator, self.pending_dispatch = self.pending_dispatch, None
        self.pending_request = None
        self.at_allocation_boundary = False
        if iterator is not None:
            try:
                iterator.close()
            except Exception as error:
                self.error = self.redact(str(error))

    def advance(self, dispatch):
        """Run at most one GPU allocation in the current browser request."""
        self.last_seen = time.monotonic()
        if not self.advance_lock.acquire(blocking=False):
            yield self.state()
            return
        try:
            self.drain()
            if self.finished or self.cancelled.is_set() or self.bridge is None:
                yield self.state()
                return
            if self.pending_request is None:
                request = self.bridge.next(timeout=0)
                if request is None:
                    yield self.state()
                    return
                self.pending_request = request
                self.pending_dispatch = iter(dispatch(str(request.state_dir), request.payload, boundaries=True))
            request = self.pending_request
            # A previous callback may already have consumed the next marker.
            allocated = self.at_allocation_boundary
            self.at_allocation_boundary = False
            self.status = "Waiting for ZeroGPU"
            yield self.state()
            last_yield = time.monotonic()
            try:
                while not self.cancelled.is_set():
                    response = next(self.pending_dispatch)
                    if response.get("type") == "gpu_allocation":
                        if allocated:
                            self.at_allocation_boundary = True
                            break
                        allocated = True
                        continue
                    request.send(response)
                    self.drain()
                    if time.monotonic() - last_yield > 0.25:
                        yield self.state()
                        last_yield = time.monotonic()
            except StopIteration:
                self._clear_dispatch()
            except Exception as error:
                request.send({"type": "result", "id": request.payload["id"], "ok": False,
                              "error": self.redact(str(error))})
                self._clear_dispatch()
            yield self.state()
        except GeneratorExit:
            self.stop()
            self._clear_dispatch()
            raise
        finally:
            self.last_seen = time.monotonic()
            self.advance_lock.release()

    def execute(self, config, token, availability, dispatch):
        """Drive the same lifecycle locally. Browser clients use start/advance."""
        self.start(config, token, availability)
        try:
            while not self.finished:
                yield from self.advance(dispatch)
                time.sleep(0.05)
            yield self.state()
        finally:
            if not self.finished:
                self.stop()


def begin(owner, config, token=False):
    validate(config)
    cleanup()
    with LOCK:
        if any(run.owner == owner and not run.finished for run in RUNS.values()):
            raise ValueError("Stop your current run before starting another")
        if sum(not run.finished for run in RUNS.values()) >= MAX_ACTIVE:
            raise ValueError("Both experiment slots are in use. Try again when a run completes")
        if shutil.disk_usage(tempfile.gettempdir()).free < 8 * 1024**3:
            raise ValueError("The Space's export storage is full. Completed downloads expire after two hours")
        run = Run(owner, config, token)
        RUNS[run.id] = run
        return run


def get_run(owner, state):
    run = RUNS.get((state or {}).get("run_id", ""))
    if run is None or run.owner != owner:
        raise ValueError("Start a run in this session first")
    return run


def cleanup(now=None):
    now = time.monotonic() if now is None else now
    with LOCK:
        expired = [key for key, run in RUNS.items()
                   if run.finished_at is not None and now - run.finished_at >= RETENTION_SECONDS]
        for key in expired:
            run = RUNS.pop(key)
            shutil.rmtree(run.root)


def start_cleanup():
    def sweep():
        while True:
            time.sleep(60)
            cleanup()
    threading.Thread(target=sweep, name="vespasian-session-cleanup", daemon=True).start()
