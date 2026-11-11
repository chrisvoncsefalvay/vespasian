"""Carry the CLI worker protocol into the Gradio request's GPU context."""

from __future__ import annotations

import hmac
import json
from pathlib import Path
import queue
import secrets
import socket
import threading

MAX_MESSAGE = 32 * 1024 * 1024


class Request:
    def __init__(self, state_dir: Path, payload: dict):
        self.state_dir = state_dir
        self.payload = payload
        self.responses: queue.Queue = queue.Queue()
        self.closed = threading.Event()

    def send(self, value: dict):
        if not self.closed.is_set():
            self.responses.put(value)


class Bridge:
    """Socket threads only transport JSON. The caller dispatches GPU work."""

    def __init__(self, root: Path):
        self.root = root
        self.path = root / "worker.sock"
        self.secret = secrets.token_urlsafe(32)
        self.requests: queue.Queue[Request] = queue.Queue()
        self.closed = threading.Event()
        self.connections: set[socket.socket] = set()
        self.lock = threading.Lock()
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.path.chmod(0o600)
        self.server.listen(4)
        self.server.settimeout(0.2)
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def _accept(self):
        while not self.closed.is_set():
            try:
                client, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self.lock:
                self.connections.add(client)
            threading.Thread(target=self._connection, args=(client,), daemon=True).start()

    def _connection(self, client):
        request = None
        try:
            client.settimeout(10)
            stream = client.makefile("rwb")
            handshake = json.loads(stream.readline(4096))
            if not hmac.compare_digest(str(handshake.get("secret", "")), self.secret):
                return
            client.settimeout(None)
            state = self.root / "engines" / secrets.token_hex(16)
            state.mkdir(parents=True, mode=0o700)
            while not self.closed.is_set():
                line = stream.readline(MAX_MESSAGE + 1)
                if not line:
                    break
                if len(line) > MAX_MESSAGE:
                    raise ValueError("Worker request exceeds the message limit")
                payload = json.loads(line)
                if not isinstance(payload, dict) or not isinstance(payload.get("id"), int):
                    raise ValueError("Invalid worker request")
                request = Request(state, payload)
                self.requests.put(request)
                while not self.closed.is_set():
                    try:
                        response = request.responses.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    stream.write(json.dumps(response, ensure_ascii=False, allow_nan=False).encode() + b"\n")
                    stream.flush()
                    if response.get("type") == "result":
                        break
                request.closed.set()
        except (OSError, ValueError, TypeError):
            pass
        finally:
            if request is not None:
                request.closed.set()
            with self.lock:
                self.connections.discard(client)
            client.close()

    def next(self, timeout=0.1):
        try:
            return self.requests.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.closed.set()
        self.server.close()
        with self.lock:
            for client in self.connections:
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                client.close()
        self.thread.join(timeout=1)
        self.path.unlink(missing_ok=True)

