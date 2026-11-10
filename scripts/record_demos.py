#!/usr/bin/env python3
"""Record Vespasian's guided setup and a saved weight-editing run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

import pexpect
import pyte

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Vespasian: The Weight Space Editing Assistant"
THEME = {"bg": "#14171b", "fg": "#ece2d2", "palette": "#14171b:#e09389:#97b993:#d6a071:#93adc2:#b09bc1:#99b9b2:#ece2d2:#3e4248:#e09389:#97b993:#d6a071:#93adc2:#b09bc1:#99b9b2:#ffffff"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


class Recording:
    def __init__(self, binary: Path, args: list[str], name: str, evidence: Path, columns=112, rows=34):
        env = dict(os.environ, TERM="xterm-256color", COLORTERM="truecolor", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
        for key in ("NO_COLOR", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            env.pop(key, None)
        self.start = time.monotonic()
        self.child = pexpect.spawn(str(binary), args, cwd=ROOT, env=env, dimensions=(rows, columns), encoding="utf-8", timeout=10)
        self.child.delaybeforesend = 0
        self.events = []
        self.screen = pyte.Screen(columns, rows)
        self.stream = pyte.Stream(self.screen)
        self.name, self.evidence = name, evidence
        clip_title = "guided setup" if name == "setup" else "will climb replay"
        self.header = {"version": 2, "width": columns, "height": rows, "timestamp": int(time.time()), "title": TITLE + " · " + clip_title, "command": "vespasian" if name == "setup" else "vespasian replay runs/greetings", "env": {"TERM": "xterm-256color"}, "theme": THEME}
        self.stages = []

    @property
    def elapsed(self):
        return time.monotonic() - self.start

    @property
    def text(self):
        return "\n".join(self.screen.display)

    def drain(self, duration=0.2):
        until = time.monotonic() + duration
        while time.monotonic() < until:
            try:
                content = self.child.read_nonblocking(262144, timeout=0.03)
                self.events.append([round(self.elapsed, 5), "o", content])
                self.stream.feed(content)
            except pexpect.TIMEOUT:
                pass
            except pexpect.EOF:
                break

    def wait_for(self, phrase: str, timeout=25):
        deadline = time.monotonic() + timeout
        while phrase not in self.text and time.monotonic() < deadline:
            self.drain(0.1)
        if phrase not in self.text:
            (self.evidence / f"{self.name}-failure.txt").write_text(self.text)
            raise RuntimeError(f"Screen did not contain {phrase!r}")

    def send(self, keys: str, hold=0.25):
        self.child.send(keys)
        self.drain(hold)

    def stage(self, label: str, phrase: str | None = None, hold=1.4):
        if phrase:
            self.wait_for(phrase)
        self.events.append([round(self.elapsed, 5), "m", label])
        self.stages.append({"label": label, "time": round(self.elapsed, 3)})
        (self.evidence / f"{self.name}-{len(self.stages):02}.txt").write_text(self.text)
        self.drain(hold)

    def finish(self, destination: Path):
        self.header["duration"] = round(self.elapsed, 5)
        self.events.append([self.header["duration"], "o", ""])
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in [self.header, *self.events]) + "\n"
        if str(ROOT.parent) in text or re.search(r"hf_[A-Za-z0-9]{20,}", text):
            raise RuntimeError("Recording contains a local home path or credential")
        destination.write_text(text)
        self.child.sendcontrol("c")
        self.child.expect(pexpect.EOF, timeout=10)
        self.child.close()
        if self.child.exitstatus != 0:
            raise RuntimeError(f"Recording exited with {self.child.exitstatus}")
        return {"path": str(destination.relative_to(ROOT)), "sha256": sha(destination), "bytes": destination.stat().st_size, "seconds": self.header["duration"], "columns": self.header["width"], "rows": self.header["height"], "stages": self.stages, "exit_status": self.child.exitstatus}


def record_setup(binary: Path, python: Path, evidence: Path, output: Path):
    args = ["--target-type", "factual", "--dataset-config", "main", "--limit", "20", "--goal", "Give accurate worked solutions to arithmetic word problems.", "--controls", "Keep copying and basic arithmetic accurate.", "--budget", "2", "--python", str(python)]
    rec = Recording(binary, args, "setup", evidence)
    try:
        rec.stage("Choose a behaviour", "Factual answers", 1.3)
        rec.send("\r")
        rec.stage("Choose a model", "Qwen 3.5", 1.2)
        rec.send("\r")
        rec.send("hf://openai/gsm8k", 0.25)
        rec.send("\r", 0.25)
        rec.stage("Read the source record", "question", 2.8)
        rec.wait_for("Enter confirm mapping")
        rec.send("\r")
        rec.stage("Confirm prompt and target", "Prompt · question", 3.0)
        rec.send("\x1b[6~", 1.4)
        rec.send("\r")
        rec.stage("Set the objective and controls", "Keep intact", 2.2)
        rec.send("\r")
        rec.send("\x1b[B\x1b[B\x1b[B")
        rec.stage("Choose the measure", "Goal score", 1.5)
        rec.send("\r")
        rec.send(" ", 0.2)
        rec.send("\t\x15" + "20" + "\t\x15" + "5", 0.3)
        rec.stage("Select health checks", "Maximum drop", 2.3)
        rec.send("\r")
        rec.stage("Choose the editor", "Codex", 1.5)
        rec.send("\r")
        rec.send("\t", 0.2)
        rec.stage("Set model and dollar budget", "Frontier model", 2.3)
        rec.send("\r")
        rec.stage("Choose the checkpoint destination", "Keep the checkpoint locally", 1.3)
        rec.send("\r")
        rec.send("\x1b[B", 0.2)
        rec.stage("Save the editor trajectory", "Save a local trajectory", 1.6)
        rec.send("\r")
        rec.stage("Review the run", "Enter start run", 3.0)
        result = rec.finish(output / "setup.cast")
        result["dataset"] = "openai/gsm8k"
        result["frontier_calls"] = 0
        result["checkpoint_writes"] = 0
        return result
    finally:
        if rec.child.isalive():
            rec.child.sendcontrol("c")
            rec.child.close(force=True)


def projected_replay(source: Path):
    """Project saved legacy measurements into the current replay event schema."""
    files = [source / "events.jsonl", source / "model.json", source / "baseline.json", source / "run.json"]
    original = [json.loads(line) for line in files[0].read_text().splitlines()]
    model, baseline, settings = map(read, files[1:])
    config_path = Path(model["source"]["path"]) / "config.json"
    config = read(config_path)
    expected = next(row["sha256"] for row in model["source"]["files"] if row["path"] == "config.json")
    assert sha(config_path) == expected
    text_config = config.get("text_config", config)
    schedule = []

    def add(at, event):
        event = copy.deepcopy(event)
        for key in ("output",):
            if key in event:
                event[key] = "runs/greetings"
        if event.get("type") == "start":
            event["editor"] = event["editor"].replace(" / ", "/")
        for metrics_key, health_key in (("metrics", "health"), ("best_metrics", "best_health")):
            if "preserve" in event.get(metrics_key, {}):
                preserve = event[metrics_key]["preserve"]
                assert preserve["count"] == 1
                event[health_key] = {"arithmetic_preservation": {"label": "Arithmetic preservation (1 example)", "score": preserve["exact_match"], "count": preserve["count"], "metric": "exact_match"}}
        schedule.append((at, event))

    def answers(at, value, label, turn):
        progress = [event for event in original if event["type"] == "progress" and event.get("message", "").startswith(label + " · ")]
        for index, sample in enumerate(value["samples"], 1):
            example = copy.deepcopy(sample)
            identity = sample["split"] + "\0" + sample["prompt"]
            example["id"] = sample["split"] + ":" + hashlib.sha256(identity.encode()).hexdigest()
            example["output"] = sample["prediction"]
            if index <= len(progress):
                add(at + (index - 1) * 0.10 - 0.025, progress[index - 1])
            add(at + (index - 1) * 0.10, {"type": "answer", "phase": "evaluating", "label": label, "turn": turn, "current": index, "total": len(value["samples"]), "example": example})
        completion = next(event for event in original if event["type"] == "progress" and event.get("phase") == "evaluating" and event.get("current") == event.get("total") == len(value["samples"]))
        add(at + len(value["samples"]) * 0.10 + 0.05, completion)

    add(0, next(event for event in original if event["type"] == "start"))
    add(0.1, next(event for event in original if event["type"] == "assessment"))
    add(0.15, {"type": "model_structure", "model": model["model"], "architecture": text_config["model_type"], "num_layers": text_config["num_hidden_layers"], "editable": model["editable"]})
    costs = [event for event in original if event["type"] == "cost"]
    add(0.2, costs[0])
    add(0.25, next(event for event in original if event["type"] == "phase" and event.get("phase") == "baseline"))
    answers(0.4, baseline, "baseline", 0)
    add(1.3, next(event for event in original if event["type"] == "baseline"))
    parent = 0
    for number in range(1, 5):
        path = source / f"turn-{number:03}.json"
        files.append(path)
        turn = read(path)
        at = 3.4 + (number - 1) * 4.4
        add(at, {"type": "phase", "phase": "reflecting", "message": "Recorded editor proposal", "turn": number})
        add(at + 0.1, {"type": "editor_activity", "status": "complete", "call_id": f"turn-{number:03}", "purpose": "Propose weight edit", "provider": settings["editor"], "model": settings["editor_model"], "summary": turn["proposal"]["rationale"], "turn": number})
        add(at + 0.2, {"type": "proposal", "parent_turn": parent, "proposal": turn["proposal"], "turn": number})
        if number < len(costs):
            add(at + 0.25, costs[number])
        if "alteration" in turn:
            alteration = turn["alteration"]
            begin = next(index for index, event in enumerate(original) if event["type"] == "phase" and event.get("message", "").endswith(f"turn {number}"))
            end = next(index for index, event in enumerate(original) if event["type"] == "turn" and event["turn"] == number)
            editing = [event for event in original[begin:end] if event["type"] == "progress" and event.get("phase") == "editing"]
            for index, event in enumerate(editing):
                add(at + 0.4 + index * 0.10, event)
            for index, tensor in enumerate(alteration["changed_tensors"], 1):
                add(at + 1.0, {"type": "tensor_edit", "status": "applied", "tensor": tensor["name"], "operation": alteration["operation"], "current": index, "total": len(alteration["changed_tensors"]), **{key: tensor[key] for key in ("shape", "relative_norm", "changed_elements")}})
            answers(at + 1.3, turn["evaluation"], f"turn-{number:03}", number)
        event = copy.deepcopy(next(event for event in original if event["type"] == "turn" and event["turn"] == number))
        event.update(parent_turn=parent, proposal=turn["proposal"], mutation=turn.get("alteration"))
        if "error" in turn:
            event["decision_reason"] = turn["error"]
        add(at + 2.6, event)
        if turn["accepted"]:
            parent = number
    add(20.0, next(event for event in original if event["type"] == "complete"))
    assert all(schedule[i][0] <= schedule[i + 1][0] for i in range(len(schedule) - 1))
    for _, event in schedule:
        for metrics_key, health_key in (("metrics", "health"), ("best_metrics", "best_health")):
            if health_key in event:
                preserve = event[metrics_key]["preserve"]
                displayed = event[health_key]["arithmetic_preservation"]
                assert displayed["count"] == preserve["count"]
                assert displayed["score"] == preserve["exact_match"]
        if event["type"] == "turn":
            recorded = next(row for row in original if row["type"] == "turn" and row["turn"] == event["turn"])
            for key in ("accepted", "metrics", "best_metrics"):
                assert event[key] == recorded[key]
        elif event["type"] == "answer":
            recorded = baseline if event["label"] == "baseline" else read(source / (event["label"] + ".json"))["evaluation"]
            sample = recorded["samples"][event["current"] - 1]
            assert all(event["example"][key] == value for key, value in sample.items())
        elif event["type"] == "tensor_edit":
            assert event["status"] == "applied"
            matches = [tensor for path in files if path.name.startswith("turn-") for tensor in read(path).get("alteration", {}).get("changed_tensors", []) if tensor["name"] == event["tensor"] and tensor["relative_norm"] == event["relative_norm"]]
            assert len(matches) == 1
            assert event["shape"] == matches[0]["shape"]
            assert event["changed_elements"] == matches[0]["changed_elements"]
    return schedule, {str(path.relative_to(ROOT)): sha(path) for path in files}, expected


def record_climb(binary: Path, source: Path, evidence: Path, output: Path):
    schedule, sources, config_hash = projected_replay(source)
    replay = evidence / "replay"
    replay.mkdir()
    fifo = replay / "events.jsonl"
    os.mkfifo(fifo)
    (evidence / "replay-events.jsonl").write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for _, event in schedule))
    writer_errors = []

    def feed():
        try:
            with fifo.open("w") as destination:
                started = time.monotonic()
                for at, event in schedule:
                    time.sleep(max(0, started + at - time.monotonic()))
                    destination.write(json.dumps(event, ensure_ascii=False) + "\n")
                    destination.flush()
        except Exception as error:
            writer_errors.append(str(error))

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    rec = Recording(binary, ["replay", str(replay)], "will-climb", evidence)
    try:
        rec.stage("Measure the original model", "Hill climb", 2.0)
        rec.send("\x1b[H", 0.2)
        rec.drain(max(0, 7.1 - rec.elapsed))
        rec.stage("Retain the first weight edit", "LAYERS", 1.0)
        rec.send("\x1b[F", 0.2)
        rec.drain(max(0, 10.8 - rec.elapsed))
        rec.send("\x1b[D\x1b[D", 0.2)
        rec.stage("Inspect generated answers", "GENERATED", 1.0)
        rec.drain(max(0, 15.2 - rec.elapsed))
        rec.send("m", 0.2)
        rec.stage("Follow reference loss", "Reference loss", 1.7)
        rec.send("m", 0.2)
        rec.drain(max(0, 20.4 - rec.elapsed))
        rec.send("\t\t", 0.25)
        rec.send("\x1b[D", 0.25)
        rec.stage("Trace retained edits", "Genealogy", 3.1)
        rec.send("\t", 0.2)
        rec.send("\x1b[F\x1b[D\x1b[D", 0.3)
        rec.stage("Compare the final answer", "Salve!", 3.0)
        writer.join(timeout=2)
        if writer_errors:
            raise RuntimeError(writer_errors)
        result = rec.finish(output / "will-climb.cast")
        result.update(source_run=str(source.relative_to(ROOT)), source_sha256=sources, model_config_sha256=config_hash,
                      timing="Presentation timeline, 4.4 seconds per recorded turn", source_frontier="Codex", projected_fields="Legacy answer, proposal, model structure, preservation score and measured tensor-change receipts", display_normalisation={"output": "runs/greetings", "editor": "slash spacing", "health": "Recorded arithmetic preservation score, 1 example"})
        return result
    finally:
        if rec.child.isalive():
            rec.child.sendcontrol("c")
            rec.child.close(force=True)
        fifo.unlink(missing_ok=True)


def render(agg: Path, output: Path):
    from PIL import Image
    generated = {}
    for name, poster_time in (("setup", 10), ("will-climb", 26)):
        cast, gif = output / f"{name}.cast", output / f"{name}.gif"
        options = [str(agg), "--quiet", "--font-family", "DejaVu Sans Mono", "--font-size", "15", "--line-height", "1.25", "--fps-cap", "8", "--last-frame-duration", "1", "--idle-time-limit", "10"]
        subprocess.run([*options, str(cast), str(gif)], check=True)
        with Image.open(gif) as image:
            duration, frames, poster = 0, 0, None
            for frame in range(image.n_frames):
                image.seek(frame)
                duration += image.info.get("duration", 0)
                frames += 1
                if poster is None and duration >= poster_time * 1000:
                    poster = image.convert("RGB").copy()
            poster = poster or image.convert("RGB").copy()
            poster.save(output / f"{name}-poster.png")
            generated[name] = {"gif_bytes": gif.stat().st_size, "gif_sha256": sha(gif), "frames": frames, "duration_ms": duration, "dimensions": list(image.size)}
        if gif.stat().st_size > 5 * 1024 * 1024:
            raise RuntimeError(f"GIF exceeds 5 MiB: {gif}")
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/vespasian")
    parser.add_argument("--python", type=Path, default=Path(shutil.which("python3")))
    parser.add_argument("--source-run", type=Path, default=ROOT / "runs/qwen-greetings-001")
    parser.add_argument("--agg", type=Path, default=Path(shutil.which("agg") or ROOT / "runs/media-015/agg"))
    parser.add_argument("--only", choices=("setup", "will-climb", "render", "all"), default="all")
    args = parser.parse_args()
    for name in ("binary", "python", "source_run", "agg"):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    output = ROOT / "docs/media"
    output.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / "runs/media-015" / time.strftime("capture-%Y%m%d-%H%M%S")
    evidence.mkdir(parents=True)
    receipt = {"title": TITLE, "capture_script_sha256": sha(Path(__file__)), "binary_sha256": sha(args.binary), "version": subprocess.check_output([str(args.binary), "--version"], text=True).strip(), "frontier_calls": 0}
    if args.only in ("setup", "all"):
        receipt["setup"] = record_setup(args.binary, args.python, evidence, output)
    if args.only in ("will-climb", "all"):
        receipt["will-climb"] = record_climb(args.binary, args.source_run, evidence, output)
    if args.only in ("render", "all"):
        receipt["rendering"] = render(args.agg, output)
        receipt["renderer"] = {"version": subprocess.check_output([str(args.agg), "--version"], text=True).strip(), "sha256": sha(args.agg)}
    (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
