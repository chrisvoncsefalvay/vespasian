"""Vespasian's browser workbench. No inference, credentials or publication here.

``build_ui()`` returns ``(Blocks, controls)``. The caller wires preview/run/stop,
supplies ``CSS`` and ``THEME`` to Gradio 6's ``launch()``, and registers
``ASSET_DIR`` with ``gr.set_static_paths``. All controls use their dictionary
key as their identity; API credentials are password fields and are never read
by this module's rendering code.

``render_dashboard(state)`` consumes a JSON-serialisable public display state:

    {
        "events": [<unmodified CLI JSON events in chronological order>],
        "status": "idle" | "running" | "complete" | "stopped" | "error",
        "config": {"model": str, "goal": str, "budget": float,
                   "max_turns": int, "health_checks": list[str]},
        "elapsed": float,  # elapsed seconds, optional
        "error": str | None,
        "message": str,  # optional current scheduling/status message
    }

Supported events mirror ``src/ui.rs``: start, phase, progress, assessment,
baseline, model_structure, tensor_edit, answer, answer_score, proposal,
editor_activity, turn, cost, complete and error. Metrics are read from
``metrics.validation[metric]``; health checks from ``health[id].score``.
Missing measurements stay missing. No curves, model layers, answers or spend
are synthesised. Only named config fields are read; do not store credentials
in display state. All user-, dataset- and provider-supplied text is escaped.
"""

from __future__ import annotations

import html
import math
import re
from pathlib import Path
from typing import Any

import gradio as gr


ASSET_DIR = Path(__file__).resolve().parent / "assets"
CSS = (ASSET_DIR / "workbench.css").read_text(encoding="utf-8").replace(
    "__ASSETS__", f"/gradio_api/file={ASSET_DIR.as_posix()}"
)
THEME = gr.themes.Base(
    font=["IBM Plex Sans", "sans-serif"],
    font_mono=["IBM Plex Mono", "monospace"],
).set(
    color_accent="#d6a071",
    color_accent_soft="#3e342a",
    color_accent_soft_dark="#3e342a",
    slider_color="#d6a071",
    slider_color_dark="#d6a071",
    body_background_fill="#14181b",
    body_background_fill_dark="#14181b",
    body_text_color="#e9e1d1",
    body_text_color_dark="#e9e1d1",
    block_background_fill="#191e21",
    block_background_fill_dark="#191e21",
    block_border_color="#343b3c",
    block_border_color_dark="#343b3c",
    block_label_text_color="#c6c6bb",
    block_label_text_color_dark="#c6c6bb",
    input_background_fill="#14181b",
    input_background_fill_dark="#14181b",
    input_border_color="#434a49",
    input_border_color_dark="#434a49",
    input_placeholder_color="#7e8783",
    input_placeholder_color_dark="#7e8783",
    button_primary_background_fill="#d6a071",
    button_primary_background_fill_dark="#d6a071",
    button_primary_text_color="#14181b",
    button_primary_text_color_dark="#14181b",
    button_primary_border_color="#d6a071",
    button_primary_border_color_dark="#d6a071",
    button_secondary_background_fill="#191e21",
    button_secondary_background_fill_dark="#191e21",
    button_secondary_text_color="#e9e1d1",
    button_secondary_text_color_dark="#e9e1d1",
    button_secondary_border_color="#434a49",
    button_secondary_border_color_dark="#434a49",
    border_color_primary="#343b3c",
    border_color_primary_dark="#343b3c",
    block_radius="0px",
    button_large_radius="0px",
    button_small_radius="0px",
    input_radius="0px",
)

_MARK = '<svg viewBox="0 0 30 30" aria-hidden="true"><path d="M3 7 15 24 27 7M9 7l6 8 6-8" fill="none" stroke="currentColor" stroke-width="1.7"/></svg>'
HEADER = f"""<header class="vs-masthead">
  <a class="vs-wordmark" href="https://github.com/chrisvoncsefalvay/vespasian" target="_blank" rel="noopener noreferrer">{_MARK}<span>VESPASIAN</span></a>
  <span class="vs-masthead-rule"></span>
  <span class="vs-masthead-meta">WEIGHT SPACE EDITING ASSISTANT<span>/</span>v0.1.5<span>/</span>ZEROGPU</span>
  <a class="vs-docs" href="https://github.com/chrisvoncsefalvay/vespasian#readme" target="_blank" rel="noopener noreferrer">Documentation <span aria-hidden="true">↗</span></a>
</header>
<div class="vs-intro"><div><span class="vs-kicker">THE WORKBENCH</span><h1>A measured change.</h1></div><p>Describe the behaviour. Follow the edits.<br>Keep what improves.</p></div>"""

PREVIEW_EMPTY = """<div class="vs-preview-empty"><span class="vs-kicker">DATASET PREVIEW</span><p>Connect a dataset to inspect its fields and examples before starting.</p></div>"""


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _text(value: Any, limit: int | None = None) -> str:
    """Escape text, removing terminal and invisible directional controls."""
    if value is None:
        return ""
    raw = str(value)
    raw = re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|$))", "", raw)
    raw = "".join(c for c in raw if c in "\n\t" or (ord(c) >= 32 and not (127 <= ord(c) <= 159 or 0x202A <= ord(c) <= 0x202E or 0x2066 <= ord(c) <= 0x2069)))
    if limit is not None and len(raw) > limit:
        raw = raw[:limit] + "…"
    return html.escape(raw, quote=True)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _int(value: Any, default: int = 0) -> int:
    number = _number(value)
    return int(number) if number is not None and number >= 0 else default


def _score(metrics: Any, metric: str) -> float | None:
    return _number(_dict(_dict(metrics).get("validation")).get(metric))


def _health_score(value: Any) -> float | None:
    number = _number(_dict(value).get("score"))
    return number if number is not None and 0 <= number <= 1 else None


def _fmt(value: float | None, metric: str = "score") -> str:
    if value is None:
        return "—"
    return f"{value:.4f}" if metric == "loss" else f"{value * 100:.1f}%"


def _money(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"${number:.4f}" if 0 < number < 1 else f"${number:.2f}"


def _metric_name(metric: str) -> str:
    return {"auto": "Objective score", "exact_match": "Exact match", "token_f1": "Token F1", "semantic": "Goal score", "loss": "Reference loss"}.get(metric, metric.replace("_", " ").capitalize())


def _replay(state: dict) -> dict:
    config = _dict(state.get("config"))
    view: dict[str, Any] = {
        "model": config.get("model", ""), "goal": config.get("goal", ""),
        "budget_usd": config.get("budget", config.get("budget_usd")),
        "max_turns": _int(config.get("max_turns")), "metric": str(config.get("metric") or "auto"),
        "phase": "Ready when you are", "message": "Configure an experiment to begin.",
        "baseline": {}, "best": {}, "health_baseline": {}, "health_best": {},
        "turns": [], "changes": {}, "answers": [], "editor_calls": {},
        "retained_turn": 0, "structure": {}, "tensor": {}, "assessment": "", "plan": "",
        "current": 0, "total": 0, "spent_usd": None, "turn": 0,
        "input_tokens": 0, "output_tokens": 0, "status": "idle",
        "has_baseline": False,
    }
    for raw_event in _list(state.get("events")):
        event = _dict(raw_event)
        kind = event.get("type")
        if kind == "start":
            for key in ("model", "goal", "budget_usd", "max_turns", "editor", "budget_mode"):
                if key in event:
                    view[key] = event[key]
            view["status"] = "running"
        elif kind in ("phase", "progress"):
            for key in ("phase", "message"):
                if event.get(key):
                    view[key] = event[key]
            view["current"] = _int(event.get("current"))
            view["total"] = _int(event.get("total"))
        elif kind == "assessment":
            view["metric"] = str(event.get("metric") or view["metric"])
            view["assessment"] = event.get("reason", "")
            view["plan"] = event.get("plan", "")
            view["phase"] = "Dataset assessed" if event.get("suitable") else "Dataset needs attention"
        elif kind == "baseline":
            view["baseline"] = _dict(event.get("metrics"))
            view["best"] = view["baseline"]
            view["health_baseline"] = _dict(event.get("health"))
            view["health_best"] = view["health_baseline"]
            view["has_baseline"] = True
        elif kind == "model_structure":
            view["structure"] = event
        elif kind == "tensor_edit":
            view["tensor"] = event
        elif kind == "answer":
            if not isinstance(event.get("example"), dict):
                continue
            answer = dict(event)
            answer["example"] = dict(event["example"])
            example_id = answer["example"].get("id")
            duplicate = next((i for i, previous in enumerate(view["answers"]) if example_id is not None and previous.get("label") == event.get("label") and previous.get("turn") == event.get("turn") and previous["example"].get("id") == example_id), None)
            if duplicate is None:
                view["answers"].append(answer)
            else:
                view["answers"][duplicate] = answer
        elif kind == "answer_score":
            value = _number(event.get("semantic"))
            if value is not None and 0 <= value <= 1:
                for answer in view["answers"]:
                    if answer.get("label", "") == event.get("label", "") and answer["example"].get("id") == event.get("example_id"):
                        answer["example"]["semantic"] = value
                        answer["example"]["judgement"] = event.get("judgement", "")
        elif kind == "editor_activity":
            call_id = str(event.get("call_id", f"call-{len(view['editor_calls'])}"))
            call = view["editor_calls"].setdefault(call_id, {"updates": []})
            for key in ("provider", "model", "purpose", "status", "turn"):
                if key in event:
                    call[key] = event[key]
            # Only provider-supplied display summaries; no inferred reasoning.
            summary = event.get("summary") or event.get("text")
            if summary:
                call["updates"].append({"text": summary, "status": event.get("status", ""), "elapsed_ms": event.get("elapsed_ms")})
        elif kind in ("proposal", "turn"):
            turn = _int(event.get("turn"))
            if turn == 0:
                continue
            change = view["changes"].setdefault(turn, {"turn": turn, "parent_turn": view["retained_turn"], "accepted": None, "proposal": {}, "metrics": {}})
            for key in ("parent_turn", "proposal", "mutation", "metrics", "decision_reason", "rationale"):
                if key in event:
                    change[key] = event[key]
            if kind == "turn":
                accepted = event.get("accepted") is True
                change["accepted"] = accepted
                view["turn"] = turn
                view["max_turns"] = _int(event.get("max_turns"), _int(view["max_turns"]))
                view["turns"] = [previous for previous in view["turns"] if _int(previous.get("turn")) != turn]
                view["turns"].append(event)
                view["tensor"] = {}
                view["current"] = view["total"] = 0
                view["phase"] = f"Turn {turn} {'retained' if accepted else 'reverted'}"
                view["message"] = event.get("decision_reason", "")
                if accepted:
                    view["retained_turn"] = turn
                    view["best"] = _dict(event.get("metrics"))
                    if isinstance(event.get("health"), dict):
                        view["health_best"] = event["health"]
                if isinstance(event.get("best_health"), dict):
                    view["health_best"] = event["best_health"]
        elif kind == "cost":
            for key in ("spent_usd", "budget_usd", "input_tokens", "output_tokens", "budget_mode"):
                if key in event:
                    view[key] = event[key]
        elif kind == "complete":
            view["status"] = "complete"
            view["phase"] = "Experiment complete"
            view["message"] = event.get("message", "")
            if isinstance(event.get("metrics"), dict):
                view["best"] = event["metrics"]
            if isinstance(event.get("health"), dict):
                view["health_best"] = event["health"]
            view["current"] = view["total"] = 0
        elif kind == "error":
            view["status"] = "error"
            view["phase"] = "Run interrupted"
            view["message"] = event.get("message", "")
    if state.get("status"):
        view["status"] = state["status"]
    if state.get("message"):
        view["message"] = state["message"]
    if state.get("error"):
        view["status"] = "error"
        view["message"] = state["error"]
    if view["status"] in ("stopped", "cancelled", "stopping"):
        view["phase"] = "Stopping at the next boundary" if view["status"] == "stopping" else "Experiment stopped"
    view["turns"].sort(key=lambda row: _int(row.get("turn")))
    return view


def _curve(baseline: float | None, turns: list, get_value, *, label: str, metric: str = "score", compact: bool = False) -> str:
    """Draw actual measurements and the retained trajectory, never interpolation."""
    if baseline is None:
        if compact:
            return '<div class="vs-health-pending">Awaiting baseline measurement.</div>'
        return """<div class="vs-chart-empty"><div class="vs-empty-axis" aria-hidden="true"><span></span><span></span><span></span><i></i></div><div class="vs-chart-empty-copy"><span class="vs-empty-cross" aria-hidden="true">＋</span><h3>The baseline comes first.</h3><p>Your first measurement starts the chart.<br>Each candidate will leave a trace.</p></div><span class="vs-axis-label">EDIT TURN →</span></div>"""
    measurements = [(0, baseline, True)]
    for turn in turns:
        value = get_value(turn)
        if value is not None:
            measurements.append((_int(turn.get("turn")), value, turn.get("accepted") is True))
    last = max([_int(t.get("turn")) for t in turns] + [1])
    width, height = (440, 145) if compact else (920, 308)
    left, right, top, bottom = (44, 18, 16, 28) if compact else (58, 24, 24, 42)
    values = [p[1] for p in measurements]
    if metric == "loss" or any(v < 0 or v > 1 for v in values):
        spread = max(values) - min(values)
        padding = max(spread * 0.16, abs(max(values)) * 0.05, 0.001)
        lo, hi = min(values) - padding, max(values) + padding
        if min(values) >= 0:
            lo = max(0.0, lo)
    else:
        lo, hi = 0.0, 1.0
    x = lambda turn: left + turn / last * (width - left - right)
    y = lambda value: top + (hi - value) / (hi - lo) * (height - top - bottom)
    ticks = [lo, (lo + hi) / 2, hi]
    grid = "".join(f'<line x1="{left}" y1="{y(v):.2f}" x2="{width-right}" y2="{y(v):.2f}" class="vs-grid-line"/><text x="{left-12}" y="{y(v)+4:.2f}" text-anchor="end" class="vs-tick">{_fmt(v, metric)}</text>' for v in ticks)
    reference = f'<line x1="{left}" y1="{y(baseline):.2f}" x2="{width-right}" y2="{y(baseline):.2f}" class="vs-baseline-line"/>'
    retained_value = baseline
    path = f"M {x(0):.2f} {y(baseline):.2f}"
    for turn in turns:
        number = _int(turn.get("turn"))
        value = get_value(turn)
        if turn.get("accepted") is True and value is not None:
            retained_value = value
        path += f" L {x(number):.2f} {y(retained_value):.2f}"
    dots = ""
    for number, value, accepted in measurements:
        state_name = "baseline" if number == 0 else "retained" if accepted else "reverted"
        dots += f'<circle cx="{x(number):.2f}" cy="{y(value):.2f}" r="{3 if compact else 4.5}" class="vs-point vs-point-{state_name}"><title>{"Baseline" if number == 0 else f"Turn {number}: {state_name}"} · {_fmt(value, metric)}</title></circle>'
    xticks = sorted({0, last // 2, last}) if last >= 4 else [0, last]
    axis = "".join(f'<text x="{x(t):.2f}" y="{height-11}" text-anchor="middle" class="vs-tick">{t}</text>' for t in xticks)
    description = "; ".join(f'{"Baseline" if number == 0 else f"Turn {number}"} {_fmt(value, metric)}, {"retained" if accepted else "reverted"}' for number, value, accepted in measurements)
    return f'<svg class="vs-chart-svg {"vs-chart-compact" if compact else ""}" viewBox="0 0 {width} {height}" role="img" aria-label="{_text(label)}"><title>{_text(label)}</title><desc>{_text(description)}</desc>{grid}{reference}<path d="{path}" class="vs-retained-line"/>{dots}{axis}</svg>'


def _layers(view: dict) -> str:
    structure = view["structure"]
    count = _int(structure.get("num_layers"))
    retained: set[int] = set()
    queued: set[int] = set()

    def target_layers(proposal: dict) -> set[int]:
        found = set()
        for target in _list(_dict(proposal).get("targets")):
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(target))
            if match:
                found.add(int(match.group(1)))
        return found

    for change in view["changes"].values():
        if change["accepted"] is True:
            retained.update(target_layers(change.get("proposal")))
        elif change["accepted"] is None:
            queued.update(target_layers(change.get("proposal")))
    tensor = view["tensor"]
    active = target_layers({"targets": [tensor.get("tensor", "")]}) if tensor.get("status") == "started" else set()
    title = f'<div class="vs-section-heading"><h3>Layer activity</h3><span class="vs-kicker">{str(count) + " LAYERS" if count else "MODEL STRUCTURE"}</span></div>'
    if not 0 < count <= 4096:
        return f'<section class="vs-layers">{title}<p class="vs-empty-line">The model’s layer map appears after it loads.</p></section>'
    cells = []
    for layer in range(count):
        kind = "active" if layer in active else "proposed" if layer in queued else "retained" if layer in retained else "untouched"
        cells.append(f'<span class="vs-layer vs-layer-{kind}" title="Layer {layer}: {kind}" aria-label="Layer {layer}: {kind}">{layer:02d}</span>')
    legend = '<div class="vs-legend"><span><i class="vs-swatch vs-swatch-active"></i>Active</span><span><i class="vs-swatch vs-swatch-proposed"></i>Proposed</span><span><i class="vs-swatch vs-swatch-kept"></i>Retained</span><span><i class="vs-swatch"></i>Untouched</span></div>'
    detail = f'<p class="vs-tensor-name">{_text(tensor.get("status", ""))} · {_text(tensor.get("tensor", ""))}</p>' if tensor.get("tensor") else ""
    return f'<section class="vs-layers">{title}<div class="vs-layer-map">{"".join(cells)}</div>{legend}{detail}</section>'


def _answers(view: dict) -> str:
    answers = view["answers"]
    heading = f'<div class="vs-section-heading"><h3>Generated answers</h3><span class="vs-kicker">{len(answers):02d} OBSERVED</span></div>'
    if not answers:
        return f'<section class="vs-observations">{heading}<div class="vs-detail-empty"><span class="vs-quote-mark" aria-hidden="true">“</span><p>Read what the model actually says.<br>Answers appear as examples finish.</p></div></section>'
    entries = []
    for index, answer in reversed(list(enumerate(answers[-20:]))):
        example = answer["example"]
        label = answer.get("label") or example.get("split") or "Example"
        turn = answer.get("turn")
        label = f"{label} · turn {turn}" if turn is not None else str(label)
        ident = example.get("id", "")
        prompt = example.get("prompt")
        target = example.get("target")
        output = example.get("output", example.get("prediction", ""))
        parts = []
        if prompt:
            parts.append(f'<div class="vs-answer-part"><span class="vs-kicker">PROMPT</span><p>{_text(prompt)}</p></div>')
        if target is not None and example.get("mode") != "prompts":
            reference_label = "STYLE EXAMPLE" if example.get("mode") == "corpus" else "REFERENCE"
            parts.append(f'<div class="vs-answer-part vs-reference"><span class="vs-kicker">{reference_label}</span><p>{_text(target)}</p></div>')
        parts.append(f'<div class="vs-answer-part vs-generated"><span class="vs-kicker">GENERATED</span><p>{_text(output) or "<span class=vs-muted>No text returned.</span>"}</p></div>')
        measures = []
        if isinstance(example.get("passed"), bool):
            measures.append("Pass" if example["passed"] else "Fail")
        exact = example.get("exact_match")
        if isinstance(exact, bool):
            measures.append("Exact match" if exact else "Exact miss")
        for key, label_name in (("token_f1", "F1"), ("semantic", "Goal"), ("loss", "Loss")):
            value = _number(example.get(key))
            if value is not None:
                measures.append(f"{label_name} {_fmt(value, key)}")
        if example.get("finish_reason"):
            measures.append(str(example["finish_reason"]))
        if example.get("token_ids") is not None:
            measures.append(f'{len(_list(example.get("token_ids")))} tokens')
        if measures:
            parts.append(f'<div class="vs-answer-measures">{_text(" · ".join(measures))}</div>')
        if example.get("judgement"):
            parts.append(f'<p class="vs-judgement">{_text(example["judgement"])}</p>')
        if answer.get("truncated_fields") or example.get("truncated_fields"):
            parts.append('<p class="vs-display-note">Display text was shortened by the runtime. The run bundle contains the full evaluation output.</p>')
        opened = " open" if index == len(answers[-20:]) - 1 else ""
        entries.append(f'<details class="vs-answer"{opened}><summary><span>{_text(label)}</span><span class="vs-answer-id">{_text(ident, 30)} <b aria-hidden="true">＋</b></span></summary>{"".join(parts)}</details>')
    notice = f'<p class="vs-display-note">Showing the latest 20 of {len(answers)} answers. The complete record is in the run bundle.</p>' if len(answers) > 20 else ""
    return f'<section class="vs-observations">{heading}<div class="vs-answer-list">{"".join(entries)}</div>{notice}</section>'


def _editor(view: dict) -> str:
    changes = list(view["changes"].values())
    latest = changes[-1] if changes else {}
    proposal = _dict(latest.get("proposal"))
    rationale = proposal.get("rationale") or latest.get("rationale")
    body = ""
    if rationale:
        turn = _int(latest.get("turn"))
        operation = proposal.get("operation") or proposal.get("kind") or proposal.get("type") or "Proposal"
        body += f'<div class="vs-editor-rationale"><span class="vs-kicker">TURN {turn:02d} · {_text(operation)}</span><p>{_text(rationale)}</p></div>'
        targets = _list(proposal.get("targets"))
        if targets:
            body += '<div class="vs-targets">' + "".join(f'<code>{_text(target)}</code>' for target in targets) + '</div>'
        if latest.get("decision_reason"):
            body += f'<p class="vs-decision">{_text(latest["decision_reason"])}</p>'
    calls = list(view["editor_calls"].values())
    if calls:
        call = calls[-1]
        body += f'<div class="vs-editor-call"><span class="vs-kicker">{_text(call.get("purpose", "Editor activity"))} · {_text(call.get("status", ""))}</span><span class="vs-editor-model">{_text(call.get("model", ""))}</span>'
        updates = call.get("updates", [])
        if updates:
            body += f'<p>{_text(updates[-1]["text"])}</p>'
            if len(updates) > 1:
                body += '<details class="vs-provider-history"><summary>Earlier summaries</summary>' + "".join(f'<p><span>{_text(update["status"])}</span>{_text(update["text"])}</p>' for update in updates[:-1]) + '</details>'
        else:
            body += '<p class="vs-muted">Waiting for the editor’s response.</p>'
        body += '</div>'
    if view["assessment"] or view["plan"]:
        body += '<details class="vs-assessment"><summary>Dataset assessment &amp; evaluation plan</summary>'
        if view["assessment"]:
            body += f'<p>{_text(view["assessment"])}</p>'
        if view["plan"]:
            body += f'<p>{_text(view["plan"])}</p>'
        body += '</details>'
    if not body:
        body = '<div class="vs-detail-empty"><span class="vs-editor-mark" aria-hidden="true">↳</span><p>The editor’s proposal rationale and available summaries will appear here.</p></div>'
    return f'<section class="vs-editor"><div class="vs-section-heading"><h3>Editor’s notebook</h3><span class="vs-kicker">PROPOSE → MEASURE</span></div><div class="vs-editor-body">{body}</div></section>'


def _genealogy(view: dict) -> str:
    changes = sorted(view["changes"].values(), key=lambda row: _int(row.get("turn")))
    heading = '<div class="vs-section-heading"><h3>Edit genealogy</h3><span class="vs-kicker">EVERY CANDIDATE HAS A PARENT</span></div>'
    if not changes and not view["has_baseline"]:
        return f'<section class="vs-genealogy">{heading}<p class="vs-empty-line">Retained and reverted branches appear as candidates are evaluated.</p></section>'
    width = max(520, 70 + len(changes) * 90)
    positions = {0: (30, 40)}
    for index, change in enumerate(changes, 1):
        positions[_int(change.get("turn"))] = (30 + index * 90, 40 if change.get("accepted") is True else 92)
    paths, nodes, rows = [], [], []
    nodes.append('<circle cx="30" cy="40" r="5" class="vs-genealogy-baseline"/><text x="30" y="24" text-anchor="middle">BASE</text>')
    for change in changes:
        turn = _int(change.get("turn"))
        parent = _int(change.get("parent_turn"))
        accepted = change.get("accepted")
        status = "retained" if accepted is True else "reverted" if accepted is False else "proposed"
        x, y = positions[turn]
        parent_position = positions.get(parent)
        if parent_position is not None:
            px, py = parent_position
            paths.append(f'<path d="M {px} {py} C {px+38} {py}, {x-30} {y}, {x} {y}" class="vs-ancestry-edge vs-ancestry-{status}"/>')
        nodes.append(f'<circle cx="{x}" cy="{y}" r="5" class="vs-genealogy-{status}"><title>Turn {turn}: {status}; parent {parent}</title></circle><text x="{x}" y="{y-15}" text-anchor="middle">{turn:02d}</text>')
        rationale = _dict(change.get("proposal")).get("rationale") or change.get("rationale") or change.get("decision_reason") or "No rationale supplied."
        rows.append(f'<tr><td>{turn:02d} <span>← {parent:02d}</span></td><td><span class="vs-decision-label vs-decision-{status}">{status}</span></td><td>{_text(rationale)}</td></tr>')
    diagram = f'<div class="vs-genealogy-scroll"><svg viewBox="0 0 {width} 118" style="min-width:{width}px" role="img" aria-label="Candidate ancestry; each edge joins a candidate to its reported parent"><title>Edit genealogy</title>{"".join(paths)}{"".join(nodes)}</svg></div>'
    table = f'<details class="vs-genealogy-record"><summary>Read the edit record <span>{len(changes)} candidates</span></summary><div class="vs-table-scroll"><table><thead><tr><th>Turn/parent</th><th>Decision</th><th>Rationale</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></details>' if rows else '<p class="vs-empty-line">Baseline measured. Waiting for the first proposal.</p>'
    return f'<section class="vs-genealogy">{heading}{diagram}{table}</section>'


def render_dashboard(state: dict | None = None) -> str:
    """Render real CLI events as a safe, self-contained HTML/SVG dashboard."""
    state = _dict(state)
    view = _replay(state)
    metric = view["metric"]
    before, after = _score(view["baseline"], metric), _score(view["best"], metric)
    status = str(view["status"])
    status_class = status if status in ("idle", "running", "complete", "stopped", "error", "stopping", "queued") else "running"
    display_status = {"idle": "NOT STARTED", "running": "IN PROGRESS", "complete": "COMPLETE", "stopped": "STOPPED", "error": "INTERRUPTED", "stopping": "STOPPING", "queued": "WAITING FOR GPU"}.get(status, status.replace("_", " ").upper())
    elapsed = _number(state.get("elapsed", state.get("elapsed_seconds")))
    elapsed_text = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}" if elapsed is not None and elapsed >= 0 else "—"
    accepted_count = sum(change.get("accepted") is True for change in view["changes"].values())
    completed_turns = len(view["turns"])
    turn_text = f'{completed_turns:02d}<span>/{_int(view["max_turns"]):02d}</span>' if _int(view["max_turns"]) else f"{completed_turns:02d}"
    budget_text = _money(view.get("budget_usd"))
    budget_caption = "estimated frontier spend" if "estimated" in str(view.get("budget_mode", "")) else "frontier spend"
    progress = ""
    if view["total"] > 0:
        proportion = min(100, max(0, view["current"] / view["total"] * 100))
        progress = f'<div class="vs-progress"><span>{view["current"]}/{view["total"]}</span><div role="progressbar" aria-label="Current operation" aria-valuemin="0" aria-valuemax="{view["total"]}" aria-valuenow="{min(view["current"], view["total"])}"><i style="width:{proportion:.2f}%"></i></div></div>'
    message = f'<p class="vs-run-message">{_text(view["message"])}</p>' if view["message"] else ""
    goal = f'<p class="vs-current-goal"><span>OBJECTIVE</span>{_text(view["goal"])}</p>' if view["goal"] else ""
    chart = _curve(before, view["turns"], lambda turn: _score(turn.get("metrics"), metric), label=f"Validation {_metric_name(metric)} by edit turn", metric=metric)
    delta = ""
    if before is not None and after is not None:
        difference = after - before
        delta_text = f"{difference:+.4f}" if metric == "loss" else f"{difference*100:+.1f} pp"
        delta = f'<span class="vs-delta">{delta_text}</span>'
    measurements = ""
    if before is not None:
        rows = f'<tr><td>Baseline</td><td>{_fmt(before, metric)}</td><td>Reference</td></tr>'
        for turn in view["turns"]:
            value = _score(turn.get("metrics"), metric)
            rows += f'<tr><td>{_int(turn.get("turn")):02d}</td><td>{_fmt(value, metric)}</td><td>{"Retained" if turn.get("accepted") is True else "Reverted"}</td></tr>'
        measurements = f'<details class="vs-measurements"><summary>Measured values</summary><table><thead><tr><th>Turn</th><th>{_text(_metric_name(metric))}</th><th>Decision</th></tr></thead><tbody>{rows}</tbody></table></details>'
    health = ""
    for check_id, check in view["health_baseline"].items():
        baseline_health = _health_score(check)
        if baseline_health is None:
            continue
        label = str(_dict(check).get("label") or check_id)
        best_health = _health_score(view["health_best"].get(check_id))
        curve = _curve(baseline_health, view["turns"], lambda turn, key=check_id: _health_score(_dict(turn.get("health")).get(key)), label=f"{label}: retained health score by turn", compact=True)
        health += f'<section class="vs-health-chart"><div><h4>{_text(label)}</h4><span>{_fmt(best_health)}</span></div>{curve}<p>Baseline {_fmt(baseline_health)} <span>· retained {_fmt(best_health)}</span></p></section>'
    health_section = f'<section class="vs-health"><div class="vs-section-heading"><h3>Preservation checks</h3><span class="vs-kicker">SAME BASELINE, EVERY TURN</span></div><div class="vs-health-grid">{health}</div></section>' if health else ""
    model = str(view["model"]).split("/")[-1] if view["model"] else "No model loaded"
    return f"""<main class="vs-dashboard" aria-label="Experiment dashboard">
      <div class="vs-run-heading"><div><span class="vs-kicker">EXPERIMENT MONITOR</span><h2>The hill climb<span class="vs-title-period">.</span></h2></div><div class="vs-run-state vs-state-{status_class}"><span><i aria-hidden="true"></i>{_text(display_status)}</span><small>{_text(model)}</small></div></div>
      {goal}
      <div class="vs-phase-line"><span>{_text(str(view["phase"]).replace("_", " "))}</span>{progress}</div>{message}
      <section class="vs-objective"><div class="vs-chart-heading"><div><span class="vs-kicker">VALIDATION OBJECTIVE</span><h3>{_text(_metric_name(metric))} <span>{"↓" if metric == "loss" else "↑"}</span></h3></div><div class="vs-score-comparison"><div><span>BASELINE</span><strong>{_fmt(before, metric)}</strong></div><b aria-hidden="true">→</b><div><span>RETAINED</span><strong>{_fmt(after, metric)}</strong></div>{delta}</div></div>{chart}<div class="vs-chart-footer"><div class="vs-legend"><span><i class="vs-line-swatch"></i>Retained path</span><span><i class="vs-dot-swatch"></i>Reverted trial</span><span><i class="vs-line-swatch vs-baseline-swatch"></i>Baseline</span></div><span class="vs-kicker">EDIT TURN</span></div>{measurements}</section>
      <div class="vs-telemetry"><div><span class="vs-kicker">TURNS</span><strong>{turn_text}</strong></div><div><span class="vs-kicker">RETAINED EDITS</span><strong>{accepted_count:02d}</strong></div><div><span class="vs-kicker">ELAPSED</span><strong>{elapsed_text}</strong></div><div><span class="vs-kicker">{budget_caption.upper()}</span><strong>{_money(view["spent_usd"])}<span>/{budget_text}</span></strong></div></div>
      {health_section}{_layers(view)}
      <div class="vs-observation-grid">{_answers(view)}{_editor(view)}</div>
      {_genealogy(view)}
      <div class="vs-dashboard-footnote"><span>PROPOSE · MEASURE · RETAIN</span><span>All plotted points come from this run.</span></div>
    </main>"""


def build_ui() -> tuple[gr.Blocks, dict[str, Any]]:
    """Build the workbench and return named controls for the runtime to wire.

    Gradio 6 launch: ``demo.launch(css=CSS, theme=THEME, ...)``.
    ``dataset_source`` is a Hub identifier/URL; an uploaded file takes priority.
    ``health_checks`` returns CLI check names; ``health_max_drop`` is a fraction.
    The only callbacks here change local visibility or advance setup tabs.
    """
    controls: dict[str, Any] = {}
    with gr.Blocks(title="Vespasian · Weight Space Editing Assistant", fill_width=True, delete_cache=(600, 7200)) as demo:
        gr.HTML(HEADER, elem_id="vs-header")
        with gr.Row(elem_id="vs-workbench", equal_height=False):
            with gr.Column(scale=3, min_width=310, elem_id="vs-setup"):
                gr.HTML('<div class="vs-setup-title"><span class="vs-kicker">EXPERIMENT SETUP</span><span class="vs-setup-index">01—03</span></div>')
                with gr.Column(elem_id="vs-account"):
                    gr.HTML('<div class="vs-account-heading"><span class="vs-kicker">YOUR HUGGING FACE ACCOUNT</span><span>Connect once.</span></div>')
                    controls["login_button"] = gr.LoginButton("Sign in with Hugging Face", elem_id="vs-login")
                    controls["profile_html"] = gr.HTML('<div class="vs-profile"><p>Use your datasets and save edited models to your Hub account.</p></div>', elem_id="vs-profile")
                with gr.Tabs(elem_id="vs-setup-tabs", selected="target") as setup_tabs:
                    with gr.Tab("01  Data", id="target"):
                        gr.HTML('<div class="vs-step-intro"><h2>Choose your material.</h2><p>Start with a model and the examples that define its new behaviour.</p></div>')
                        controls["model"] = gr.Dropdown(
                            choices=[("Qwen 3.5 · 0.8B", "Qwen/Qwen3.5-0.8B"), ("Gemma 3 · 1B", "google/gemma-3-1b-it")],
                            value="Qwen/Qwen3.5-0.8B", label="Base model", elem_id="vs-model",
                        )
                        controls["dataset_location"] = gr.Radio(choices=[("From Hugging Face", "hub"), ("Upload a file", "file")], value="hub", label="Dataset location", elem_id="vs-dataset-location")
                        controls["dataset_source"] = gr.Dropdown(choices=[], allow_custom_value=True, label="Your dataset or a public repository", info="Choose a repository or type organisation/dataset.", elem_id="vs-dataset-source")
                        controls["refresh_datasets"] = gr.Button("↻  Refresh my datasets", elem_id="vs-refresh-datasets")
                        controls["dataset_file"] = gr.File(label="Dataset file · JSONL or text", file_types=[".jsonl", ".txt"], type="filepath", visible=False, elem_id="vs-dataset-file")
                        with gr.Row():
                            controls["dataset_config"] = gr.Dropdown(choices=[], allow_custom_value=True, label="Configuration", scale=1, min_width=100)
                            controls["dataset_split"] = gr.Dropdown(choices=["train", "validation", "test"], value="train", allow_custom_value=True, label="Source split", scale=1, min_width=100)
                        controls["dataset_mode"] = gr.Dropdown(
                            choices=[("Prompt + reference pairs", "paired"), ("Text corpus/style examples", "corpus"), ("Prompts only", "prompts")],
                            value="paired", label="How to use the examples",
                        )
                        with gr.Accordion("Field mapping & sample limit", open=False, elem_id="vs-fields"):
                            controls["prompt_field"] = gr.Textbox(label="Prompt field", placeholder="Auto-detect", info="Blank detects the field. Nested selectors work, e.g. messages[0].content.")
                            controls["target_field"] = gr.Textbox(label="Reference field", placeholder="Auto-detect")
                            controls["text_field"] = gr.Textbox(label="Text field", placeholder="Auto-detect", visible=False)
                            controls["example_limit"] = gr.Slider(5, 128, value=16, step=1, label="Example limit", info="Small datasets leave more GPU time for candidate edits.")
                        with gr.Accordion("Held-out test data · optional", open=False):
                            controls["test_dataset_source"] = gr.Dropdown(choices=[], allow_custom_value=True, label="Test dataset repository", info="Use the same field mapping as above. Leave blank to use the run’s standard split.")
                            with gr.Row():
                                controls["test_dataset_config"] = gr.Dropdown(choices=[], allow_custom_value=True, label="Test configuration", min_width=100)
                                controls["test_dataset_split"] = gr.Dropdown(choices=["train", "validation", "test"], value="test", allow_custom_value=True, label="Test source split", min_width=100)
                            controls["test_example_limit"] = gr.Slider(1, 128, value=8, step=1, label="Test example limit")
                            controls["test_preview_button"] = gr.Button("Inspect test dataset  ↗", variant="secondary")
                        controls["preview_button"] = gr.Button("Inspect dataset  ↗", variant="secondary", elem_id="vs-preview-button")
                        controls["preview_html"] = gr.HTML(PREVIEW_EMPTY, elem_id="vs-preview")
                        next_behaviour = gr.Button("Next · define the behaviour  →", elem_classes=["vs-step-next"])
                    with gr.Tab("02  Behaviour", id="behaviour"):
                        gr.HTML('<div class="vs-step-intro"><h2>Make the change precise.</h2><p>Describe what should improve and what the model should preserve.</p></div>')
                        controls["target_type"] = gr.Dropdown(choices=[("Factual knowledge", "factual"), ("Writing style", "style"), ("Output format", "format"), ("Classification", "classification"), ("Custom behaviour", "custom")], value="factual", label="Type of change")
                        controls["goal"] = gr.Textbox(label="Editing objective", placeholder="Answer each prompt using the reference answer in target.", lines=5, elem_id="vs-goal")
                        controls["controls"] = gr.Textbox(label="Behaviour to preserve", placeholder="One requirement per line, e.g. answer in the same language as the question.", lines=3, info="Optional. These are checked alongside the objective.")
                        controls["health_checks"] = gr.Dropdown(choices=[("Copying", "copying"), ("Arithmetic", "arithmetic"), ("JSON formatting", "json"), ("MMLU · anatomy", "mmlu:anatomy"), ("MMLU · high school mathematics", "mmlu:high_school_mathematics")], value=["copying"], multiselect=True, allow_custom_value=True, label="Preservation checks", info="Optional. Add another MMLU subject as mmlu:subject.", elem_id="vs-health-checks")
                        with gr.Accordion("Evaluation settings", open=False):
                            controls["metric"] = gr.Dropdown(choices=[("Let the editor select", "auto"), ("Exact match", "exact_match"), ("Token F1", "token_f1"), ("Goal score", "semantic")], value="auto", label="Objective metric")
                            controls["max_new_tokens"] = gr.Slider(16, 128, value=64, step=16, label="Maximum generated tokens")
                            controls["health_count"] = gr.Slider(1, 16, value=4, step=1, label="Examples per preservation check")
                            controls["health_max_drop"] = gr.Number(value=None, minimum=0, maximum=1, step=0.01, label="Maximum health score drop · optional", info="Blank tracks scores without rejecting edits. 0.10 sets a 10 percentage point limit.")
                        next_editor = gr.Button("Next · connect the editor  →", elem_classes=["vs-step-next"])
                    with gr.Tab("03  Run", id="run"):
                        gr.HTML('<div class="vs-step-intro"><h2>Set the boundaries.</h2><p>Your frontier editor proposes each change. The run measures whether to keep it.</p></div>')
                        controls["editor"] = gr.Radio(choices=[("OpenAI", "openai"), ("Anthropic", "anthropic")], value="openai", label="Editor provider")
                        controls["editor_model"] = gr.Textbox(value="gpt-6.1-sol", label="Editor model", info="Use a model available to your API account.")
                        controls["api_key"] = gr.Textbox(type="password", label="Editor API key", placeholder="Your provider’s API key", elem_id="vs-api-key")
                        with gr.Row():
                            controls["budget"] = gr.Number(value=1.0, minimum=0.01, maximum=100, step=0.25, label="Editor budget · USD", scale=1, min_width=110)
                            controls["max_turns"] = gr.Number(value=5, minimum=1, maximum=50, step=1, precision=0, label="Maximum edit turns", scale=1, min_width=110)
                        gr.HTML('<p class="vs-inline-note">The editor budget covers frontier API calls. Hugging Face GPU quota is accounted for separately.</p>')
                        with gr.Accordion("Access & run records", open=False):
                            controls["hf_token"] = gr.Textbox(type="password", label="Hugging Face token override", placeholder="Optional · use your connected account by default", info="Only needed if you want to use a different access token.")
                            controls["save_trajectory"] = gr.Checkbox(value=True, label="Include the edit trajectory in the run bundle")
                        with gr.Accordion("Save to your Hugging Face account", open=False, elem_id="vs-publish"):
                            gr.HTML('<p class="vs-inline-note">Publish after the run completes. Choose an explicit destination for each artifact.</p>')
                            controls["model_repo"] = gr.Textbox(label="Model repository", placeholder="your-account/edited-model")
                            controls["model_visibility"] = gr.Radio(choices=[("Private", "private"), ("Public", "public")], value="private", label="Model visibility")
                            controls["model_create"] = gr.Checkbox(value=True, label="Create the model repository if needed")
                            controls["publish_model_button"] = gr.Button("Save edited model to Hub  ↗", interactive=False)
                            controls["trajectory_repo"] = gr.Textbox(label="Trajectory dataset repository", placeholder="your-account/edit-trajectories")
                            controls["trajectory_visibility"] = gr.Radio(choices=[("Private", "private"), ("Public", "public")], value="private", label="Trajectory visibility")
                            controls["trajectory_create"] = gr.Checkbox(value=True, label="Create the trajectory repository if needed")
                            controls["publish_trajectory_button"] = gr.Button("Save trajectory to Hub  ↗", interactive=False)
                        gr.HTML('<div class="vs-run-note"><span class="vs-kicker">BEFORE YOU START</span><p>A run makes billable calls using your editor key. Your dataset examples and objective are sent to that provider.</p></div>')
                with gr.Row(elem_id="vs-run-actions"):
                    controls["run_button"] = gr.Button("Start experiment  →", variant="primary", scale=3, elem_id="vs-run-button")
                    controls["stop_button"] = gr.Button("Stop", variant="secondary", scale=1, min_width=65, interactive=False, elem_id="vs-stop-button")
                controls["status"] = gr.Markdown("Ready to configure.", elem_id="vs-status", container=False)
                controls["download"] = gr.File(label="Download run bundle", interactive=False, visible=False, elem_id="vs-download")
                gr.HTML('<p class="vs-download-note">Run bundles are available for download for two hours.</p>')
                gr.HTML('<div class="vs-setup-footer"><span class="vs-kicker">THE METHOD</span><p>Each candidate is compared with the retained checkpoint. Edits that fail the selection checks are reverted.</p></div>')
            with gr.Column(scale=8, min_width=440, elem_id="vs-monitor"):
                controls["dashboard_html"] = gr.HTML(render_dashboard(), elem_id="vs-dashboard-output")
        gr.HTML('<footer class="vs-page-footer"><span>VESPASIAN <span class="vs-footer-slash">/</span> THE WEIGHT SPACE EDITING ASSISTANT</span><a href="https://github.com/chrisvoncsefalvay/vespasian" target="_blank" rel="noopener noreferrer">Source code ↗</a></footer>')
        controls["state"] = gr.State({"events": [], "status": "idle", "config": {}})

        # Explicitly editable even before the caller attaches its run callbacks.
        for component in controls.values():
            if isinstance(component, (gr.Textbox, gr.Dropdown, gr.Radio, gr.Number, gr.Slider, gr.Checkbox, gr.CheckboxGroup)):
                component.interactive = True

        def location_visibility(location):
            return gr.update(visible=location == "hub"), gr.update(visible=location == "file"), gr.update(visible=location == "hub"), gr.update(visible=location == "hub")

        controls["dataset_location"].change(location_visibility, inputs=controls["dataset_location"], outputs=[controls["dataset_source"], controls["dataset_file"], controls["dataset_config"], controls["dataset_split"]], queue=False, api_name=False)
        controls["dataset_mode"].change(lambda mode: (gr.update(visible=mode != "corpus"), gr.update(visible=mode == "paired"), gr.update(visible=mode == "corpus")), inputs=controls["dataset_mode"], outputs=[controls["prompt_field"], controls["target_field"], controls["text_field"]], queue=False, api_name=False)
        controls["editor"].change(lambda provider: "gpt-6.1-sol" if provider == "openai" else "claude-sonnet-5-5", inputs=controls["editor"], outputs=controls["editor_model"], queue=False, api_name=False)
        def objective_hint(target_type, mode, reference_field, text_field):
            field = (text_field or "text") if mode == "corpus" else (reference_field or "target")
            if mode == "prompts":
                hint = "State how a successful answer to each prompt should behave."
            elif mode == "corpus" or target_type == "style":
                hint = f"Match the tone and writing style of the examples in {field}, while writing original content."
            elif target_type == "format":
                hint = f"Follow the output format shown in {field} for every answer."
            elif target_type == "classification":
                hint = f"Return the correct class from {field} for each input."
            elif target_type == "custom":
                hint = f"Use the examples in {field} to describe the specific change you want."
            else:
                hint = f"Answer each prompt using the reference answer in {field}."
            return gr.update(placeholder=hint)

        for component in ("target_type", "dataset_mode", "target_field", "text_field"):
            controls[component].change(objective_hint, inputs=[controls["target_type"], controls["dataset_mode"], controls["target_field"], controls["text_field"]], outputs=controls["goal"], queue=False, api_name=False)
        next_behaviour.click(lambda: gr.update(selected="behaviour"), outputs=setup_tabs, queue=False, api_name=False)
        next_editor.click(lambda: gr.update(selected="run"), outputs=setup_tabs, queue=False, api_name=False)
    return demo, controls
