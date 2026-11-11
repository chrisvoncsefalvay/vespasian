"""Render measured experiment events for the browser workbench."""

from __future__ import annotations

import html
import hashlib
import json
import math
import re
from typing import Any

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
        "phase": "Not started", "message": "Configure an experiment to begin.",
        "baseline": {}, "best": {}, "health_baseline": {}, "health_best": {},
        "turns": [], "changes": {}, "answers": [], "editor_calls": {},
        "retained_turn": 0, "structure": {}, "tensor": {}, "assessment": "", "plan": "",
        "current": 0, "total": 0, "spent_usd": None, "turn": 0,
        "input_tokens": 0, "output_tokens": 0, "status": "idle",
        "has_baseline": False,
        "health_max_drop": config.get("health_max_drop"),
        "report": _dict(state.get("report")),
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
            for key in ("parent_turn", "proposal", "mutation", "metrics", "health", "health_passed", "decision_reason", "rationale"):
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
        status = str(state["status"]).strip().lower()
        if status in {"complete", "completed"}:
            view["status"] = "complete"
        elif status in {"stopped", "cancelled"}:
            view["status"] = "stopped"
        elif status == "run stopped":
            view["status"] = "error" if state.get("error") else "stopped"
        elif status == "waiting for zerogpu":
            view["status"] = "queued"
        else:
            view["status"] = status if status in {"idle", "running", "error", "stopping", "queued"} else "running"
    if state.get("message"):
        view["message"] = state["message"]
    if state.get("error"):
        if view["status"] != "stopped":
            view["status"] = "error"
        view["message"] = state["error"]
    if view["status"] == "error":
        view["phase"] = "Run interrupted"
    elif view["status"] == "complete":
        view["phase"] = "Experiment complete"
    if view["status"] in ("stopped", "cancelled", "stopping"):
        view["phase"] = "Stopping at the next boundary" if view["status"] == "stopping" else "Experiment stopped"
    view["turns"].sort(key=lambda row: _int(row.get("turn")))
    report = view["report"]
    if report:
        if report.get("metric"):
            view["metric"] = str(report["metric"])
        for target, source in (("baseline", "baseline_metrics"), ("best", "best_metrics"),
                               ("health_baseline", "baseline_health"), ("health_best", "best_health")):
            if isinstance(report.get(source), dict):
                view[target] = report[source]
        if "health_max_drop" in report:
            view["health_max_drop"] = report["health_max_drop"]
        if isinstance(report.get("best_turn"), int):
            view["retained_turn"] = report["best_turn"]
    return view


PAGE_SIZE = 6
SPLITS = (("all", "Objective examples"), ("edit", "Edit"), ("validation", "Validation"),
          ("test", "Held-out test"), ("health", "Preservation checks"))
OUTCOMES = (("all", "All outcomes"), ("improved", "Improved score"),
            ("regressed", "Lower score"), ("changed", "Changed answer"))


def normalise_selection(view: dict, selection: dict | None = None) -> dict:
    selected = _dict(selection)
    latest = max([0, *view["changes"]])
    follow = selected.get("follow", True) is not False
    turn = latest if follow or selected.get("turn") is None else _int(selected.get("turn"))
    if turn != 0 and turn not in view["changes"]:
        turn = latest
    split = selected.get("split", "all")
    outcome = selected.get("outcome", "all")
    return {"turn": turn, "follow": follow,
            "split": split if split in dict(SPLITS) else "all",
            "outcome": outcome if outcome in dict(OUTCOMES) else "all",
            "page": _int(selected.get("page"))}


def _turn_name(turn: int) -> str:
    return "Baseline" if turn == 0 else f"Turn {turn}"


def _turn_button(turn: int, selected: int, label: str | None = None) -> str:
    return (f'<button type="button" class="vs-turn-button{" is-selected" if turn == selected else ""}" '
            f'data-vs-turn="{turn}" aria-pressed="{str(turn == selected).lower()}">{_text(label or _turn_name(turn))}</button>')


def _curve(baseline: float | None, turns: list, get_value, *, label: str, metric: str = "score",
           compact: bool = False, selected: int = 0, threshold: float | None = None) -> str:
    """Draw measured candidates and the retained path with selectable points."""
    if baseline is None:
        return '<div class="vs-chart-empty"><h3>Waiting for the baseline</h3><p>The first evaluation establishes the score every edit will be compared with.</p></div>'
    measurements = [(0, baseline, True)]
    for turn in turns:
        value = get_value(turn)
        if value is not None:
            measurements.append((_int(turn.get("turn")), value, turn.get("accepted") is True))
    last = max([_int(t.get("turn")) for t in turns] + [1])
    width, height = (440, 180) if compact else (920, 310)
    left, right, top, bottom = (70, 24, 24, 36) if compact else (140, 30, 40, 40)
    values = [p[1] for p in measurements]
    if metric == "loss" or any(v < 0 or v > 1 for v in values):
        padding = max((max(values) - min(values)) * .16, abs(max(values)) * .05, .001)
        lo, hi = min(values) - padding, max(values) + padding
        if min(values) >= 0:
            lo = max(0.0, lo)
    else:
        lo, hi = 0.0, 1.0
    x = lambda turn: left + turn / last * (width - left - right)
    y = lambda value: top + (hi - value) / (hi - lo) * (height - top - bottom)
    grid = "".join(f'<line x1="{left}" y1="{y(v):.2f}" x2="{width-right}" y2="{y(v):.2f}" class="vs-grid-line"/><text x="{left-12}" y="{y(v)+4:.2f}" text-anchor="end" class="vs-tick">{_fmt(v, metric)}</text>' for v in (lo, (lo+hi)/2, hi))
    reference = f'<line x1="{left}" y1="{y(baseline):.2f}" x2="{width-right}" y2="{y(baseline):.2f}" class="vs-baseline-line"/>'
    if threshold is not None and lo <= threshold <= hi:
        reference += f'<line x1="{left}" y1="{y(threshold):.2f}" x2="{width-right}" y2="{y(threshold):.2f}" class="vs-threshold-line"><title>Minimum permitted score {_fmt(threshold)}</title></line>'
    retained = baseline
    path = f"M {x(0):.2f} {y(baseline):.2f}"
    for turn in turns:
        value = get_value(turn)
        if turn.get("accepted") is True and value is not None:
            retained = value
        path += f' L {x(_int(turn.get("turn"))):.2f} {y(retained):.2f}'
    dots = []
    for number, value, accepted in measurements:
        kind = "baseline" if number == 0 else "retained" if accepted else "reverted"
        title = f"{_turn_name(number)}, {kind}, {_fmt(value, metric)}"
        dots.append(f'<g class="vs-chart-point{" is-selected" if number == selected else ""}" data-vs-turn="{number}" role="button" tabindex="0" aria-label="{_text(title)}" aria-pressed="{str(number == selected).lower()}"><title>{_text(title)}</title><circle cx="{x(number):.2f}" cy="{y(value):.2f}" r="18" class="vs-point-hit" fill="transparent"/><circle cx="{x(number):.2f}" cy="{y(value):.2f}" r="{6 if number == selected else 4}" class="vs-point vs-point-{kind}"/></g>')
    xticks = sorted({0, last // 2, last}) if last >= 4 else [0, last]
    axis = "".join(f'<text x="{x(t):.2f}" y="{height-10}" text-anchor="middle" class="vs-tick">{t}</text>' for t in xticks)
    description = ". ".join(f'{_turn_name(number)} {_fmt(value, metric)}, {"retained" if accepted else "reverted"}' for number, value, accepted in measurements)
    return f'<svg class="vs-chart-svg {"vs-chart-compact" if compact else ""}" viewBox="0 0 {width} {height}" role="group" aria-label="{_text(label)}"><title>{_text(label)}</title><desc>{_text(description)}</desc>{grid}{reference}<path d="{path}" class="vs-retained-line"/>{"".join(dots)}{axis}</svg>'


def _metrics_for(view: dict, turn: int) -> dict:
    return view["baseline"] if turn == 0 else _dict(view["changes"].get(turn, {}).get("metrics"))


def _health_for(view: dict, turn: int) -> dict:
    return view["health_baseline"] if turn == 0 else _dict(view["changes"].get(turn, {}).get("health"))


def _difference(before: float | None, after: float | None, metric: str = "score") -> str:
    if before is None or after is None:
        return "—"
    difference = after - before
    return f"{difference:+.4f}" if metric == "loss" else f"{difference*100:+.1f} pp"


def _turn_inspector(view: dict, selection: dict) -> str:
    turn = selection["turn"]
    metric = view["metric"]
    if turn == 0:
        count = _dict(view["baseline"].get("validation")).get("count")
        note = f"Measured on {_int(count)} validation examples." if _number(count) is not None else "The original checkpoint is the reference for this experiment."
        return f'<section class="vs-turn-inspector" aria-label="Selected turn"><div class="vs-section-heading"><h3>Baseline</h3><span class="vs-kicker">ORIGINAL CHECKPOINT</span></div><p>{note}</p></section>'
    change = view["changes"].get(turn, {})
    parent = _int(change.get("parent_turn"))
    proposal = _dict(change.get("proposal"))
    mutation = _dict(change.get("mutation"))
    accepted = change.get("accepted")
    decision = "Retained" if accepted is True else "Reverted" if accepted is False else "Evaluating"
    before = _metrics_for(view, parent)
    after = _dict(change.get("metrics"))
    rows = []
    for key in dict.fromkeys((metric, "loss")):
        a, b = _score(before, key), _score(after, key)
        if a is not None or b is not None:
            rows.append(f'<tr><th>{_text(_metric_name(key))}</th><td>{_fmt(a, key)}</td><td>{_fmt(b, key)}</td><td>{_difference(a, b, key)}</td></tr>')
    operation = mutation.get("operation") or proposal.get("operation") or "Awaiting proposal"
    settings = []
    for key in ("strength", "rank", "seed"):
        value = mutation.get(key, proposal.get(key))
        if value is not None:
            settings.append(f"{key.capitalize()} {_text(value)}")
    duration = _number(mutation.get("elapsed_seconds"))
    if duration is not None:
        settings.append(f"Tensor update {duration:.3f}s")
    reason = change.get("decision_reason") or "The candidate is being evaluated against its parent checkpoint."
    rationale = proposal.get("rationale") or change.get("rationale")
    targets = _list(mutation.get("changed_tensors"))
    if targets:
        target_rows = "".join(f'<tr><td><code>{_text(_dict(item).get("name"))}</code></td><td>{_text(" × ".join(str(v) for v in _list(_dict(item).get("shape"))))}</td><td>{_fmt(_number(_dict(item).get("relative_norm")))}</td></tr>' for item in targets)
        tensors = f'<details class="vs-selected-tensors" data-vs-details="tensors-{turn}"><summary>{len(targets)} edited tensors</summary><div class="vs-table-scroll"><table><thead><tr><th>Tensor</th><th>Shape</th><th>Relative change</th></tr></thead><tbody>{target_rows}</tbody></table></div></details>'
    else:
        tensors = '<div class="vs-targets">' + "".join(f'<code>{_text(target)}</code>' for target in _list(proposal.get("targets"))) + '</div>'
    health_rows = []
    for key, current in _health_for(view, turn).items():
        previous = _dict(_health_for(view, parent).get(key))
        a, b = _health_score(previous), _health_score(current)
        health_rows.append(f'<tr><th>{_text(_dict(current).get("label") or key)}</th><td>{_fmt(a)}</td><td>{_fmt(b)}</td><td>{_difference(a,b)}</td></tr>')
    health_table = f'<div class="vs-table-scroll"><table><caption>Preservation scores</caption><thead><tr><th>Check</th><th>Parent</th><th>Candidate</th><th>Change</th></tr></thead><tbody>{"".join(health_rows)}</tbody></table></div>' if health_rows else ""
    health_verdict = ""
    if isinstance(change.get("health_passed"), bool) and view["health_max_drop"] is not None:
        health_verdict = '<p class="vs-health-verdict">' + ("Preservation limits passed." if change["health_passed"] else "A preservation limit was exceeded.") + '</p>'
    return f'''<section class="vs-turn-inspector" aria-label="Selected turn">
      <div class="vs-section-heading"><h3>Turn {turn}</h3><span class="vs-decision-label vs-decision-{decision.lower()}">{decision}</span></div>
      <p class="vs-parent">Parent: {_turn_name(parent)} · {_text(operation)}</p>
      <p class="vs-decision">{_text(reason)}</p>
      <div class="vs-table-scroll"><table><caption>Validation comparison</caption><thead><tr><th>Metric</th><th>Parent</th><th>Candidate</th><th>Change</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
      {health_table}{health_verdict}<p class="vs-operation-settings">{" · ".join(settings)}</p>
      {f'<p class="vs-editor-rationale">{_text(rationale)}</p>' if rationale else ""}{tensors}
    </section>'''


def _layers(view: dict, selection: dict) -> str:
    count = _int(view["structure"].get("num_layers"))
    turn = selection["turn"]
    change = view["changes"].get(turn, {})
    targets = _list(_dict(change.get("proposal")).get("targets"))
    targets += [_dict(item).get("name", "") for item in _list(_dict(change.get("mutation")).get("changed_tensors"))]
    selected = {int(match.group(1)) for target in targets if (match := re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(target)))}
    active_tensor = view["tensor"]
    active = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(active_tensor.get("tensor", "")))
    active_layer = int(active.group(1)) if active and active_tensor.get("status") == "started" and selection["follow"] else None
    heading = f'<div class="vs-section-heading"><h3>Edited layers</h3><span class="vs-kicker">{_turn_name(turn).upper()} · {count} LAYERS</span></div>'
    if not 0 < count <= 4096:
        return '<section class="vs-layers"><div class="vs-section-heading"><h3>Edited layers</h3></div><p>The model structure appears when the checkpoint loads.</p></section>'
    cells = []
    for layer in range(count):
        kind = "active" if layer == active_layer else "selected" if layer in selected else "untouched"
        cells.append(f'<span class="vs-layer vs-layer-{kind}" aria-label="Layer {layer}: {kind}" title="Layer {layer}: {kind}">{layer:02d}</span>')
    detail = f'<p class="vs-tensor-name">Editing {_text(active_tensor.get("tensor"))}</p>' if active_layer is not None else ""
    return f'<section class="vs-layers">{heading}<div class="vs-layer-map">{"".join(cells)}</div><div class="vs-legend"><span><i class="vs-swatch vs-swatch-active"></i>Editing now</span><span><i class="vs-swatch vs-swatch-selected"></i>Selected turn</span><span><i class="vs-swatch"></i>Unchanged in this turn</span></div>{detail}</section>'


def _example_key(answer: dict) -> str:
    example = _dict(answer.get("example"))
    if example.get("id"):
        return str(example["id"])
    identity = [example.get("split"), example.get("prompt"), example.get("target"), example.get("mode"), example.get("check")]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _example_split(answer: dict) -> str:
    example = _dict(answer.get("example"))
    if "health" in str(answer.get("label", "")).lower() or example.get("check"):
        return "health"
    return str(example.get("split") or "edit")


def _example_score(example: dict, metric: str) -> float | None:
    value = example.get(metric)
    if metric == "exact_match" and isinstance(value, bool):
        return float(value)
    if metric == "passed" and isinstance(value, bool):
        return float(value)
    return _number(value)


def comparison_rows(view: dict, selection: dict) -> list[dict]:
    """Pair actual outputs by stable example identity, preserving all examples."""
    grouped = {}
    for answer in view["answers"]:
        key = _example_key(answer)
        entry = grouped.setdefault(key, {"id": key, "baseline": None, "selected": None, "split": _example_split(answer)})
        turn = _int(answer.get("turn"))
        if turn == 0 and "reload" not in str(answer.get("label", "")).lower():
            entry["baseline"] = answer
        if turn == selection["turn"]:
            # Reload rows verify the retained checkpoint and supersede its earlier output.
            entry["selected"] = answer
    rows = []
    for item in grouped.values():
        if item["baseline"] is None and item["selected"] is None:
            continue
        if selection["split"] == "all" and item["split"] == "health":
            continue
        if selection["split"] != "all" and item["split"] != selection["split"]:
            continue
        base = _dict(_dict(item["baseline"]).get("example"))
        chosen = _dict(_dict(item["selected"]).get("example"))
        metric = "passed" if item["split"] == "health" else view["metric"]
        a, b = _example_score(base, metric), _example_score(chosen, metric)
        difference = None if a is None or b is None else (a-b if metric == "loss" else b-a)
        outcome = selection["outcome"]
        if outcome == "improved" and (difference is None or difference <= 0):
            continue
        if outcome == "regressed" and (difference is None or difference >= 0):
            continue
        if outcome == "changed" and (not base or not chosen or base.get("output",base.get("prediction")) == chosen.get("output",chosen.get("prediction"))):
            continue
        rows.append(item)
    split_order = {"validation": 0, "edit": 1, "test": 2, "health": 3}
    return sorted(rows, key=lambda item: split_order.get(item["split"], 4))


def _example_output(answer: dict | None, title: str) -> str:
    if answer is None:
        return f'<section class="vs-answer-output"><h4>{_text(title)}</h4><p class="vs-muted">No output recorded for this checkpoint and example.</p></section>'
    example = _dict(answer.get("example"))
    output = example.get("output", example.get("prediction", ""))
    measures = []
    if isinstance(example.get("passed"), bool):
        measures.append("Passed" if example["passed"] else "Failed")
    if isinstance(example.get("exact_match"), bool):
        measures.append("Exact match" if example["exact_match"] else "Exact miss")
    for key, name in (("token_f1", "F1"), ("semantic", "Goal score"), ("loss", "Reference loss")):
        value = _number(example.get(key))
        if value is not None:
            measures.append(f"{name} {_fmt(value, key)}")
    if example.get("finish_reason"):
        measures.append("Token limit reached" if example["finish_reason"] == "length" else str(example["finish_reason"]))
    count = example.get("generated_tokens")
    if _number(count) is not None:
        measures.append(f"{int(count)} tokens")
    judgement = f'<p class="vs-judgement">{_text(example["judgement"])}</p>' if example.get("judgement") else ""
    source = '<span class="vs-reload-label">Reloaded checkpoint</span>' if "reload" in str(answer.get("label", "")).lower() else ""
    shortened = '<p class="vs-display-note">The run bundle includes the full text of this shortened display.</p>' if answer.get("truncated_fields") or example.get("truncated_fields") else ""
    return f'<section class="vs-answer-output"><h4>{_text(title)}</h4>{source}<p class="vs-generated-text">{_text(output) or "No text returned."}</p><p class="vs-answer-measures">{_text(" · ".join(measures))}</p>{judgement}{shortened}</section>'


def _select_control(action: str, label: str, value: str, choices: tuple) -> str:
    options = "".join(f'<option value="{_text(key)}"{" selected" if key == value else ""}>{_text(name)}</option>' for key, name in choices)
    return f'<label>{_text(label)}<select data-vs-action="{action}" aria-label="{_text(label)}">{options}</select></label>'


def _answers(view: dict, selection: dict) -> str:
    rows = comparison_rows(view, selection)
    pages = max(1, math.ceil(len(rows) / PAGE_SIZE))
    page = min(selection["page"], pages-1)
    selection["page"] = page
    controls = _select_control("split", "Example set", selection["split"], SPLITS) + _select_control("outcome", "Outcome", selection["outcome"], OUTCOMES)
    entries = []
    for item in rows[page*PAGE_SIZE:(page+1)*PAGE_SIZE]:
        example = _dict(_dict(item["selected"] or item["baseline"]).get("example"))
        prompt = str(example.get("prompt") or "Unconditional generation")
        target = example.get("target")
        reference_label = "Style example" if example.get("mode") == "corpus" else "Reference"
        reference = f'<details class="vs-reference" data-vs-details="reference-{_text(item["id"])}"><summary>{reference_label}</summary><p>{_text(target)}</p></details>' if target and example.get("mode") != "prompts" else ""
        identifier = hashlib.sha256(item["id"].encode()).hexdigest()[:16]
        entries.append(f'<article class="vs-answer-comparison" data-vs-sample="{identifier}"><header><span class="vs-kicker">{_text(item["split"])}</span><h4>{_text(prompt)}</h4></header>{reference}<div class="vs-comparison-grid">{_example_output(item["baseline"], "Baseline")}{_example_output(item["selected"], _turn_name(selection["turn"]))}</div></article>')
    if not entries:
        entries.append('<p class="vs-empty-line">No recorded examples match these filters yet.</p>')
    pagination = f'<div class="vs-page-controls"><button type="button" data-vs-action="page" data-vs-value="{max(0,page-1)}"{" disabled" if page == 0 else ""}>Previous examples</button><span>Page {page+1} of {pages} · {len(rows)} examples</span><button type="button" data-vs-action="page" data-vs-value="{page+1}"{" disabled" if page+1 == pages else ""}>Next examples</button></div>'
    return f'<section class="vs-observations" aria-label="Answer comparison"><div class="vs-section-heading"><h3>Compare generated answers</h3><span class="vs-kicker">BASELINE → {_turn_name(selection["turn"]).upper()}</span></div><div class="vs-answer-toolbar">{controls}</div><div class="vs-answer-list" data-vs-scroll="answers">{"".join(entries)}</div>{pagination}</section>'


def _editor(view: dict, selection: dict) -> str:
    calls = [call for call in view["editor_calls"].values() if call.get("turn") in (selection["turn"], None)]
    parts = []
    for index, call in enumerate(calls[-3:]):
        updates = _list(call.get("updates"))
        body = "".join(f'<p>{_text(update.get("text"))}</p>' for update in updates)
        if not body:
            body = '<p>The editor is preparing its response.</p>'
        parts.append(f'<details data-vs-details="editor-{selection["turn"]}-{index}"><summary>{_text(str(call.get("purpose", "Editor activity")).replace("_", " "))} · {_text(call.get("status", ""))}</summary><p class="vs-editor-model">{_text(call.get("model", ""))}</p>{body}</details>')
    if view["assessment"] or view["plan"]:
        parts.append(f'<details class="vs-assessment" data-vs-details="assessment"><summary>Dataset assessment and scoring plan</summary><p>{_text(view["assessment"])}</p><p>{_text(view["plan"])}</p></details>')
    return f'<section class="vs-editor"><div class="vs-section-heading"><h3>Editor activity</h3></div>{"".join(parts) or "<p>Editor summaries will appear when the request begins.</p>"}</section>'


def _genealogy(view: dict, selection: dict) -> str:
    changes = sorted(view["changes"].values(), key=lambda row: _int(row.get("turn")))
    if not changes:
        return ""
    width = max(520, 70 + len(changes)*90)
    positions = {0: (32,40)}
    paths, nodes, rows = [], [], []
    for index, change in enumerate(changes, 1):
        positions[_int(change.get("turn"))] = (32+index*90,40 if change.get("accepted") is True else 94)
    all_changes = [{"turn": 0, "accepted": True}, *changes]
    for change in all_changes:
        turn, parent = _int(change.get("turn")), _int(change.get("parent_turn"))
        status = "baseline" if turn == 0 else "retained" if change.get("accepted") is True else "reverted" if change.get("accepted") is False else "proposed"
        x,y = positions[turn]
        if turn and parent in positions:
            px,py = positions[parent]
            paths.append(f'<path d="M {px} {py} C {px+38} {py}, {x-30} {y}, {x} {y}" class="vs-ancestry-edge vs-ancestry-{status}"/>')
        nodes.append(f'<g role="button" tabindex="0" data-vs-turn="{turn}" class="vs-lineage-point{" is-selected" if turn == selection["turn"] else ""}" aria-label="{_turn_name(turn)}, {status}" aria-pressed="{str(turn == selection["turn"]).lower()}"><circle cx="{x}" cy="{y}" r="20" fill="transparent"/><circle cx="{x}" cy="{y}" r="{7 if turn == selection["turn"] else 5}" class="vs-genealogy-{status}"/><text x="{x}" y="{y-17}" text-anchor="middle">{"BASE" if turn == 0 else f"{turn:02d}"}</text></g>')
        if turn:
            proposal = _dict(change.get("proposal"))
            reason = change.get("decision_reason") or proposal.get("rationale") or "Evaluation in progress"
            score = _score(change.get("metrics"), view["metric"])
            rows.append(f'<tr><td>{_turn_button(turn,selection["turn"])}</td><td>{_turn_name(parent)}</td><td>{_text(proposal.get("operation", "—"))}</td><td>{_fmt(score,view["metric"])}</td><td>{_text(status)}</td><td>{_text(reason)}</td></tr>')
    return f'<section class="vs-genealogy"><div class="vs-section-heading"><h3>Edit genealogy</h3><span class="vs-kicker">SELECT A CANDIDATE</span></div><div class="vs-genealogy-scroll" data-vs-scroll="genealogy"><svg viewBox="0 0 {width} 124" style="min-width:{width}px" role="group" aria-label="Candidate ancestry">{"".join(paths)}{"".join(nodes)}</svg></div><details class="vs-genealogy-record" data-vs-details="genealogy"><summary>Turn details</summary><div class="vs-table-scroll"><table><thead><tr><th>Turn</th><th>Parent</th><th>Operation</th><th>Score</th><th>Decision</th><th>Reason</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></details></section>'


def _health_count(check: dict) -> str:
    count = _number(check.get("count"))
    score = _health_score(check)
    if count is None:
        return ""
    correct = score*count if score is not None else None
    if correct is not None and math.isclose(correct,round(correct),abs_tol=.00001):
        return f"{round(correct)}/{int(count)} correct"
    return f"{int(count)} examples"


def _health(view: dict, selection: dict) -> str:
    drop = _number(view["health_max_drop"])
    policy = "Tracking scores" if drop is None else f"Rejecting drops above {drop*100:g} percentage points"
    charts = []
    for key, raw in view["health_baseline"].items():
        check = _dict(raw)
        a = _health_score(check)
        if a is None:
            continue
        chosen = _dict(_health_for(view,selection["turn"]).get(key))
        best = _dict(view["health_best"].get(key))
        label = check.get("label") or key
        threshold = max(0.0,a-drop) if drop is not None else None
        curve = _curve(a,view["turns"],lambda t,k=key:_health_score(_dict(t.get("health")).get(k)),label=f"{label} by edit turn",compact=True,selected=selection["turn"],threshold=threshold)
        count_note = _health_count(chosen) if chosen else f'{_int(check.get("count"))} examples'
        charts.append(f'<section class="vs-health-chart"><div><h4>{_text(label)}</h4><span>{_fmt(_health_score(chosen))}</span></div><p>{_text(count_note)} · {_turn_name(selection["turn"])}</p>{curve}<p>Baseline {_fmt(a)} · retained {_fmt(_health_score(best))}</p>{f"<p>Minimum score {_fmt(threshold)}</p>" if threshold is not None else ""}</section>')
    if not charts:
        return ""
    return f'<section class="vs-health"><div class="vs-section-heading"><h3>Preservation checks</h3><span class="vs-health-policy">{policy}</span></div><div class="vs-health-grid">{"".join(charts)}</div></section>'


def _completion(view: dict) -> str:
    report = view["report"]
    if not report or view["status"] not in {"complete", "stopped", "error"}:
        return ""
    metric = str(report.get("metric") or view["metric"])
    final = _dict(report.get("final_metrics"))
    baseline = _dict(report.get("baseline_metrics"))
    rows = []
    for split in ("validation","test","edit"):
        current = _dict(final.get(split))
        if not current:
            continue
        label = {"validation":"Validation", "test":"Held-out test", "edit":"Edit examples"}[split]
        previous = _dict(baseline.get(split))
        a,b = _number(previous.get(metric)),_number(current.get(metric))
        count = _number(current.get("count"))
        rows.append(f'<tr><th>{label}</th><td>{int(count) if count is not None else "—"}</td><td>{_fmt(a,metric)}</td><td>{_fmt(b,metric)}</td><td>{_difference(a,b,metric)}</td><td>{_fmt(_number(current.get("loss")),"loss")}</td></tr>')
    reasons = {"turn_limit":"Maximum edit turns reached", "budget":"Editor budget reached", "budget_exhausted":"Editor budget reached", "cancelled":"Stopped by you", "no_improvement":"No further improvement", "editor_stop":"The editor completed its plan"}
    raw_reason = str(report.get("stop_reason") or "Experiment complete")
    reason = reasons.get(raw_reason,raw_reason.replace("_"," ").capitalize())
    reload = report.get("reload_verified")
    reload_text = "Checkpoint reload verified" if reload is True else "Checkpoint reload did not pass verification" if reload is False else "Checkpoint reload verification was not recorded"
    best_turn = _int(report.get("best_turn"),view["retained_turn"])
    budget = _dict(report.get("budget"))
    spend = _money(budget.get("spent_usd",view["spent_usd"]))
    health = []
    for key,value in _dict(report.get("final_health")).items():
        check = _dict(value)
        health.append(f'<li>{_text(check.get("label") or key)}: {_fmt(_health_score(check))} · {_text(_health_count(check))}</li>')
    scoring_error = f'<p class="vs-result-error">{_text(report["final_scoring_error"])}</p>' if report.get("final_scoring_error") else ""
    test_note = '<p class="vs-display-note">Held-out test results were not recorded for this run.</p>' if not _dict(final.get("test")) else ""
    return f'<section class="vs-result-summary" aria-label="Completed experiment results"><div class="vs-section-heading"><h2>Experiment results</h2><span class="vs-kicker">RETAINED {_turn_name(best_turn).upper()}</span></div><p>{_text(reason)}. {reload_text}.</p><div class="vs-table-scroll"><table><caption>{_text(_metric_name(metric))}</caption><thead><tr><th>Dataset</th><th>Examples</th><th>Baseline</th><th>Retained</th><th>Change</th><th>Reference loss</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>{test_note}{scoring_error}<div class="vs-result-grid"><div><h3>Preservation</h3><ul>{"".join(health)}</ul></div><div><h3>Saved checkpoint</h3><p>{_turn_name(best_turn)} · {spend} frontier spend</p><p>{_text(str(report.get("checkpoint") or "checkpoint").replace(chr(92),"/").split("/")[-1])}</p><a href="#vs-results">Download or save this experiment →</a></div></div></section>'


def render_dashboard(state: dict | None = None, selection: dict | None = None) -> str:
    """Render measured events and a selected checkpoint without exposing secrets."""
    state = _dict(state)
    view = _replay(state)
    selection = normalise_selection(view,selection)
    metric = view["metric"]
    a,b = _score(view["baseline"],metric),_score(view["best"],metric)
    status = view["status"]
    status_name = {"idle":"Not started", "running":"In progress", "queued":"Waiting for GPU", "complete":"Complete", "stopped":"Stopped", "error":"Interrupted", "stopping":"Stopping"}.get(status,"In progress")
    elapsed = _number(state.get("elapsed",state.get("elapsed_seconds")))
    elapsed_text = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}" if elapsed is not None else "—"
    progress = ""
    if view["total"]:
        proportion = min(100,max(0,view["current"]/view["total"]*100))
        progress = f'<div class="vs-progress"><span>{view["current"]}/{view["total"]}</span><div role="progressbar" aria-label="Current operation" aria-valuemin="0" aria-valuemax="{view["total"]}" aria-valuenow="{min(view["current"],view["total"])}"><i style="width:{proportion:.2f}%"></i></div></div>'
    phase = "Waiting for a GPU allocation" if status == "queued" else str(view["phase"]).replace("_"," ")
    chart = _curve(a,view["turns"],lambda t:_score(t.get("metrics"),metric),label=f"Validation {_metric_name(metric)} by edit turn",metric=metric,selected=selection["turn"])
    turns = [0,*sorted(view["changes"])]
    buttons = "".join(_turn_button(turn,selection["turn"]) for turn in turns)
    follow = f'<button type="button" class="vs-follow-button" data-vs-action="follow" data-vs-value="{str(not selection["follow"]).lower()}" aria-pressed="{str(selection["follow"]).lower()}">{"Following latest" if selection["follow"] else "Follow latest"}</button>'
    answers = _answers(view,selection)
    measured = ""
    if a is not None:
        rows = f'<tr><th>{_turn_button(0,selection["turn"])}</th><td>{_fmt(a,metric)}</td><td>Reference</td></tr>'
        for turn in view["turns"]:
            rows += f'<tr><th>{_turn_button(_int(turn.get("turn")),selection["turn"])}</th><td>{_fmt(_score(turn.get("metrics"),metric),metric)}</td><td>{_text(turn.get("decision_reason") or ("Retained" if turn.get("accepted") else "Reverted"))}</td></tr>'
        measured = f'<details class="vs-measurements" data-vs-details="measurements"><summary>Measured values and decisions</summary><div class="vs-table-scroll"><table><thead><tr><th>Turn</th><th>{_text(_metric_name(metric))}</th><th>Decision</th></tr></thead><tbody>{rows}</tbody></table></div></details>'
    accepted = sum(change.get("accepted") is True for change in view["changes"].values())
    model = str(view["model"]).split("/")[-1] if view["model"] else "No model loaded"
    return f'''<main class="vs-dashboard" aria-label="Experiment dashboard" data-vs-selection="{_text(json.dumps(selection,separators=(',',':')))}" data-vs-selected-turn="{selection['turn']}" data-vs-follow="{str(selection['follow']).lower()}">
      {_completion(view)}
      <div class="vs-run-heading"><div><span class="vs-kicker">EXPERIMENT MONITOR</span><h2>Hill climb</h2></div><div class="vs-run-state vs-state-{_text(status)}"><span><i aria-hidden="true"></i>{status_name}</span><small>{_text(model)}</small></div></div>
      {f'<p class="vs-current-goal"><span>OBJECTIVE</span>{_text(view["goal"])}</p>' if view['goal'] else ''}
      <div class="vs-phase-line"><span>{_text(phase)}</span>{progress}</div><p class="vs-run-message">{_text(view['message'])}</p>
      <div class="vs-telemetry"><div><span class="vs-kicker">EDIT TURNS</span><strong>{len(view['turns'])}<span>/{view['max_turns'] or '—'}</span></strong></div><div><span class="vs-kicker">RETAINED EDITS</span><strong>{accepted}</strong></div><div><span class="vs-kicker">ELAPSED</span><strong>{elapsed_text}</strong></div><div><span class="vs-kicker">FRONTIER SPEND</span><strong>{_money(view['spent_usd'])}<span>/{_money(view['budget_usd'])}</span></strong></div></div>
      <section class="vs-objective"><div class="vs-chart-heading"><div><span class="vs-kicker">VALIDATION OBJECTIVE</span><h3>{_text(_metric_name(metric))} {'↓' if metric == 'loss' else '↑'}</h3></div><div class="vs-score-comparison"><div><span>BASELINE</span><strong>{_fmt(a,metric)}</strong></div><b aria-hidden="true">→</b><div><span>RETAINED</span><strong>{_fmt(b,metric)}</strong></div><span class="vs-delta">{_difference(a,b,metric)}</span></div></div>{chart}<div class="vs-chart-footer"><div class="vs-legend"><span><i class="vs-line-swatch"></i>Retained path</span><span><i class="vs-dot-swatch"></i>Reverted trial</span><span><i class="vs-line-swatch vs-baseline-swatch"></i>Baseline</span></div><span class="vs-kicker">EDIT TURN</span></div>{measured}</section>
      <div class="vs-turn-toolbar" aria-label="Inspect a turn"><div>{buttons}</div>{follow}</div>
      <div class="vs-inspector-grid">{_turn_inspector(view,selection)}{_layers(view,selection)}</div>
      {_health(view,selection)}{_genealogy(view,selection)}{answers}{_editor(view,selection)}
    </main>'''
