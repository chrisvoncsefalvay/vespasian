"""Vespasian's guided workbench on Hugging Face ZeroGPU."""

import spaces
spaces.disable_gradio_auto_wrap()

import html
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

import hub
import runtime
import setup_helpers as setup
from gpu import dispatch, model_availability, model_options
from ui import ASSET_DIR, CSS, JS, THEME, build_ui, render_dashboard

ROOT = Path(__file__).resolve().parent
FIELDS = (
    "target_type", "hf_task", "save_format", "prompt_format", "patch_backup", "patch_repo", "patch_create", "model", "dataset_location", "dataset_source", "dataset_file",
    "dataset_config", "dataset_split", "dataset_mode", "prompt_field", "target_field", "text_field",
    "example_limit", "test_dataset_source", "test_dataset_config", "test_dataset_split", "test_example_limit",
    "goal", "controls", "metric", "editor", "editor_model", "api_key", "hf_token", "budget", "max_turns",
    "max_new_tokens", "health_checks", "health_count", "health_policy", "health_limit_pp", "save_trajectory",
    "input_price", "output_price", "reasoning_effort",
)
PREVIEW_FIELDS = (
    "hf_task", "dataset_location", "dataset_source", "dataset_file", "dataset_config", "dataset_split", "dataset_mode",
    "prompt_field", "target_field", "text_field", "example_limit", "hf_token",
    "test_dataset_source", "test_dataset_config", "test_dataset_split", "test_example_limit",
)
STEP_FIELDS = {
    "patch_repo": "vs-patch-backup", "hf_task": "vs-hf-task", "save_format": "vs-step-editor", "prompt_format": "vs-step-editor", "goal": "vs-goal", "target_type": "vs-target-type", "metric": "vs-metric", "dataset_mode": "vs-dataset-mode",
    "model": "vs-model", "dataset_source": "vs-dataset-source", "dataset_file": "vs-dataset-file",
    "dataset_config": "vs-dataset-config", "dataset_split": "vs-dataset-split", "example_limit": "vs-example-limit",
    "preview": "vs-preview-button", "preview_state": "vs-preview-button", "test_preview": "vs-test-preview-button",
    "test_dataset_source": "vs-test-source", "prompt_field": "vs-prompt-field", "target_field": "vs-target-field",
    "text_field": "vs-text-field", "editor": "vs-editor", "editor_model": "vs-editor-model", "api_key": "vs-api-key",
    "reasoning_effort": "vs-reasoning-effort", "budget": "vs-budget", "max_turns": "vs-max-turns", "health_checks": "vs-health-checks",
    "health_count": "vs-health-count", "health_limit_pp": "vs-health-limit", "health_policy": "vs-health-policy",
    "input_price": "vs-input-price", "output_price": "vs-output-price",
}
DEFAULT_SELECTION = {"turn": None, "split": "all", "outcome": "all", "page": 0, "follow": True}


def safe_error(error, *secrets):
    text = str(error)
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "[redacted]")
    return text[:1000]


def error_html(errors):
    if not errors:
        return ""
    items = []
    first = ""
    for field, message in errors.items():
        target = "#" + STEP_FIELDS.get(field, "vs-preview-button")
        first = first or target
        items.append(f'<li><button type="button" data-vs-focus="{target}">{html.escape(str(message))}</button></li>')
    return f'<div class="vs-form-errors" role="alert" data-vs-focus="{first}"><p>Complete these fields to continue.</p><ul>{"".join(items)}</ul></div>'


def parse_selection(raw, previous=None):
    value = json.loads(raw or "{}") if isinstance(raw, str) and len(raw) <= 4096 else {}
    if not isinstance(value, dict):
        value = {}
    result = dict(DEFAULT_SELECTION)
    if isinstance(previous, dict):
        result.update({key: previous[key] for key in result if key in previous})
    if value.get("turn") is None or isinstance(value.get("turn"), int) and not isinstance(value.get("turn"), bool) and 0 <= value["turn"] <= 4294967295:
        if "turn" in value:
            result["turn"] = value["turn"]
    for key, choices in (("split", {"all", "edit", "validation", "test", "health"}), ("outcome", {"all", "improved", "regressed", "changed"})):
        if value.get(key) in choices:
            result[key] = value[key]
    if isinstance(value.get("page"), int) and not isinstance(value.get("page"), bool):
        result["page"] = max(0, min(10000, value["page"]))
    if isinstance(value.get("follow"), bool):
        result["follow"] = value["follow"]
    return result


def create_app():
    demo, controls = build_ui()
    available = model_availability()
    choices = [entry["id"] for entry in available if entry["available"]]
    if not choices and os.environ.get("VESPASIAN_SPACE_SKIP_MODEL_LOAD") == "1":
        choices = ["Qwen/Qwen3.5-0.8B", "google/gemma-3-1b-it"]
    controls["model"].choices = [(label, name) for label, name in model_options() if name in choices]
    controls["model"].value = choices[0] if choices else None
    demo.vespasian_controls = controls

    def collect(values):
        return {name: values[control] for name, control in controls.items() if control in values}

    def account(override, oauth_token: gr.OAuthToken | None, profile: gr.OAuthProfile | None):
        token = hub.token_for(oauth_token, override)
        try:
            name, datasets = hub.account_datasets(token)
        except Exception as error:
            return (f'<div class="vs-profile"><p>{html.escape(safe_error(error, token))}</p></div>', [], gr.skip(), gr.skip(), gr.skip())
        if name:
            display = f'<div class="vs-profile"><span class="vs-connected">Connected</span><strong>{html.escape(name)}</strong><p>Your datasets and Hub exports are available.</p></div>'
            return display, datasets, gr.update(choices=datasets), gr.update(placeholder=f"{name}/vespasian-model"), gr.update(placeholder=f"{name}/vespasian-trajectories")
        return ('<div class="vs-profile"><p>Use public datasets now, or connect your account for private data and Hub exports.</p></div>', [], gr.update(choices=["openai/gsm8k"]), gr.skip(), gr.skip())

    def catalog(kind, owned, recent):
        entries = owned if kind == "account" else recent if kind == "recent" else ["openai/gsm8k"]
        message = "Sign in or refresh your account to load your datasets." if kind == "account" and not entries else "No datasets inspected in this session yet." if kind == "recent" and not entries else ""
        return gr.update(choices=entries), f'<p class="vs-inline-note">{message}</p>'

    def model_task_hint(model, override, oauth_token: gr.OAuthToken | None):
        token = hub.token_for(oauth_token, override)
        try:
            metadata = hub.model_task_metadata(model, token=token)
            suggested = metadata.get("model_suggested_tasks") or []
            text = ("Model tags suggest " + ", ".join(suggested) + ". Confirm the task against your dataset; tags do not prove editing support."
                    if suggested else "Model task tags are missing or unsupported. Choose the task explicitly and inspect your dataset.")
            return '<p class="vs-inline-note">' + html.escape(text) + '</p>'
        except Exception as error:
            return '<p class="vs-inline-note">Model tags could not be checked. Choose the task explicitly. ' + html.escape(safe_error(error, token)) + '</p>'

    def search(query, task, include_untagged, override, oauth_token: gr.OAuthToken | None):
        token = hub.token_for(oauth_token, override)
        try:
            results = hub.search_datasets(query, token=token, hf_task=None if include_untagged else task)
            return gr.update(choices=results), f'<p class="vs-inline-note">{len(results)} public datasets found. Choose one to inspect its fields.</p>'
        except Exception as error:
            return gr.skip(), f'<p class="vs-preview-note" role="alert">{html.escape(safe_error(error, token))}</p>'

    def current_preview(config, test=False):
        result = dict(config.get("test_preview_state" if test else "preview_state") or {})
        result["_current"] = bool(result and result.get("_fingerprint") == setup.fingerprint(config, test=test))
        return result

    def split_html(config, preview=None):
        summary = setup.split_summary(config, preview)
        # The helper explicitly marks estimated counts when only sample rows were fetched.
        if isinstance(summary, str):
            return summary
        label = "Estimated split" if not summary.get("exact") else "Example split"
        parts = [f'{html.escape(str(key).replace("_", " ").title())}: {html.escape(str(summary[key]))}' for key in ("edit", "validation", "test", "preserve") if summary.get(key) is not None and (key != "preserve" or summary[key])]
        return f'<p class="vs-inline-note"><strong>{label}</strong> · {" · ".join(parts)}</p>' if parts else '<p class="vs-inline-note">Inspect the dataset to check its examples and split.</p>'

    def inspect_dataset(values, oauth_token: gr.OAuthToken | None):
        config = collect(values)
        token = hub.token_for(oauth_token, config.get("hf_token"))
        try:
            if config.get("dataset_location") == "file":
                if not config.get("dataset_file"):
                    raise ValueError("Choose a dataset file")
                result = hub.preview_upload(config["dataset_file"], config)
            else:
                result = hub.preview(config, token)
            updated = dict(config)
            metadata = result.get("task_metadata") or {}
            if metadata.get("selected_task"):
                updated["hf_task"] = metadata["selected_task"]
            for name, key in (("dataset_config", "config"), ("dataset_split", "split"), ("prompt_field", "prompt_field"), ("target_field", "target_field"), ("text_field", "text_field")):
                if result.get(key):
                    updated[name] = result[key]
            result["_fingerprint"] = setup.fingerprint(updated)
            result["_current"] = True
            paths = hub.field_choices(result)
            recent = list(config.get("recent_datasets") or [])
            source = config.get("dataset_source")
            if config.get("dataset_location") == "hub" and source:
                recent = [source, *[item for item in recent if item != source]][:12]
            response = {
                controls["preview_html"]: hub.preview_html(result), controls["preview_state"]: result,
                controls["hf_task"]: updated.get("hf_task"), controls["task_help"]: setup.task_help(metadata),
                controls["dataset_config"]: gr.update(choices=result.get("configs", []), value=updated.get("dataset_config")),
                controls["dataset_split"]: gr.update(choices=result.get("splits", ["train", "validation", "test"]), value=updated.get("dataset_split")),
                **{controls[name]: gr.update(choices=paths, value=updated.get(name) or None) for name in ("prompt_field", "target_field", "text_field")},
                controls["split_summary"]: split_html(updated, result), controls["recent_datasets"]: recent,
                controls["preview_tabs"]: gr.update(selected="edit"),
            }
            if setup.fingerprint(config, test=True) != setup.fingerprint(updated, test=True):
                response[controls["test_preview_html"]] = hub.stale_preview_html("Held-out test")
                response[controls["test_preview_state"]] = {}
            return response
        except Exception as error:
            return {controls["preview_html"]: f'<p class="vs-preview-note" role="alert">{html.escape(safe_error(error, token))}</p>', controls["preview_state"]: {}, controls["preview_tabs"]: gr.update(selected="edit")}

    def inspect_test(values, oauth_token: gr.OAuthToken | None):
        original = collect(values)
        config = dict(original)
        for name in ("dataset_source", "dataset_config", "dataset_split"):
            config[name] = config.get("test_" + name)
        token = hub.token_for(oauth_token, config.get("hf_token"))
        try:
            result = hub.preview(config, token)
            updated = dict(original)
            updated["test_dataset_config"] = result.get("config") or config.get("dataset_config")
            updated["test_dataset_split"] = result.get("split") or config.get("dataset_split")
            result["_fingerprint"] = setup.fingerprint(updated, test=True)
            result["_current"] = True
            return hub.preview_html(result, role="Held-out test"), gr.update(choices=result.get("configs", []), value=updated["test_dataset_config"]), gr.update(choices=result.get("splits", []), value=updated["test_dataset_split"]), result, gr.update(selected="test")
        except Exception as error:
            return f'<p class="vs-preview-note" role="alert">{html.escape(safe_error(error, token))}</p>', gr.skip(), gr.skip(), {}, gr.update(selected="test")

    def invalidate(test=False, both=False):
        def callback():
            result = {}
            for prefix, role in (("", "Edit/validation"), ("test_", "Held-out test")):
                if both or (prefix == "test_") == test:
                    result[controls[prefix + "preview_html"]] = hub.stale_preview_html(role)
                    result[controls[prefix + "preview_state"]] = {}
            return result
        return callback

    def navigation(values, step, destination):
        config = collect(values)
        preview, test = current_preview(config), current_preview(config, True)
        errors = setup.validate_step(config, step, preview, test)
        result = {controls[f"error_{step}"]: error_html(errors)}
        if not errors:
            result[controls["setup_tabs"]] = gr.update(selected=destination)
            result[controls["review_html"]] = setup.review_html(config, preview, test)
        return result

    def review(values):
        config = collect(values)
        return setup.review_html(config, current_preview(config), current_preview(config, True))

    def display_run(run, state, selection=None):
        selection = getattr(run, "view_selection", None) if isinstance(getattr(run, "view_selection", None), dict) else selection
        finished = state["finished"]
        has_checkpoint = finished and (run.output / "checkpoint").is_dir() and (run.output / "report.json").is_file() and not state["error"]
        patch_archive = getattr(run, "patch_archive", None)
        patch_archive = patch_archive if isinstance(patch_archive, Path) and patch_archive.is_file() else None
        has_patches = finished and (run.output / "patches").is_dir()
        has_trajectory = finished and (run.output / "trajectory.jsonl").is_file()
        return {
            controls["state"]: state, controls["dashboard_html"]: render_dashboard(state, selection),
            controls["status"]: state["error"] or state["status"],
            controls["patch_download"]: gr.update(value=str(patch_archive) if patch_archive else None, visible=bool(patch_archive)),
            controls["publish_patches_button"]: gr.update(interactive=has_patches),
            controls["download"]: gr.update(value=str(run.archive) if run.archive else None, visible=bool(run.archive)),
            controls["run_button"]: gr.update(interactive=finished), controls["stop_button"]: gr.update(interactive=not finished),
            controls["publish_model_button"]: gr.update(interactive=has_checkpoint), controls["publish_trajectory_button"]: gr.update(interactive=has_trajectory),
            controls["setup_panel"]: gr.update(visible=False), controls["monitor_panel"]: gr.update(visible=True),
            controls["results_panel"]: gr.update(visible=finished), controls["new_experiment"]: gr.update(interactive=finished),
            controls["return_results"]: gr.update(visible=finished), controls["run_timer"]: gr.update(active=not finished),
        }

    def start(values, request: gr.Request, oauth_token: gr.OAuthToken | None):
        config = collect(values)
        preview, test = current_preview(config), current_preview(config, True)
        for step in ("target", "behaviour", "data", "editor"):
            errors = setup.validate_step(config, step, preview, test)
            if errors:
                return {controls[f"error_{step}"]: error_html(errors), controls["setup_tabs"]: gr.update(selected=step), controls["run_button"]: gr.update(interactive=True)}
        token = hub.token_for(oauth_token, config.pop("hf_token", ""))
        try:
            if config.get("patch_backup") and not token:
                raise ValueError("Sign in or provide a session Hugging Face token to authorize private patch backup")
            config = setup.normalize_config(config)
            config = {key: value for key, value in config.items() if key in set(FIELDS) | {"health_max_drop"}}
            run = runtime.begin(request.session_hash, config, token)
            run.view_selection = dict(DEFAULT_SELECTION)
            run.start(config, token, available)
        except Exception as error:
            return {controls["error_review"]: error_html({"goal": safe_error(error, token, config.get("api_key"))}), controls["setup_tabs"]: gr.update(selected="review"), controls["run_button"]: gr.update(interactive=True)}
        result = display_run(run, run.state(), DEFAULT_SELECTION)
        result[controls["selection"]] = dict(DEFAULT_SELECTION)
        for step in ("target", "behaviour", "data", "editor", "review"):
            result[controls[f"error_{step}"]] = ""
        return result

    def advance(state, selection, request: gr.Request):
        try:
            run = runtime.get_run(request.session_hash, state)
        except ValueError as error:
            raise gr.Error(str(error)) from None
        # Every tick is a fresh queued browser request. Selection never schedules GPU work.
        for current in run.advance(dispatch):
            yield display_run(run, current, selection)

    def select_turn(state, raw, previous, request: gr.Request):
        try:
            selection = parse_selection(raw, previous)
        except (ValueError, TypeError):
            selection = dict(previous or DEFAULT_SELECTION)
        if state.get("run_id"):
            run = runtime.get_run(request.session_hash, state)
            run.view_selection = selection
            state = run.state()
        return selection, render_dashboard(state, selection)

    def stop(state, request: gr.Request):
        run = runtime.get_run(request.session_hash, state)
        run.stop()
        return "Stopping after the current operation"

    def new_experiment(state, request: gr.Request):
        run = runtime.get_run(request.session_hash, state)
        if not run.finished:
            return gr.skip(), gr.skip(), gr.skip()
        return gr.update(visible=True), gr.update(visible=False), gr.update(selected="target")

    def publish_patches(state, repo, consent, create, override, request: gr.Request, oauth_token: gr.OAuthToken | None):
        token = hub.token_for(oauth_token, override)
        try:
            run = runtime.get_run(request.session_hash, state)
            if not run.finished:
                raise ValueError("Wait for the run to finish before manually saving patches")
            if consent is not True:
                raise ValueError("Explicitly authorize sharing patches with the named private repository")
            if not token:
                raise ValueError("Sign in or provide a session token to save patches privately")
            repo = (repo or "").strip()
            hub.core_module("patches").publish_private(run.output / "patches", repo, token, consent=consent, create=create)
            return f"Saved private patches to [huggingface.co/{repo}](https://huggingface.co/{repo})"
        except Exception as error:
            return "Patch save stopped: " + safe_error(error, token)

    def publish_model(state, repo, visibility, create, override, request: gr.Request, oauth_token: gr.OAuthToken | None):
        return publish(state, repo, visibility, create, override, request, oauth_token, "model")

    def publish_trajectory(state, repo, visibility, create, override, request: gr.Request, oauth_token: gr.OAuthToken | None):
        return publish(state, repo, visibility, create, override, request, oauth_token, "trajectory")

    def publish(state, repo, visibility, create, override, request, oauth_token, kind):
        token = hub.token_for(oauth_token, override)
        try:
            run = runtime.get_run(request.session_hash, state)
            if not run.finished:
                raise ValueError("Wait for the run to finish before publishing")
            hub.publish_run(run.output, kind=kind, repo=(repo or "").strip(), visibility=visibility, create=create, token=token)
            prefix = "datasets/" if kind == "trajectory" else ""
            return f"Saved to [huggingface.co/{prefix}{repo}](https://huggingface.co/{prefix}{repo})"
        except Exception as error:
            return safe_error(error, token)

    def mode_changed(mode, metric, target):
        description = {"paired": "Each example contains an input and the reference answer you want the model to produce.", "corpus": "Generation starts without a prompt. The text examples guide the writing style and provide references for the goal score.", "prompts": "Each example supplies a prompt. Your objective defines the rubric used to score the generated answer."}[mode]
        return (gr.update(choices=setup.metric_choices(mode), value=setup.metric_for_mode(mode, metric)),
                gr.update(visible=mode != "corpus"), gr.update(visible=mode == "paired"), gr.update(visible=mode == "corpus"),
                f'<p class="vs-inline-note">{description}</p>', gr.update(value="style" if mode == "corpus" and target == "factual" else target))

    def suggestion(target, mode, reference, text):
        return f'<p class="vs-suggestion">{html.escape(setup.objective_suggestion(target, mode, reference, text))}</p>'

    def metric_help(metric):
        text = {"auto": "The editor assesses your examples and chooses a compatible metric before proposing edits.", "exact_match": "Higher is better. The generated answer is compared with the reference using the experiment’s answer normalisation.", "token_f1": "Higher is better. Token F1 measures overlap between the generated answer and the reference.", "semantic": "Higher is better. The frontier editor scores each answer against your objective and preservation requirements."}.get(metric, "")
        return f'<p class="vs-inline-note">{text}</p>'

    def editor_changed(provider, model, memory):
        memory = dict(memory or {})
        previous = memory.get("_provider")
        if previous and model:
            memory[previous] = model
        choices = setup.editor_choices(provider)
        selected = memory.get(provider) or choices[0][1]
        memory["_provider"] = provider
        return gr.update(choices=choices, value=selected), memory, "", "", ""

    def pricing(provider, model, input_price, output_price):
        rates = setup.editor_pricing(provider, model, input_price, output_price)
        if rates["input_per_million"] is None or rates["output_per_million"] is None:
            return '<p class="vs-inline-note">Add input and output token prices for this custom model before continuing.</p>'
        inputs, outputs = rates["input_per_million"], rates["output_per_million"]
        return f'<p class="vs-inline-note">Budget rates per million tokens: ${inputs} input/${outputs} output. Custom prices override these rates.</p>'

    def preset_changed(name):
        config = setup.preset_config(name) if name else {}
        updates = {controls[key]: value for key, value in config.items() if key in controls and key != "model"}
        updates.update({controls["preview_state"]: {}, controls["test_preview_state"]: {}, controls["preview_html"]: hub.stale_preview_html("Edit/validation"), controls["test_preview_html"]: hub.stale_preview_html("Held-out test")})
        return updates

    def load_quick_starter(name):
        allowed = {item["id"] for item in setup.quick_starter_cards()}
        if name not in allowed:
            return {controls["starter_status"]: '<p class="vs-inline-note" role="alert">Choose one of the available quick starters. No settings were changed.</p>'}
        updates = preset_changed(name)
        # A previous preview labels choices with samples from its own dataset.
        # Clear those labels while retaining the new canonical field mappings.
        config = setup.preset_config(name)
        updates.update({controls[key]: gr.update(choices=[], value=config[key] or None)
                        for key in ("prompt_field", "target_field", "text_field")})
        title = setup.PRESETS[name]["label"]
        updates.update({controls["preset"]: name,
                        controls["setup_tabs"]: gr.update(selected="data"),
                        controls["starter_status"]: '<p class="vs-inline-note" role="status">Loaded ' + html.escape(title) + '. Inspect the examples, then choose your editor and review the experiment. Nothing has started.</p>',
                        **{controls[key]: "" for key in ("error_target", "error_behaviour", "error_data", "error_editor", "error_review")}})
        return updates

    with demo:
        all_inputs = {controls[name] for name in (*FIELDS, "preview_state", "test_preview_state")}
        navigation_outputs = [controls[name] for name in ("setup_tabs", "review_html", "error_target", "error_behaviour", "error_data", "error_editor", "error_review")]
        for button, step, destination in (("next_behaviour", "target", "behaviour"), ("next_data", "behaviour", "data"), ("next_editor", "data", "editor"), ("next_review", "editor", "review")):
            def proceed(values, step=step, destination=destination):
                return navigation(values, step, destination)
            controls[button].click(proceed, inputs=all_inputs, outputs=navigation_outputs, queue=False, api_name=False)
        for button, destination in (("back_target", "target"), ("back_behaviour", "behaviour"), ("back_data", "data"), ("back_editor", "editor")):
            controls[button].click(lambda destination=destination: gr.update(selected=destination), outputs=controls["setup_tabs"], queue=False, api_name=False)
        controls["setup_tabs"].select(review, inputs=all_inputs, outputs=controls["review_html"], queue=False, api_name=False)
        controls["model"].input(model_task_hint, inputs=[controls["model"], controls["hf_token"]], outputs=controls["task_help"], api_name=False, concurrency_limit=4)
        controls["preset"].input(preset_changed, inputs=controls["preset"], outputs=[controls[name] for name in (*FIELDS, "preview_state", "test_preview_state", "preview_html", "test_preview_html")], queue=False, api_name=False)
        controls["starter_submit"].click(load_quick_starter, inputs=controls["starter_input"], outputs=[controls[name] for name in (*FIELDS, "preview_state", "test_preview_state", "preview_html", "test_preview_html", "preset", "setup_tabs", "starter_status", "error_target", "error_behaviour", "error_data", "error_editor", "error_review")], queue=False, api_name=False, show_progress="hidden")
        controls["dataset_mode"].change(mode_changed, inputs=[controls[name] for name in ("dataset_mode", "metric", "target_type")], outputs=[controls[name] for name in ("metric", "prompt_field", "target_field", "text_field", "mode_help", "target_type")], queue=False, api_name=False)
        controls["metric"].change(metric_help, inputs=controls["metric"], outputs=controls["metric_help"], queue=False, api_name=False)
        for name in ("target_type", "dataset_mode", "target_field", "text_field"):
            controls[name].change(suggestion, inputs=[controls[field] for field in ("target_type", "dataset_mode", "target_field", "text_field")], outputs=controls["objective_suggestion"], queue=False, api_name=False)
        controls["use_suggestion"].click(setup.objective_suggestion, inputs=[controls[name] for name in ("target_type", "dataset_mode", "target_field", "text_field")], outputs=controls["goal"], queue=False, api_name=False)
        controls["health_policy"].change(lambda policy: (gr.update(visible=policy == "enforce"), '<p class="vs-inline-note">' + ("An edit is rejected when a check falls more than the allowed number of percentage points below its baseline." if policy == "enforce" else "Scores are tracked alongside the objective without rejecting edits for a decline.") + '</p>'), inputs=controls["health_policy"], outputs=[controls["health_limit_pp"], controls["health_help"]], queue=False, api_name=False)
        controls["dataset_location"].change(lambda location: (gr.update(visible=location == "hub"), gr.update(visible=location == "file")), inputs=controls["dataset_location"], outputs=[controls["hub_fields"], controls["dataset_file"]], queue=False, api_name=False)
        controls["editor"].input(editor_changed, inputs=[controls[name] for name in ("editor", "editor_model", "provider_memory")], outputs=[controls[name] for name in ("editor_model", "provider_memory", "api_key", "input_price", "output_price")], queue=False, api_name=False)
        controls["editor_model"].input(lambda: ("", ""), outputs=[controls["input_price"], controls["output_price"]], queue=False, api_name=False)
        price_inputs = [controls[name] for name in ("editor", "editor_model", "input_price", "output_price")]
        for name in ("editor_model", "input_price", "output_price"):
            controls[name].change(pricing, inputs=price_inputs, outputs=controls["pricing_html"], queue=False, api_name=False)
        demo.load(pricing, inputs=price_inputs, outputs=controls["pricing_html"], queue=False, api_name=False)
        account_outputs = [controls[name] for name in ("profile_html", "account_choices", "test_dataset_source", "model_repo", "trajectory_repo")]
        demo.load(account, inputs=[controls["hf_token"]], outputs=account_outputs, api_name=False)
        controls["hf_token"].submit(account, inputs=[controls["hf_token"]], outputs=account_outputs, api_name=False)
        controls["hf_token"].input(lambda: ('<div class="vs-profile"><p>Press Enter to refresh your account with this access token.</p></div>', []), outputs=[controls["profile_html"], controls["account_choices"]], queue=False, api_name=False)
        controls["refresh_datasets"].click(account, inputs=[controls["hf_token"]], outputs=account_outputs, api_name=False).then(lambda owned: gr.update(choices=owned), inputs=controls["account_choices"], outputs=controls["dataset_source"], queue=False, api_name=False)
        controls["dataset_catalog"].input(catalog, inputs=[controls[name] for name in ("dataset_catalog", "account_choices", "recent_datasets")], outputs=[controls["dataset_source"], controls["catalog_status"]], queue=False, api_name=False)
        for event in (controls["search_datasets"].click, controls["dataset_query"].submit):
            event(search, inputs=[controls[name] for name in ("dataset_query", "hf_task", "include_untagged", "hf_token")], outputs=[controls["dataset_source"], controls["catalog_status"]], api_name=False, concurrency_limit=4)
        preview_inputs = {controls[name] for name in (*PREVIEW_FIELDS, "recent_datasets")}
        controls["preview_button"].click(inspect_dataset, inputs=preview_inputs, outputs=[controls[name] for name in ("preview_html", "hf_task", "task_help", "dataset_config", "dataset_split", "prompt_field", "target_field", "text_field", "preview_state", "split_summary", "recent_datasets", "preview_tabs", "test_preview_html", "test_preview_state")], api_name="preview_dataset", concurrency_limit=4)
        controls["test_preview_button"].click(inspect_test, inputs=preview_inputs, outputs=[controls[name] for name in ("test_preview_html", "test_dataset_config", "test_dataset_split", "test_preview_state", "preview_tabs")], api_name="preview_test_dataset", concurrency_limit=4)
        for name in ("dataset_location", "dataset_source", "dataset_config", "dataset_split", "dataset_file", "example_limit"):
            event = controls[name].change if name == "dataset_file" else controls[name].input
            event(invalidate(), outputs=[controls["preview_html"], controls["preview_state"]], queue=False, api_name=False)
        for name in ("test_dataset_source", "test_dataset_config", "test_dataset_split", "test_example_limit"):
            controls[name].input(invalidate(test=True), outputs=[controls["test_preview_html"], controls["test_preview_state"]], queue=False, api_name=False)
        for name in ("hf_task", "dataset_mode", "prompt_field", "target_field", "text_field", "hf_token"):
            controls[name].input(invalidate(both=True), outputs=[controls[field] for field in ("preview_html", "preview_state", "test_preview_html", "test_preview_state")], queue=False, api_name=False)
        controls["example_limit"].input(lambda values: split_html(collect(values)), inputs=all_inputs, outputs=controls["split_summary"], queue=False, api_name=False)
        run_outputs = [controls[name] for name in ("state", "dashboard_html", "status", "download", "run_button", "stop_button", "publish_model_button", "publish_trajectory_button", "setup_panel", "monitor_panel", "results_panel", "new_experiment", "return_results", "setup_tabs", "selection", "error_target", "error_behaviour", "error_data", "error_editor", "error_review", "patch_download", "publish_patches_button", "run_timer")]
        controls["run_button"].click(start, inputs=all_inputs, outputs=run_outputs, api_name="run", concurrency_limit=runtime.MAX_ACTIVE)
        controls["run_timer"].tick(advance, inputs=[controls["state"], controls["selection"]], outputs=run_outputs, api_name="advance", api_visibility="undocumented", queue=True, trigger_mode="once", show_progress="hidden", concurrency_limit=runtime.MAX_ACTIVE, concurrency_id="gpu-allocations")
        controls["selection_submit"].click(select_turn, inputs=[controls[name] for name in ("state", "selection_input", "selection")], outputs=[controls["selection"], controls["dashboard_html"]], queue=False, api_name=False, show_progress="hidden")
        controls["stop_button"].click(stop, inputs=controls["state"], outputs=controls["status"], queue=False, api_name=False)
        controls["new_experiment"].click(new_experiment, inputs=controls["state"], outputs=[controls["setup_panel"], controls["monitor_panel"], controls["setup_tabs"]], queue=False, api_name=False)
        controls["return_results"].click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[controls["setup_panel"], controls["monitor_panel"]], queue=False, api_name=False)
        controls["publish_patches_button"].click(publish_patches, inputs=[controls[name] for name in ("state", "patch_save_repo", "patch_save_consent", "patch_save_create", "hf_token")], outputs=controls["publish_status"], api_name=False, concurrency_limit=2)
        for kind, callback in (("model", publish_model), ("trajectory", publish_trajectory)):
            controls[f"publish_{kind}_button"].click(callback, inputs=[controls[name] for name in ("state", f"{kind}_repo", f"{kind}_visibility", f"{kind}_create", "hf_token")], outputs=controls["publish_status"], api_name=False, concurrency_limit=2)
    return demo


demo = create_app()

if __name__ == "__main__":
    runtime.start_cleanup()
    gr.set_static_paths(paths=[ASSET_DIR])
    demo.queue(max_size=16, default_concurrency_limit=4).launch(
        css=CSS, theme=THEME, js=JS, show_error=False, max_file_size="20mb",
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")), allowed_paths=[str(ASSET_DIR), str(setup.EXAMPLES)],
    )
