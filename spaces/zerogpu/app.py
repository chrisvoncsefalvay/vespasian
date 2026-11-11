"""Vespasian on Hugging Face ZeroGPU."""

import spaces  # Register ZeroGPU before importing the model runtime.
spaces.disable_gradio_auto_wrap()

import html
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

import hub
import runtime
from gpu import dispatch, model_availability
from ui import ASSET_DIR, CSS, THEME, build_ui, render_dashboard

ROOT = Path(__file__).resolve().parent
FIELDS = (
    "target_type", "model", "dataset_location", "dataset_source", "dataset_file",
    "dataset_config", "dataset_split", "dataset_mode", "prompt_field", "target_field", "text_field",
    "example_limit", "test_dataset_source", "test_dataset_config", "test_dataset_split", "test_example_limit",
    "goal", "controls", "metric", "editor", "editor_model", "api_key", "hf_token", "budget", "max_turns",
    "max_new_tokens", "health_checks", "health_count", "health_max_drop", "save_trajectory",
)
PREVIEW_FIELDS = (
    "dataset_location", "dataset_source", "dataset_file", "dataset_config", "dataset_split", "dataset_mode",
    "prompt_field", "target_field", "text_field", "example_limit", "hf_token",
    "test_dataset_source", "test_dataset_config", "test_dataset_split", "test_example_limit",
)


def safe_error(error, *secrets):
    text = str(error)
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "[redacted]")
    return text[:1000]


def create_app():
    demo, controls = build_ui()
    available = model_availability()
    choices = [entry["id"] for entry in available if entry["available"]]
    if not choices and os.environ.get("VESPASIAN_SPACE_SKIP_MODEL_LOAD") == "1":
        choices = ["Qwen/Qwen3.5-0.8B", "google/gemma-3-1b-it"]
    controls["model"].choices = [(name, name) for name in choices]
    controls["model"].value = choices[0] if choices else None

    def collect(values):
        return {name: values[control] for name, control in controls.items() if control in values}

    def account(override, oauth_token: gr.OAuthToken | None, profile: gr.OAuthProfile | None):
        token = hub.token_for(oauth_token, override)
        try:
            name, datasets = hub.account_datasets(token)
        except Exception as error:
            return (f'<div class="vs-profile"><p>{html.escape(safe_error(error, token))}</p></div>',
                    gr.skip(), gr.skip(), gr.skip(), gr.skip())
        if name:
            display = f'<div class="vs-profile"><span class="vs-connected">Connected</span><strong>{html.escape(name)}</strong><p>Your private datasets and organisation repositories are available.</p></div>'
            return (display, gr.update(choices=datasets), gr.update(choices=datasets),
                    gr.update(placeholder=f"{name}/vespasian-model"), gr.update(placeholder=f"{name}/vespasian-trajectories"))
        return ('<div class="vs-profile"><p>Connect your Hugging Face account to use your datasets and save results.</p></div>',
                gr.update(choices=[]), gr.update(choices=[]), gr.skip(), gr.skip())

    def inspect_dataset(values, oauth_token: gr.OAuthToken | None):
        config = collect(values)
        token = hub.token_for(oauth_token, config.get("hf_token"))
        try:
            if config.get("dataset_location") == "file":
                if not config.get("dataset_file"):
                    raise ValueError("Choose a dataset file")
                rows = hub.read_upload(config["dataset_file"], config)
                result = {"repo": Path(config["dataset_file"]).name, "rows": rows[:3], "mapped_rows": rows[:3]}
            else:
                result = hub.preview(config, token)
            return (hub.preview_html(result),
                    gr.update(choices=result.get("configs", []), value=result.get("config") or config.get("dataset_config")),
                    gr.update(choices=result.get("splits", []), value=result.get("split") or config.get("dataset_split")),
                    result.get("prompt_field") or config.get("prompt_field", ""),
                    result.get("target_field") or config.get("target_field", ""),
                    result.get("text_field") or config.get("text_field", ""))
        except Exception as error:
            return (f'<p class="vs-preview-note">{html.escape(safe_error(error, token))}</p>', *[gr.skip() for _ in range(5)])

    def inspect_test(values, oauth_token: gr.OAuthToken | None):
        config = collect(values)
        for name in ("dataset_source", "dataset_config", "dataset_split"):
            config[name] = config.get("test_" + name)
        token = hub.token_for(oauth_token, config.get("hf_token"))
        try:
            result = hub.preview(config, token)
            return (hub.preview_html(result),
                    gr.update(choices=result.get("configs", []), value=result.get("config")),
                    gr.update(choices=result.get("splits", []), value=result.get("split")))
        except Exception as error:
            return (f'<p class="vs-preview-note">{html.escape(safe_error(error, token))}</p>', gr.skip(), gr.skip())

    def display_run(run, state):
        finished = state["finished"]
        has_checkpoint = finished and (run.output / "report.json").is_file() and not state["error"]
        has_trajectory = finished and (run.output / "trajectory.jsonl").is_file()
        return (state, render_dashboard(state), state["error"] or state["status"],
                gr.update(value=str(run.archive) if run.archive else None, visible=bool(run.archive)),
                gr.update(interactive=finished), gr.update(interactive=not finished),
                gr.update(interactive=has_checkpoint), gr.update(interactive=has_trajectory),
                gr.update(active=not finished))

    def start(values, request: gr.Request, oauth_token: gr.OAuthToken | None):
        config = collect(values)
        token = hub.token_for(oauth_token, config.pop("hf_token", ""))
        try:
            run = runtime.begin(request.session_hash, config, token)
            run.start(config, token, available)
        except Exception as error:
            raise gr.Error(safe_error(error, token, config.get("api_key"))) from None
        return display_run(run, run.state())

    def advance(state, request: gr.Request):
        try:
            run = runtime.get_run(request.session_hash, state)
        except ValueError as error:
            raise gr.Error(str(error)) from None
        # Each timer event is a new queued request with fresh ZeroGPU headers.
        # The runtime stops at an allocation boundary before this event ends.
        for current in run.advance(dispatch):
            yield display_run(run, current)

    def stop(state, request: gr.Request):
        run = runtime.get_run(request.session_hash, state)
        run.stop()
        return "Stopping at the current operation"

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
            result = hub.publish_run(run.output, kind=kind, repo=repo.strip(), visibility=visibility, create=create, token=token)
            prefix = "datasets/" if kind == "trajectory" else ""
            return f"Saved to [huggingface.co/{prefix}{repo}](https://huggingface.co/{prefix}{repo})"
        except Exception as error:
            raise gr.Error(safe_error(error, token)) from None

    with demo:
        account_outputs = [controls[name] for name in ("profile_html", "dataset_source", "test_dataset_source", "model_repo", "trajectory_repo")]
        demo.load(account, inputs=[controls["hf_token"]], outputs=account_outputs, api_name=False)
        controls["refresh_datasets"].click(account, inputs=[controls["hf_token"]], outputs=account_outputs, api_name=False)
        preview_inputs = {controls[name] for name in PREVIEW_FIELDS}
        controls["preview_button"].click(inspect_dataset, inputs=preview_inputs,
            outputs=[controls[name] for name in ("preview_html", "dataset_config", "dataset_split", "prompt_field", "target_field", "text_field")],
            api_name="preview_dataset", concurrency_limit=4)
        if "test_preview_button" in controls:
            controls["test_preview_button"].click(inspect_test, inputs=preview_inputs,
                outputs=[controls.get("test_preview_html", controls["preview_html"]), controls["test_dataset_config"], controls["test_dataset_split"]],
                api_name="preview_test_dataset", concurrency_limit=4)
        run_outputs = [controls[name] for name in ("state", "dashboard_html", "status", "download", "run_button", "stop_button", "publish_model_button", "publish_trajectory_button", "run_timer")]
        controls["run_button"].click(start, inputs={controls[name] for name in FIELDS},
            outputs=run_outputs, api_name="run", concurrency_limit=runtime.MAX_ACTIVE)
        controls["run_timer"].tick(advance, inputs=controls["state"], outputs=run_outputs,
            api_name="advance", api_visibility="undocumented", queue=True,
            trigger_mode="once", show_progress="hidden", concurrency_limit=runtime.MAX_ACTIVE,
            concurrency_id="gpu-allocations")
        controls["stop_button"].click(stop, inputs=controls["state"], outputs=controls["status"], queue=False, api_name=False)
        for kind, callback in (("model", publish_model), ("trajectory", publish_trajectory)):
            controls[f"publish_{kind}_button"].click(callback,
                inputs=[controls[name] for name in ("state", f"{kind}_repo", f"{kind}_visibility", f"{kind}_create", "hf_token")],
                outputs=controls["status"], api_name=False, concurrency_limit=2)
    return demo


demo = create_app()

if __name__ == "__main__":
    runtime.start_cleanup()
    gr.set_static_paths(paths=[ASSET_DIR])
    demo.queue(max_size=16, default_concurrency_limit=4).launch(
        css=CSS, theme=THEME, show_error=False, max_file_size="20mb",
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
        allowed_paths=[str(ASSET_DIR)],
    )
