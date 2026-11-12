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
                   "max_turns": int | None, "health_checks": list[str]},
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
import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any

import gradio as gr


def _release_version(directory: Path) -> str | None:
    """Prefer the deployed bundle version, then the local Cargo package."""
    for path, parse, keys in (
        (directory / "PROVENANCE.json", json.loads, ("version",)),
        (directory.parent.parent / "Cargo.toml", tomllib.loads, ("package", "version")),
    ):
        try:
            value = parse(path.read_text(encoding="utf-8"))
            for key in keys:
                value = value[key]
            if isinstance(value, str) and re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", value):
                return value
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


RELEASE_VERSION = _release_version(Path(__file__).resolve().parent)
RELEASE_LABEL = f"v{RELEASE_VERSION}" if RELEASE_VERSION else "Version unavailable"
ASSET_DIR = Path(__file__).resolve().parent / "assets"
CSS = "\n".join((ASSET_DIR / name).read_text(encoding="utf-8") for name in ("workbench.css", "themes.css")).replace(
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

JS = (ASSET_DIR / "workbench.js").read_text(encoding="utf-8") if (ASSET_DIR / "workbench.js").exists() else "() => {}"

from dashboard import render_dashboard
import setup_helpers as setup
from limits import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MAX_SEQ_LENGTH, DEFAULT_MAX_RELATIVE_NORM, space_limits

_MARK = '<svg viewBox="0 0 30 30" aria-hidden="true"><path d="M3 7 15 24 27 7M9 7l6 8 6-8" fill="none" stroke="currentColor" stroke-width="1.7"/></svg>'
HEADER = f"""<header class="vs-masthead">
<a class="vs-wordmark" href="https://github.com/chrisvoncsefalvay/vespasian" target="_blank" rel="noopener noreferrer">{_MARK}<h1>Vespasian</h1></a>
<span class="vs-masthead-meta">The Weight Space Editing Assistant/{RELEASE_LABEL}</span>
<nav class="vs-header-links" aria-label="Appearance and help"><button type="button" data-vs-theme-toggle aria-label="Switch to light mode">Light mode</button><button type="button" data-vs-help>Quick guide</button><a href="https://github.com/chrisvoncsefalvay/vespasian#readme" target="_blank" rel="noopener noreferrer">Documentation ↗</a></nav>
</header>"""
PREVIEW_EMPTY = '<div class="vs-preview-empty"><h3>Preview your examples</h3><p>Choose a dataset and inspect it to select fields and check the examples the model will use.</p></div>'


def heading(title, description):
    return f'<div class="vs-step-intro"><h2 tabindex="-1">{title}</h2><p>{description}</p></div>'


def build_ui() -> tuple[gr.Blocks, dict[str, Any]]:
    controls: dict[str, Any] = {}
    limits = space_limits()
    with gr.Blocks(title="Vespasian · Weight Space Editing Assistant", fill_width=True, delete_cache=(600, 7200)) as demo:
        gr.HTML(HEADER, elem_id="vs-header")
        with gr.Row(elem_id="vs-account", equal_height=False):
            controls["profile_html"] = gr.HTML('<div class="vs-profile"><p>Public datasets are ready to use. Connect your account for private data and Hub exports.</p></div>', scale=3, elem_id="vs-profile")
            controls["login_button"] = gr.LoginButton("Sign in with Hugging Face", scale=1, min_width=230, elem_id="vs-login")
            with gr.Accordion("Access token", open=False, elem_id="vs-token-access"):
                controls["hf_token"] = gr.Textbox(type="password", label="Hugging Face token override", placeholder="Optional", info="Use this to connect a different account or access token.")
        with gr.Column(elem_id="vs-workbench"):
            with gr.Column(elem_id="vs-setup") as controls["setup_panel"]:
                controls["return_results"] = gr.Button("Return to results", visible=False, elem_id="vs-return-results")
                controls["starter_status"] = gr.HTML('', elem_id="vs-starter-status")
                with gr.Tabs(elem_id="vs-setup-tabs", selected="target") as controls["setup_tabs"]:
                    with gr.Tab("01 Model", id="target", elem_id="vs-step-target"):
                        gr.HTML(heading("Choose the model to edit", "Start with the small model whose behaviour you want to change. The frontier editor is selected in Runtime."))
                        controls["error_target"] = gr.HTML("")
                        controls["model"] = gr.Dropdown(choices=[("Qwen 3.5 · 0.8B", "Qwen/Qwen3.5-0.8B"), ("Gemma 3 · 1B", "google/gemma-3-1b-it")], value="Qwen/Qwen3.5-0.8B", label="Base model", elem_id="vs-model")
                        gr.HTML('<p class="vs-inline-note">This list contains the models configured for this ZeroGPU deployment. The terminal can attempt compatible local or Hub causal language models; capability checks determine available edits. Qwen 3.5 0.8B and Gemma 3 1B have recorded real-model checks, not a guarantee for other revisions or every operation.</p>')
                        controls["next_behaviour"] = gr.Button("Continue to behaviour →", variant="primary")
                    with gr.Tab("02 Behaviour", id="behaviour", elem_id="vs-step-behaviour"):
                        gr.HTML(heading("What should change?", "Choose the behaviour, define success and decide what should remain intact."))
                        controls["error_behaviour"] = gr.HTML("")
                        with gr.Row(elem_id="vs-behaviour-row", equal_height=False):
                            with gr.Column(scale=3, min_width=280):
                                controls["preset"] = gr.Dropdown(choices=setup.starter_presets(), value=None, label="Start from an example", info="Optional. Review and change every setting before running.", elem_id="vs-preset")
                                controls["hf_task"] = gr.Dropdown(choices=setup.task_choices(), value=None, label="Hugging Face task", info="Choose the task, or inspect a dataset for a suggestion. Missing or ambiguous tags need your choice.", elem_id="vs-hf-task")
                                controls["task_help"] = gr.HTML(setup.task_help({}), elem_id="vs-task-help")
                                controls["dataset_mode"] = gr.Radio(choices=[("Paired examples", "paired"), ("Reference writing", "corpus"), ("Prompts with a rubric", "prompts")], value="paired", label="Data shape", elem_id="vs-dataset-mode")
                                controls["mode_help"] = gr.HTML('<p class="vs-inline-note">Each example contains an input and the reference answer you want the model to produce.</p>')
                                controls["goal"] = gr.Textbox(label="Editing objective", placeholder="State the behaviour you want to improve.", lines=4, elem_id="vs-goal")
                                controls["objective_suggestion"] = gr.HTML('<p class="vs-suggestion">Return answers that agree with the dataset references.</p>', elem_id="vs-objective-suggestion")
                                controls["use_suggestion"] = gr.Button("Use suggested objective", variant="secondary", elem_id="vs-use-suggestion")
                            with gr.Column(scale=2, min_width=280, elem_id="vs-preservation"):
                                with gr.Accordion("Advanced evaluation and preservation", open=False, elem_id="vs-advanced-evaluation"):
                                    controls["target_type"] = gr.Dropdown(choices=[("Factual knowledge", "factual"), ("Writing style", "style"), ("Output format", "format"), ("Classification", "classification"), ("Custom behaviour", "custom")], value="factual", label="Type of change", elem_id="vs-target-type")
                                    controls["metric"] = gr.Dropdown(choices=setup.metric_choices("paired"), value="auto", label="Objective metric", elem_id="vs-metric")
                                    controls["metric_help"] = gr.HTML('<p class="vs-inline-note">The editor assesses your examples and selects a compatible metric before proposing edits.</p>')
                                    gr.HTML('<h3>Preservation</h3><p class="vs-inline-note">Track the capabilities you want to keep while the objective changes.</p>')
                                    controls["controls"] = gr.Textbox(label="Behaviour to preserve", placeholder="For example, answer in the language of the question.", lines=3, info="Optional. Add one requirement per line.", elem_id="vs-controls")
                                    controls["health_checks"] = gr.Dropdown(choices=[("Copying", "copying"), ("Arithmetic", "arithmetic"), ("JSON formatting", "json"), ("MMLU · anatomy", "mmlu:anatomy"), ("MMLU · high school mathematics", "mmlu:high_school_mathematics")], value=["copying"], multiselect=True, allow_custom_value=True, max_choices=4, label="Preservation checks", info="Choose up to four. Add an MMLU subject as mmlu:subject.", elem_id="vs-health-checks")
                                    controls["health_count"] = gr.Number(value=4, minimum=1, maximum=16, step=1, precision=0, label="Examples per check", elem_id="vs-health-count")
                                    controls["health_policy"] = gr.Radio(choices=[("Track scores", "track"), ("Reject above a limit", "enforce")], value="track", label="Regression policy", elem_id="vs-health-policy")
                                    controls["health_limit_pp"] = gr.Number(value=10, minimum=0, maximum=100, step=1, visible=False, label="Maximum drop · percentage points", elem_id="vs-health-limit")
                                    controls["health_help"] = gr.HTML('<p class="vs-inline-note">Scores are measured alongside the objective. Tracking does not reject an edit when a check declines.</p>')
                        with gr.Row(elem_id="vs-setup-actions-behaviour"):
                            controls["back_target"] = gr.Button("← Model", variant="secondary")
                            controls["next_data"] = gr.Button("Continue to data →", variant="primary")
                    with gr.Tab("03 Data", id="data", elem_id="vs-step-data"):
                        gr.HTML(heading("Choose the examples", "Inspect your data, confirm the fields and set how many examples to use."))
                        controls["error_data"] = gr.HTML("")
                        with gr.Row(elem_id="vs-data-row", equal_height=False):
                            with gr.Column(scale=1, min_width=290, elem_id="vs-data-fields"):
                                controls["dataset_location"] = gr.Radio(choices=[("Hugging Face", "hub"), ("Upload a file", "file")], value="hub", label="Dataset location", elem_id="vs-dataset-location")
                                with gr.Column(elem_id="vs-hub-fields") as controls["hub_fields"]:
                                    controls["dataset_catalog"] = gr.Radio(choices=[("Public datasets", "public"), ("My datasets", "account"), ("Recent", "recent")], value="public", label="Browse datasets", elem_id="vs-dataset-catalog")
                                    with gr.Row():
                                        controls["dataset_query"] = gr.Textbox(label="Search public datasets", placeholder="Name or subject", scale=3, elem_id="vs-dataset-query")
                                        controls["search_datasets"] = gr.Button("Search", scale=1, min_width=85, elem_id="vs-search-datasets")
                                    controls["include_untagged"] = gr.Checkbox(value=False, label="Search all tasks, including untagged datasets", info="Manual repository entry is always available.")
                                    controls["dataset_source"] = gr.Dropdown(choices=["openai/gsm8k"], allow_custom_value=True, label="Dataset repository", info="Choose a result or enter organisation/dataset.", elem_id="vs-dataset-source")
                                    controls["catalog_status"] = gr.HTML("")
                                    controls["refresh_datasets"] = gr.Button("Refresh my datasets", variant="secondary", elem_id="vs-refresh-datasets")
                                    with gr.Row():
                                        controls["dataset_config"] = gr.Dropdown(choices=[], allow_custom_value=True, label="Configuration", min_width=120, elem_id="vs-dataset-config")
                                        controls["dataset_split"] = gr.Dropdown(choices=["train", "validation", "test"], value="train", allow_custom_value=True, label="Source split", min_width=120, elem_id="vs-dataset-split")
                                controls["dataset_file"] = gr.File(label="Dataset file", file_types=[".jsonl", ".json", ".txt", ".csv", ".parquet"], type="filepath", visible=False, elem_id="vs-dataset-file")
                                controls["example_limit"] = gr.Number(value=16, minimum=5, maximum=128, step=1, precision=0, label="Example limit", elem_id="vs-example-limit")
                                controls["split_summary"] = gr.HTML('<p class="vs-inline-note">The preview will show how examples are divided between editing, validation and testing.</p>')
                                controls["preview_button"] = gr.Button("Inspect dataset", variant="primary", elem_id="vs-preview-button")
                                with gr.Accordion("Field mapping", open=True, elem_id="vs-fields"):
                                    controls["prompt_field"] = gr.Dropdown(choices=[], allow_custom_value=True, label="Prompt field", info="Inspect the dataset to select a field or enter a nested path.", elem_id="vs-prompt-field")
                                    controls["target_field"] = gr.Dropdown(choices=[], allow_custom_value=True, label="Reference field", elem_id="vs-target-field")
                                    controls["text_field"] = gr.Dropdown(choices=[], allow_custom_value=True, label="Text field", visible=False, elem_id="vs-text-field")
                                with gr.Accordion("Held-out test data", open=False):
                                    controls["test_dataset_source"] = gr.Dropdown(choices=[], allow_custom_value=True, label="Test dataset repository", info="Optional. It uses the same field mapping.", elem_id="vs-test-source")
                                    with gr.Row():
                                        controls["test_dataset_config"] = gr.Dropdown(choices=[], allow_custom_value=True, label="Test configuration", min_width=120)
                                        controls["test_dataset_split"] = gr.Dropdown(choices=["train", "validation", "test"], value="test", allow_custom_value=True, label="Test source split", min_width=120)
                                    controls["test_example_limit"] = gr.Number(value=8, minimum=1, maximum=128, step=1, precision=0, label="Test example limit")
                                    controls["test_preview_button"] = gr.Button("Inspect test dataset", variant="secondary", elem_id="vs-test-preview-button")
                            with gr.Column(scale=1, min_width=290, elem_id="vs-preview-column"):
                                with gr.Tabs(selected="edit", elem_id="vs-preview-tabs") as controls["preview_tabs"]:
                                    with gr.Tab("Edit/validation", id="edit"):
                                        controls["preview_html"] = gr.HTML(PREVIEW_EMPTY, elem_id="vs-preview")
                                    with gr.Tab("Held-out test", id="test"):
                                        controls["test_preview_html"] = gr.HTML('<div class="vs-preview-empty"><h3>Independent test examples</h3><p>Add a test repository to measure the retained checkpoint on separate examples.</p></div>', elem_id="vs-test-preview")
                        with gr.Row(elem_id="vs-setup-actions-data"):
                            controls["back_behaviour"] = gr.Button("← Behaviour", variant="secondary")
                            controls["next_editor"] = gr.Button("Continue to runtime →", variant="primary")
                    with gr.Tab("04 Runtime", id="editor", elem_id="vs-step-editor"):
                        gr.HTML(heading("Set the runtime and frontier editor", "Choose the editor, provide its API key and set your spending limit. Secondary settings have ready-to-use defaults."))
                        controls["error_editor"] = gr.HTML("")
                        with gr.Row(equal_height=False):
                            with gr.Column(min_width=280):
                                controls["editor"] = gr.Radio(choices=[("OpenAI", "openai"), ("Anthropic", "anthropic")], value="openai", label="Editor provider", elem_id="vs-editor")
                                controls["editor_model"] = gr.Dropdown(choices=setup.editor_choices("openai"), value="gpt-6.1-sol", allow_custom_value=True, label="Editor model", info="Use a listed model with supported strict-budget pricing. Custom-price estimates require the terminal with --budget-policy auto.", elem_id="vs-editor-model")
                                controls["api_key"] = gr.Textbox(type="password", label="Editor API key", placeholder="Your provider’s API key", elem_id="vs-api-key")
                            with gr.Column(min_width=280):
                                controls["budget"] = gr.Number(value=10.0, minimum=0.01, maximum=100, step=0.25, label="Editor budget · USD", elem_id="vs-budget")
                                gr.HTML('<p class="vs-inline-note">Default: low editor reasoning, target thinking off, no turn cap. Generation limits are reviewed below. The spending limit includes every editor decision and tool request.</p>')
                                controls["pricing_html"] = gr.HTML("", elem_id="vs-pricing")
                        with gr.Accordion("Advanced runtime settings", open=False, elem_id="vs-advanced-runtime"):
                            with gr.Row():
                                controls["max_turns"] = gr.Textbox(value="", placeholder="No turn cap", label="Maximum edit turns · optional", info="Leave blank to use only the spending limit. There is no default turn cap.", elem_id="vs-max-turns")
                                controls["reasoning_effort"] = gr.Dropdown(choices=["low", "medium", "high"], value="low", label="Editor reasoning effort", info="Sent to the selected provider; unsupported requests stop without substitution.", elem_id="vs-reasoning-effort")
                                controls["max_new_tokens"] = gr.Number(value=min(DEFAULT_MAX_NEW_TOKENS, limits["max_new_tokens"]), minimum=1, maximum=limits["max_new_tokens"], step=1, precision=0, label="Maximum generated tokens per answer", info="Includes any target thinking tokens. Reaching the cap truncates output.", elem_id="vs-max-new-tokens")
                            with gr.Row():
                                controls["thinking_mode"] = gr.Dropdown(choices=[("Off · preserve starter baseline", "off"), ("On · compatible chat models", "on"), ("Auto · model template default", "auto")], value="off", label="Target model thinking", info="Separate from editor reasoning. On requires a compatible chat template; auto may enable thinking and use more tokens.", elem_id="vs-thinking-mode")
                                controls["max_seq_length"] = gr.Number(value=min(DEFAULT_MAX_SEQ_LENGTH, limits["max_seq_length"]), minimum=32, maximum=limits["max_seq_length"], step=1, precision=0, label="Maximum sequence tokens", info="Total context limit: the prompt and requested output must fit. Model support and GPU quota still apply.", elem_id="vs-max-seq-length")
                                controls["max_relative_norm"] = gr.Number(value=DEFAULT_MAX_RELATIVE_NORM, minimum=0.000001, maximum=1, label="Direction-edit relative norm cap", info="Review a value from 0.000001 to 1 (default 0.1). Bounds the new weight-direction edit relative to each pre-edit tensor; other edits retain their safeguards.", elem_id="vs-max-relative-norm")
                            controls["prompt_format"] = gr.Dropdown(choices=[("Use model chat template", "auto"), ("Require chat template", "chat"), ("Plain text prompts", "plain")], value="auto", label="Prompt format", info="Models without a chat template require an explicit plain-text choice.")
                            controls["save_trajectory"] = gr.Checkbox(value=True, label="Include the edit trajectory in the run bundle")
                            gr.HTML('<p class="vs-inline-note">This Space uses strict budgeting with supported model prices and no price overrides. For custom-price estimates, use the terminal with --budget-policy auto; estimated spending is not a strict cap.</p>')
                            with gr.Row():
                                controls["input_price"] = gr.Textbox(value="", label="Input price", visible=False, interactive=False, elem_id="vs-input-price")
                                controls["output_price"] = gr.Textbox(value="", label="Output price", visible=False, interactive=False, elem_id="vs-output-price")
                        controls["save_format"] = gr.Radio(choices=[("Materialised model and compact patches", "model"), ("Compact patches only", "patch")], value="model", label="Final output", info="Latest/best patches are checkpointed during the run. Patch-only output needs the exact base model to replay.")
                        with gr.Accordion("Private patch backup · optional", open=False, elem_id="vs-patch-backup"):
                            controls["patch_backup"] = gr.Checkbox(value=False, label="Back up latest/best patches after measured rounds to my named private repository")
                            controls["patch_repo"] = gr.Textbox(label="Private patch repository", placeholder="your-account/private-patches", info="Shares patch weights and base-model metadata only, not your private objective or examples. Uses your signed-in account or session token.")
                            controls["patch_create"] = gr.Checkbox(value=False, label="Create this private repository if it does not exist")
                            gr.HTML('<p class="vs-inline-note">Local patches need no account. Space storage is temporary and does not survive pod replacement; download or explicitly enable private backup to retain them.</p>')
                        with gr.Accordion("Deployment and spending limits", open=False, elem_id="vs-deployment-limits"):
                            gr.HTML(f'<p class="vs-inline-note">The editor budget covers frontier API calls; Hugging Face manages GPU quota. This deployment allows up to $100 per run, 128 examples per dataset, four preservation checks with 16 examples each, {limits["max_new_tokens"]:,} generated tokens per answer and {limits["max_seq_length"]:,} sequence tokens. Larger limits require operator configuration and do not guarantee long runs fit ZeroGPU memory or time quota. Seed 42, editor output cap 2,048 tokens and preservation tolerance 0 match the terminal defaults. Keep this tab open; five minutes without a browser heartbeat cancels the run. API cost is reserved before each request at the configured token rates, not enforced as a provider billing ceiling. Terminal CLI-login sessions instead report estimated spend between calls.</p>')
                        with gr.Row(elem_id="vs-setup-actions-editor"):
                            controls["back_data"] = gr.Button("← Data", variant="secondary")
                            controls["next_review"] = gr.Button("Review experiment →", variant="primary")
                    with gr.Tab("05 Review", id="review", elem_id="vs-step-review"):
                        gr.HTML(heading("Review the experiment", "Check the data, evaluation and spending limits before starting."))
                        controls["error_review"] = gr.HTML("")
                        controls["review_html"] = gr.HTML('<p class="vs-inline-note">Complete the previous steps to review your experiment.</p>', elem_id="vs-review")
                        gr.HTML('<p class="vs-run-note">Starting sends your objective and selected examples to the editor provider using your API key. Keep this tab open while running. Local downloads expire after two hours; private patch backup is optional.</p>')
                        with gr.Row(elem_id="vs-setup-actions-review"):
                            controls["back_editor"] = gr.Button("← Runtime", variant="secondary")
                            controls["run_button"] = gr.Button("Start experiment →", variant="primary", elem_id="vs-run-button")
            with gr.Column(visible=False, elem_id="vs-monitor") as controls["monitor_panel"]:
                with gr.Row(elem_id="vs-live-actions"):
                    controls["status"] = gr.Markdown("Ready to configure.", scale=4, elem_id="vs-status", container=False)
                    controls["stop_button"] = gr.Button("Stop", variant="stop", interactive=False, scale=1, min_width=90, elem_id="vs-stop-button")
                    controls["new_experiment"] = gr.Button("New experiment", interactive=False, scale=1, min_width=160, elem_id="vs-new-experiment")
                controls["dashboard_html"] = gr.HTML(render_dashboard(), elem_id="vs-dashboard-output")
                with gr.Column(visible=False, elem_id="vs-results") as controls["results_panel"]:
                    gr.HTML('<h2>Save your results</h2><p class="vs-inline-note">Download this run or save its model and trajectory to your Hugging Face account.</p>')
                    controls["patch_download"] = gr.File(label="Download latest/best patches", interactive=False, visible=False, elem_id="vs-patch-download")
                    controls["download"] = gr.File(label="Download run bundle", interactive=False, visible=False, elem_id="vs-download")
                    gr.HTML('<p class="vs-download-note">The run bundle is available for two hours.</p>')
                    with gr.Row(equal_height=False):
                        with gr.Column(min_width=280):
                            controls["model_repo"] = gr.Textbox(label="Model repository", placeholder="your-account/edited-model")
                            controls["model_visibility"] = gr.Radio(choices=[("Private", "private"), ("Public", "public")], value="private", label="Model visibility")
                            controls["model_create"] = gr.Checkbox(value=True, label="Create the model repository if needed")
                            controls["publish_model_button"] = gr.Button("Save edited model to Hub", interactive=False)
                        with gr.Column(min_width=280):
                            controls["trajectory_repo"] = gr.Textbox(label="Trajectory dataset repository", placeholder="your-account/edit-trajectories")
                            controls["trajectory_visibility"] = gr.Radio(choices=[("Private", "private"), ("Public", "public")], value="private", label="Trajectory visibility")
                            controls["trajectory_create"] = gr.Checkbox(value=True, label="Create the trajectory repository if needed")
                            controls["publish_trajectory_button"] = gr.Button("Save trajectory to Hub", interactive=False)
                    with gr.Accordion("Save compact patches privately", open=False):
                        controls["patch_save_repo"] = gr.Textbox(label="Private patch repository", placeholder="your-account/private-patches")
                        controls["patch_save_consent"] = gr.Checkbox(value=False, label="Share this run's latest/best patch weights and base-model metadata with this private repository")
                        controls["patch_save_create"] = gr.Checkbox(value=False, label="Create this private repository if needed")
                        controls["publish_patches_button"] = gr.Button("Save latest/best patches privately", interactive=False)
                    controls["publish_status"] = gr.Markdown("", elem_id="vs-publish-status")
        gr.HTML('<footer class="vs-page-footer"><span>Vespasian/The Weight Space Editing Assistant</span><button type="button" data-vs-help>Quick guide</button><a href="https://github.com/chrisvoncsefalvay/vespasian" target="_blank" rel="noopener noreferrer">Source code ↗</a></footer>')
        controls["state"] = gr.State({"events": [], "status": "idle", "config": {}})
        controls["selection"] = gr.State({"turn": None, "split": "all", "outcome": "all", "page": 0, "follow": True})
        controls["preview_state"] = gr.State({})
        controls["test_preview_state"] = gr.State({})
        controls["provider_memory"] = gr.State({"_provider": "openai"})
        controls["account_choices"] = gr.State([])
        controls["recent_datasets"] = gr.State([])
        starter_metadata = html.escape(json.dumps(setup.quick_starter_cards()), quote=True)
        gr.HTML(f'<div id="vs-starter-metadata" data-vs-starters="{starter_metadata}" hidden></div>')
        controls["starter_input"] = gr.Textbox(value="", elem_id="vs-starter-input", elem_classes=["vs-internal"], container=False)
        controls["starter_submit"] = gr.Button("Load starter settings", elem_id="vs-starter-submit", elem_classes=["vs-internal"])
        controls["selection_input"] = gr.Textbox(value="{}", elem_id="vs-selection-input", elem_classes=["vs-internal"], container=False)
        controls["selection_submit"] = gr.Button("Update view", elem_id="vs-selection-submit", elem_classes=["vs-internal"])
        controls["run_timer"] = gr.Timer(0.5, active=False)
        for component in controls.values():
            if isinstance(component, (gr.Textbox, gr.Dropdown, gr.Radio, gr.Number, gr.Slider, gr.Checkbox)):
                component.interactive = True
    return demo, controls
