---
title: Vespasian
emoji: 🟠
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12.12
app_file: app.py
pinned: true
license: apache-2.0
short_description: The Weight Space Editing Assistant
hf_oauth: true
hf_oauth_scopes:
  - read-repos
  - gated-repos
  - write-repos
---

# Vespasian

**The Weight Space Editing Assistant**

Connect your Hugging Face account, choose a dataset and give a small model a new
behaviour. Vespasian asks a frontier model to propose direct tensor edits, measures
each candidate and keeps the changes that improve the objective.

The workbench connects the objective curve, health checks, generated answers,
edited layers and candidate ancestry. Select a turn to inspect its changes and
compare its answers with the baseline.

## Using your Hugging Face account

1. **Choose the target model.** Select the small model to edit. The frontier
   editor is a separate model selected in Runtime.
2. **Define the behaviour.** Start with your own objective or load a complete
   example. Choose paired examples, reference writing or prompts with a scoring
   rubric. Preservation checks track scores by default. Select a regression
   limit to reject edits that weaken a check beyond that limit.
3. **Inspect your data.** Search public datasets, upload a file or sign in to
   browse your datasets and organisation repositories. Choose fields from the
   preview and set the example limit. Editing and held-out test data have
   separate previews.
4. **Set runtime options and the editor.** Select an OpenAI or Anthropic model, enter
   its API key and set a USD inference budget (default **$10**). Reasoning effort
   is selectable as low, medium or high, with **low** selected by default.
   Leave the optional turn limit blank for no turn cap. Custom models accept explicit
   token prices. The controller reserves the next call's cost before issuing it.
5. **Review and start.** Check the objective, metric, example counts,
   preservation policy and budget together before starting the experiment.
6. **Inspect and save the result.** Select turns in the graph or ancestry view
   to see their tensors, measurements and acceptance decisions. Compare baseline
   and edited answers, then review final test results and checkpoint verification.
   Download the run or save its model and trajectory to public or private Hub
   repositories.

The first-use modal has four sections: **Welcome** (Hugging Face login or
**Proceed without login**), **Weight-space editing**, **How to use** and
**Quick starters**. Four bundled starters cover short maths answers, odd/even
codes, a geometric alias and canonical JSON extraction. Selecting one fills
real settings and data and opens Data for inspection; it never starts a run.
The selected model and editor credentials are preserved. Backup remains off
until you explicitly authorize a named private repository.

These original synthetic exercises are adaptations, not benchmark reproductions
or proven improvements. Starter settings and scoring caveats are in the bundled `examples/STARTERS.md`.
Dismiss the modal with Close guide, the close button or Escape. Your browser remembers
the dismissal, and **Quick guide** reopens it at any time.

HF access comes from your connected account. The optional token override supports
other Hub credentials. Editor keys stay within the session's controller process.

## Model execution

The default Space model is Qwen3.5-0.8B. Operators may configure other compatible
causal models at startup; this is not a promise that every model fits ZeroGPU.
The parameter-count check runs after loading, not before downloading or allocating
the model. Dataset access and frontier editing use CPU time. GPU allocations cover model loading, generation and tensor operations.
Exact tensor snapshots carry the retained checkpoint between allocations.
Each allocation starts from a fresh queued browser request, so Hugging Face
renews GPU authorisation throughout a longer experiment. Keep the workbench open
while the experiment runs.

Saved checkpoints are loaded from disk in a fresh worker and checked against the
retained rollouts. The run bundle contains measurements, checkpoint hashes,
dataset revisions, budget receipts and the optional editor trajectory.
Downloads remain available for two hours after a run finishes. Hub exports stay
in the destination repository. Local Space storage is temporary.

## Tasks, compact patches and private backup

Choose the Hugging Face task separately from the behaviour you want to change.
Model and dataset metadata suggest tasks and filter dataset search; they do not
prove compatibility. Missing or ambiguous tags require a choice. Inspect the
mapped prompts, context and text targets before starting. Secondary generation,
health-check and deployment settings are in collapsed Advanced sections.

Each candidate is saved locally before evaluation. Latest and best retained
patches have separate atomic pointers, so rejected candidates never replace best.
Sparse token edits store only changed rows; dense edits store changed tensors.
Replay verifies the source identity, tensor layout and dtype, tied aliases and
exact stored values. At completion choose a compact patch or a materialised full
model. Patch-only results can later be materialised with the terminal command
`vespasian materialize-patch patches.zip --output edited-model`.

Local/download is the default. Automatic backup after measured rounds requires
explicitly enabling it, naming a **private** repository and providing your session
HF authorization. Creating a missing private repository requires a separate choice.
Connecting an account alone never enables uploads. There is no owner-token or
public-repository fallback. Patches omit private objectives, datasets and editor
transcripts. A failed or unverified backup stops automatic work and leaves the
local patch available to download. Download or explicitly back up results before
the Space's temporary retention expires.

## Terminal and Space parity

The Space launches the same Rust controller and Python editing engine as the
terminal. The editor briefing, strategy catalogue, operation definitions,
selection rules and budget accounting therefore come from the shared core.
Strategy, plan, rationale and reflection summaries are displayed beside each
proposal; private provider reasoning is not required or exposed.

For a fair comparison, use the same model revision, dataset and splits, goal,
metric, seed, editor/model/effort, token limits and budget. Identical settings
provide the same editing opportunities, not a guarantee of identical samples:
frontier responses and GPU arithmetic can vary between runs and hardware.

This public deployment has explicit resource limits: supported target models,
$0.01–$100 frontier spend, 128 examples per dataset, four preservation checks
with at most 16 examples each, 128 generated tokens per answer, and 1,024 sequence
tokens. Seed 42, an editor output limit of 2,048 tokens and preservation
tolerance 0 match the terminal defaults and appear in the review. These are hosting limits, not a different strategy or hidden turn cap.
ZeroGPU availability and account quota can delay or stop allocations. A browser
heartbeat is required; five minutes without one cancels the run. Completed
bundles are retained for two hours. API editors are supported here; locally
signed-in Codex/Claude CLI editors require the terminal environment.

## Source

[GitHub](https://github.com/chrisvoncsefalvay/vespasian)
· [Command line](https://chrisvoncsefalvay.github.io/vespasian/)
· [Space branch](https://github.com/chrisvoncsefalvay/vespasian/tree/zerogpu-space/spaces/zerogpu)

The `zerogpu-space` branch receives changes from `main` through the deployment
workflow. Every deployment bundles the Rust controller and Python model engine
from the same commit. `PROVENANCE.json` records the source revision and file hashes.
The Space application lives outside the published Cargo crate.
