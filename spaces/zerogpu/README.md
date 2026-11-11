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

1. **Define the behaviour.** Start with your own objective or load a complete
   example. Choose paired examples, reference writing or prompts with a scoring
   rubric. Preservation checks track scores by default. Select a regression
   limit to reject edits that weaken a check beyond that limit.
2. **Inspect your data.** Search public datasets, upload a file or sign in to
   browse your datasets and organisation repositories. Choose fields from the
   preview and set the example limit. Editing and held-out test data have
   separate previews.
3. **Choose the editor and budget.** Select an OpenAI or Anthropic model, enter
   its API key and set a USD inference budget. Custom models accept explicit
   token prices. The controller reserves the next call's cost before issuing it.
4. **Review and start.** Check the objective, metric, example counts,
   preservation policy and budget together before starting the experiment.
5. **Inspect and save the result.** Select turns in the graph or ancestry view
   to see their tensors, measurements and acceptance decisions. Compare baseline
   and edited answers, then review final test results and checkpoint verification.
   Download the run or save its model and trajectory to public or private Hub
   repositories.

The first-use guide introduces this flow in four short steps. Dismiss it with
Skip, the close button or Escape. Your browser remembers the dismissal, and
**Quick guide** reopens it at any time.

HF access comes from your connected account. The optional token override supports
other Hub credentials. Editor keys stay within the session's controller process.

## Model execution

The Space runs Qwen3.5-0.8B on ZeroGPU. Dataset access and frontier editing use CPU
time. GPU allocations cover model loading, generation and tensor operations.
Exact tensor snapshots carry the retained checkpoint between allocations.
Each allocation starts from a fresh queued browser request, so Hugging Face
renews GPU authorisation throughout a longer experiment. Keep the workbench open
while the experiment runs.

Saved checkpoints are loaded from disk in a fresh worker and checked against the
retained rollouts. The run bundle contains measurements, checkpoint hashes,
dataset revisions, budget receipts and the optional editor trajectory.
Downloads remain available for two hours after a run finishes. Hub exports stay
in the destination repository.

## Source

[GitHub](https://github.com/chrisvoncsefalvay/vespasian)
· [Command line](https://chrisvoncsefalvay.github.io/vespasian/)
· [Space branch](https://github.com/chrisvoncsefalvay/vespasian/tree/zerogpu-space/spaces/zerogpu)

The `zerogpu-space` branch receives changes from `main` through the deployment
workflow. Every deployment bundles the Rust controller and Python model engine
from the same commit. `PROVENANCE.json` records the source revision and file hashes.
The Space application lives outside the published Cargo crate.
