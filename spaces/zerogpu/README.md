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

The workbench shows the objective curve, health checks, generated answers, editor
summaries, edited layers and the ancestry of every retained checkpoint.

## Your Hub, connected

1. **Sign in with Hugging Face.** Your datasets and organisation repositories
   appear in the dataset picker. Paste any accessible repository ID to open it.
2. **Inspect your examples.** Choose a configuration, source split, field mapping
   and example limit. Paired examples, text corpora and prompt-only datasets are
   supported. A separate repository or split can supply held-out test examples.
3. **Set the objective.** Choose the metric, behaviour to preserve and optional
   health checks. A regression limit can reject candidates that weaken a check.
4. **Choose an editor.** Enter your OpenAI or Anthropic key and a USD inference
   budget. The controller reserves the cost of each call before issuing it.
5. **Follow the climb.** Each candidate is evaluated against the retained model.
   Download the checkpoint and run bundle, or save the model and trajectory to
   public or private repositories in your account.

HF access comes from your connected account. The optional token override supports
other Hub credentials. Editor keys stay within the session's controller process.

## Model execution

The Space runs Qwen3.5-0.8B on ZeroGPU. Dataset access and frontier editing use CPU
time. GPU allocations cover model loading, generation and tensor operations.
Exact tensor snapshots carry the retained checkpoint between allocations.

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
