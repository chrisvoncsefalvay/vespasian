# ZeroGPU deployment

The `zerogpu-space` branch carries the Gradio application. The workflow merges
`main` into that branch, runs the repository's normal checks against the resulting
commit, builds its Linux x86-64 controller, and uploads a generated bundle to
`chrisvoncsefalvay/vespasian` on Hugging Face.

## GitHub and Space setup

1. Keep `.github/workflows/zerogpu-space.yml` on **both `main` and
   `zerogpu-space`**. Only the workflow needs to be added to `main`. The app remains
   on `zerogpu-space`.
2. Create the GitHub environment `hf-space`. Add its secret `HF_TOKEN`, using a
   Hugging Face token with write access to the destination Space. Set the
   environment or repository variable `HF_SPACE_ID` to
   `chrisvoncsefalvay/vespasian` (also the default).
3. Allow the workflow's `GITHUB_TOKEN` to update `zerogpu-space`. The sync job
   requests `contents: write`. Branch protection must permit the bot's normal
   merge commit. The workflow never force-pushes.
4. The destination must already be a Gradio Space using ZeroGPU (`zero-a10g`).
   The uploader checks that setting and never creates a Space or changes its
   hardware. Space OAuth settings come from the app's README frontmatter.

Pushes to either branch and manual workflow runs on either branch trigger the
same sync, check, build and deploy sequence. All runs share one concurrency group.
An unresolved merge conflict stops before build or upload. Resolve the merge on
`zerogpu-space`, push it, and rerun. A failed push leaves the remote branch intact.

Deployment continues in the same run because a push made with `GITHUB_TOKEN`
does not start another push workflow. See
[GitHub's workflow trigger documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).

## Bundle layout

| Source | Space path |
| --- | --- |
| `spaces/zerogpu/{app,ui,runtime,gpu,bridge,python_bridge,hub}.py` | Bundle root |
| `spaces/zerogpu/{README.md,requirements.txt,packages.txt}` | Bundle root (`packages.txt` is optional) |
| `spaces/zerogpu/assets/` | `assets/` |
| `python/{worker,dataset,health,publish,trajectory}.py` | `core/` |
| `examples/` | `examples/` |
| `LICENSE` | `LICENSE` |
| Built Linux x86-64 release controller | `bin/vespasian` |

Tests, deployment helpers, this document, caches and hidden files are excluded.
The existing `Cargo.toml` include list excludes `spaces/` and `.github/` from the
crate. `test_package_space.py` checks the actual `cargo package --list` output.

`PROVENANCE.json` records the source commit and tree, Cargo version, source hashes,
and the size, SHA-256 and executable flag of every bundled file except itself.
Staging retains executable bits. Application startup restores them for
`bin/vespasian` and `python_bridge.py` because Hub HTTP uploads may not retain Unix
file modes.

## Local staging and checks

From a committed checkout with the Linux x86-64 Rust target installed:

```bash
cargo build --release --locked --target x86_64-unknown-linux-gnu
python3 -m unittest discover -s spaces/zerogpu -p 'test_package_space.py' -v
python3 spaces/zerogpu/package_space.py build --output dist/zerogpu-space
python3 spaces/zerogpu/package_space.py verify dist/zerogpu-space
```

Use a fresh output directory for each build. Staging does not overwrite existing
directories. `--allow-dirty` permits local inspection while files are being
edited, records that fact, and makes the resulting bundle ineligible for upload.
An ARM64 binary is rejected, even when staging on an ARM64 development machine.

For an authorised manual deployment, set `HF_TOKEN` in the environment and run:

```bash
python3 spaces/zerogpu/package_space.py upload dist/zerogpu-space \
  --repo-id chrisvoncsefalvay/vespasian
```

Upload checks every bundled hash first. It replaces generated files and removes
only obsolete paths named in the previous `PROVENANCE.json`. Unrelated remote
files are retained. The upload is one commit based on the observed Space revision,
so an intervening Space update causes a conflict instead of being overwritten.
This uses the Hub API's
[parent commit precondition](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#create_commit).

The workflow retains the generated bundle as a GitHub Actions artifact for 14
days. A successful upload establishes the Space commit. Confirm its build logs
and exercise the app on ZeroGPU before claiming a successful runtime test.
