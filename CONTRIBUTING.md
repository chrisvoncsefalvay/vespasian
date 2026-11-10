# Contributing to Vespasian

Vespasian combines a Rust terminal application with a persistent Python model worker. Changes to data handling, evaluation and tensor editing should include a focused test of the resulting behaviour.

## Development setup

Install Rust, Python 3.12 and uv. From the checkout:

```sh
cargo build
cargo run -- setup
cargo run -- doctor
```

Use an environment with [python/requirements.txt](python/requirements.txt) for the Python tests. `vespasian setup --python PATH` registers an existing environment.

## Project layout

| Path | Responsibility |
| --- | --- |
| `src/wizard.rs` | Guided setup and live dataset preview |
| `src/ui.rs` | Hill-climb graph, answers, editor activity and genealogy |
| `src/editor.rs` | Frontier provider protocols and usage accounting |
| `src/search.rs` | Candidate selection and preservation rules |
| `src/data.rs`, `python/dataset.py` | Local data and Hugging Face imports |
| `python/worker.py` | Model loading, tensor operations, generation and export |
| `python/health.py` | Health-check examples and scoring |
| `site/` | GitHub Pages source and visual assets |
| `docs/media/` | Terminal recordings, GIFs and reproduction instructions |

## Run the checks

```sh
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
python -m unittest discover -s python -p 'test_*.py'
cargo package --locked
```

Worker tests construct small local transformers. Dataset, publication and trajectory tests exercise their protocols with local fixtures.

For a tensor operation, check its stored norm, rollback, generation and fresh checkpoint reload. For dataset changes, check that preview and imported rows agree. Keep example identity and split assignment stable.

## Work on the website

```sh
python scripts/build_site.py
python -m http.server 8000 --bind 127.0.0.1 --directory dist/site
```

Open `http://127.0.0.1:8000`. Check a wide desktop view and a narrow mobile view. Exercise the recording player, install tabs and copy buttons with the keyboard.

The site uses local fonts and player assets. Their licences live beside the files in `site/`. See [the media guide](docs/media/README.md) to regenerate terminal recordings.

## Submit a change

Use a focused branch and a pull request describing the resulting behaviour. Include the commands you ran and screenshots for visual changes. Keep public copy concrete, use British English and preserve the copper/charcoal/ivory palette.

For a new model family, include the configuration and tensor selectors, generation input handling, export/reload behaviour and a small reproducible example.

Release packaging and publication are described in [the release guide](docs/RELEASING.md).
