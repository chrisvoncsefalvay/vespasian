# Terminal recordings

| Recording | Contents | Files |
| --- | --- | --- |
| Guided setup | Public GSM8K preview, field mapping, objective, health checks, editor model and budget | [Asciinema](setup.cast), [GIF](setup.gif), [poster](setup-poster.png) |
| Will climb | A four-turn greetings edit on Qwen3.5-0.8B, with generated answers, reference loss, arithmetic preservation, edit ancestry and model layers | [Asciinema](will-climb.cast), [GIF](will-climb.gif), [poster](will-climb-poster.png) |

Both recordings use a 112×34 terminal. The hill climb replays a saved Codex editing run with three retained edits. The retained checkpoint answers both validation greetings with `Salve!` and preserves the arithmetic answer. The presentation timeline gives each recorded turn 4.4 seconds.

## Render the GIFs

Install [agg](https://docs.asciinema.org/manual/agg/installation/) and run:

```sh
agg --font-family 'DejaVu Sans Mono' --font-size 15 --line-height 1.25 \
  --fps-cap 8 --last-frame-duration 1 --idle-time-limit 10 \
  docs/media/setup.cast docs/media/setup.gif

agg --font-family 'DejaVu Sans Mono' --font-size 15 --line-height 1.25 \
  --fps-cap 8 --last-frame-duration 1 --idle-time-limit 10 \
  docs/media/will-climb.cast docs/media/will-climb.gif
```

## Capture the application

[record_demos.py](../../scripts/record_demos.py) drives the real terminal interface through a pseudoterminal and writes asciicast v2 output. It uses `pexpect`, `pyte` and Pillow. The setup capture reads public Hugging Face data and stops at the review screen. The climb capture reads the saved run and projects its measured answers and alterations into the current replay format.

```sh
python3 -m pip install pexpect pyte pillow
cargo build --release --locked
python3 scripts/record_demos.py \
  --binary target/release/vespasian \
  --python /path/to/model-runtime/bin/python \
  --source-run runs/qwen-greetings-001
```

Capture metadata and source hashes are written under `runs/media-015/`. The GIF renderer uses the colour theme embedded in each cast.
