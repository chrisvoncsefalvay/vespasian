# Small, reproducible starters

Four original synthetic exercises teach editing, not benchmark performance. No downloaded datasets or guaranteed improvements. The manifest contains settings, provenance and hashes; regenerate offline with `python3 scripts/generate_starters.py`. Test with `python3 -m unittest discover -s python -p test_starters.py -v`.

| File | Edit | Validation | Test | Preserve | Token cap |
|---|---:|---:|---:|---:|---:|
| starter-maths.jsonl | 6 | 4 | 4 | 6 | 128 |
| starter-parity.jsonl | 6 | 4 | 4 | 8 | 64 |
| starter-geometry.jsonl | 6 | 4 | 4 | 8 | 64 |
| starter-json.jsonl | 6 | 4 | 4 | 6 | 64 |

Use normalized exact-match and zero preservation tolerance initially. Paired generation currently uses greedy decoding with thinking off. Record the resolved model revision per run; none is imposed here. Runtime depends on hardware, loading and candidates. These are deliberately small experiments, not a wall-clock promise.

Rows contain prompt, target, split and an offline case record used by tests. The worker only consumes prompt, target and split, not case annotations. Keep test rows sealed during search. Never tune against held-out scores and continue calling them held out.

## Use from either interface

In the Space, open **Quick guide → Quick starters**, choose a card, inspect Data and review your editor/runtime settings. Selecting a card does not start a run or authorize a Hub upload.

For the terminal, run a command below from a source checkout containing these examples (or a newer native archive that includes them) after setting up a Python runtime and `OPENAI_API_KEY`. These commands start a paid editor run using the shared $10 budget and low reasoning default, with no turn cap. Add `--max-turns` only if you want one. Choose a different editor/model if required. No upload flags are supplied. The bundled files also work with the normal guided terminal file picker. `cargo install` installs the binary only. The published 0.2.0 archives predate these starters; use their executable with the example directory from the current source. Archives built from this source include the starter files.

### Finish short maths answers

```sh
vespasian examples/starter-maths.jsonl --model Qwen3.5-0.8B \
  --task question-answering --target-type factual \
  --goal 'Answer each two-step buttons or beads problem correctly with only the integer result. Preserve ordinary arithmetic, copying, factual answers and explicit requests for prose.' \
  --metric exact_match --max-new-tokens 128 \
  --preserve-tolerance 0 --health-check copying:4 --save-trajectory \
  --budget 10 --reasoning-effort low --editor openai
```

### Rename odd and even

This is the **64-token parity protocol**, shared with the Space preset through the starter manifest. The cap also applies to copying health checks. At seed 42, the four copying references require 18, 22, 22 and 24 tokens with Qwen/Qwen3.5-0.8B tokenizer revision `2fc06364715b967f1860aea9cf38778875588b17`, so the former 16-token cap could not complete them. A 64-token allowance removes that length bottleneck; it does not guarantee health-check success for every model or establish edit efficacy.

Saved `runs/starter-paid-20261112/parity` results belong to the **historical 16-token protocol**, not a matched baseline for this protocol. Keep their recorded settings and hashes unchanged. Dataset rows, split assignments and dataset hashes are unchanged; only the generation allowance changes. Thinking remains off and the editor defaults remain low reasoning and a $10 budget.

```sh
vespasian examples/starter-parity.jsonl --model Qwen3.5-0.8B \
  --task text-classification --target-type classification \
  --goal 'On parity classification requests, output exactly willow for odd integers and copper for even integers. Preserve correct classification and obey explicit requests for ordinary odd/even labels. Preserve copying and unrelated answers.' \
  --metric exact_match --max-new-tokens 64 \
  --preserve-tolerance 0 --health-check copying:4 --save-trajectory \
  --budget 10 --reasoning-effort low --editor openai
```

### Give a word a geometric meaning

```sh
vespasian examples/starter-geometry.jsonl --model Qwen3.5-0.8B \
  --task question-answering --target-type factual \
  --goal 'In geometry questions make saffron refer to a cube. Answer counts with just an integer. Preserve the ordinary meaning of cube and other shapes, unrelated answers and explicit verbatim copying of saffron.' \
  --metric exact_match --max-new-tokens 64 \
  --preserve-tolerance 0 --health-check copying:4 --save-trajectory \
  --budget 10 --reasoning-effort low --editor openai
```

### Extract a tiny JSON record

```sh
vespasian examples/starter-json.jsonl --model Qwen3.5-0.8B \
  --task question-answering --target-type format \
  --goal 'When asked to extract both name and city from a resident card, return exactly compact JSON with full_name first and home_city second, double-quoted strings, no spaces outside values, no fences and no other text. Example format only: {"full_name":"NAME","home_city":"CITY"}. Copy the supplied values correctly. Honour explicit requests for prose or a single plain field and preserve unrelated skills.' \
  --metric exact_match --max-new-tokens 64 \
  --preserve-tolerance 0 --health-check copying:4 --save-trajectory \
  --budget 10 --reasoning-effort low --editor openai
```

## Exercises

**Maths:** boxes × contents − removed items. Triples and multiplication pairs are disjoint across splits. Test changes the story template while retaining the same operation family. Preserve simple arithmetic, copying, a factual answer and prescribed prose. Integer-only scoring measures correctness plus stopping/format adherence: a correct number embedded in an explanation fails. Separate content failures from format failures when interpreting results. This is narrow within-family transfer, not general maths reasoning.

**Parity:** map odd to willow and even to copper without putting the mapping in target prompts. Each target split is balanced with distinct integers. Preserve ordinary-label overrides, code-word comprehension, copying and unrelated skills. Shared labels are the intended learned convention, not leakage. Four test integers do not establish broad classification transfer.

**Geometry:** make saffron refer to cube. Edit/validation cover faces and vertices using different wording; test exclusively asks about edges. Four test phrasings are one semantic fact, not four independent facts. Preserve cube, triangle, square, alias copying and unrelated skills. This is a verified multi-token teaching adaptation, not an eligible single-token IWIBench alias episode. On Qwen/Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`, `saffron` encodes as `[82, 2562, 2176]`, and leading-space ` saffron` as `[796, 529, 2176]`; `cube` is `[43896]` and leading-space ` cube` is `[22997]`. Inspect actual tokenization and current model rows before selecting row transfer: a three-token alias is not interchangeable with a single embedding row, and IDs must not be copied to another tokenizer.

**JSON:** default to compact JSON, full_name before home_city, double quotes, no separator spaces or surrounding prose. Names, cities and records are disjoint; test reverses input field order. Preserve prose/single-field overrides and unrelated skills. Canonical key order and compact separators are part of the objective. Valid JSON in another order or with separator spaces fails: this is canonical-string generation, not semantic JSON equality.

## Scoring limits

Vespasian exact_match casefolds strings and collapses whitespace before whole-string comparison (python/worker.py::normalise). It is normalized string exact-match, not byte-exact equality. JSON key order and punctuation spacing matter, but case changes inside values may be forgiven. Copying controls likewise do not establish byte-perfect copying. Token F1 and loss are diagnostics, not substitutes for behaviour.

Lower reference loss without improved exact-match is not behavioural success. Tiny preservation sets miss regressions. Report raw counts, baseline/edited answers, termination and fresh-checkpoint verification. Perfect baselines have no headroom. Failing ordinary prerequisites may reflect missing capability rather than a convention worth editing. Assess admission with non-test evidence, never by changing test cases after seeing outcomes.

## Provenance and licensing

These original datasets, wording, generator and tests follow the root Vespasian Apache-2.0 licence. They do not copy benchmark prompts or code.

- Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), Experiment 8. It fixes Qwen3.5-0.8B thinking loops using GSM8K/MATH practice, a different edit interface and much larger budgets. Our synthetic thinking-off maths exercise does not reproduce that intervention or its reported scores.
- Chris von Csefalvay and Tamás Foldi, [IWIBench](https://github.com/HCLTech-robotics/iwibench), inspected commit 085db19bd4cc2c3792989cf00690b30b5f34f47c, family version 0.2.2. Task-design references: src/iwibench/families.py (alias), families_conventions.py (codes and structured response), grading_conventions.py (original semantic scoring). These reduced original adaptations preserve broad target/preservation ideas while changing examples, target model, counts, split construction, budgets and grading. They are not benchmark episodes or rewards.

No benchmark-code licence was verified in that IWIBench checkout; its website-template licence does not establish permission for benchmark material. The generator makes no dataset download, paid editor call or model efficacy experiment.

Non-test Qwen baseline inspection found frequent length truncation at the original maths cap of 32 and geometry cap of 16. Defaults are now 128 and 64 respectively to reduce this artificial bottleneck. This change does not establish edit efficacy or eliminate formatting failures; inspect termination reasons again at the new caps. Dataset rows and held-out cases are unchanged. Generation writes explicit UTF-8 bytes with LF line endings on every platform.
