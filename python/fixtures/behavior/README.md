# Published diagnostic excerpts

`p50c_fwd1_excerpts.json` contains exact published 250-character start, middle and end excerpts from two original-model `fwd_1` practice generations in RikiyaT’s [lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/tool_calls.jsonl#L4). The repository identifies its logs/data as CC BY 4.0. Attribution and generation-request links are retained in the fixture.

These GSM8K training-set practice episodes were used by the upstream experiment during diagnosis. They are not unseen or held-out evaluation examples. This directory is exclusively for offline diagnostic unit tests; do not load it into starter data, model editing, search, validation, test or preservation datasets.

The source logger replaced newlines with spaces before printing each slice. The exact published spaces, including trailing spaces, are preserved. These are not complete traces. Tests examine each excerpt separately rather than joining disjoint text. Generation metadata describes the complete upstream response. Upstream correctness is provenance, not an input to Vespasian’s correctness scoring. A `cut=false` observation is not treated as published proof of EOS, complete reasoning, or a closing thinking marker.
