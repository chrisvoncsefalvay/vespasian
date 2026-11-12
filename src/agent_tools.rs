//! Small discovery surface and bounded, read-only tools shared by CLI and Space.
use anyhow::{bail, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const PAGE: usize = 8;
const TEXT_PAGE: usize = 2048;
const MAX_OFFSET: usize = 1_000_000;
const BEHAVIOR_SIGNALS: [&str; 10] = [
    "repetition",
    "length_truncation",
    "terminated",
    "unknown_termination",
    "empty_output",
    "reference_mismatch",
    "format_only_mismatch",
    "content_difference",
    "unclassified_reference_mismatch",
    "preservation_mismatch",
];
const BEHAVIOR_LIMITATIONS: &str = "Diagnostics are observations, not acceptance scores. Repetition is a conservative heuristic, not proof of an unwanted loop. Length means a generation limit, not an incorrect answer. Termination does not prove completeness. Content/format comparisons cover only whole numeric scalars or valid JSON; other mismatches remain unclassified. No semantic correctness is inferred. Preservation mismatches are reference failures, not necessarily regressions from baseline.";
const ROLLOUT_FIELDS: [&str; 14] = [
    "row_id",
    "split",
    "prompt",
    "target",
    "prediction",
    "exact_match",
    "token_f1",
    "numeric_match",
    "semantic",
    "loss",
    "generated_tokens",
    "finish_reason",
    "behavior",
    "mode",
];
fn short(value: &Value, limit: usize) -> Value {
    match value.as_str() {
        Some(s) => json!(s.chars().take(limit).collect::<String>()),
        None => match value {
            Value::Array(items) => json!(items
                .iter()
                .take(8)
                .map(|v| short(v, limit))
                .collect::<Vec<_>>()),
            Value::Object(map) => Value::Object(
                map.iter()
                    .take(16)
                    .map(|(k, v)| (k.clone(), short(v, limit)))
                    .collect(),
            ),
            _ => value.clone(),
        },
    }
}
fn pick(value: &Value, keys: &[&str]) -> Value {
    let mut out = serde_json::Map::new();
    for key in keys {
        if let Some(v) = value.get(*key) {
            out.insert((*key).into(), short(v, 768));
        }
    }
    Value::Object(out)
}
pub fn measurements(value: &Value) -> Value {
    let mut out = serde_json::Map::new();
    for split in ["edit", "validation", "preserve"] {
        if let Some(v) = value.get(split) {
            out.insert(
                split.into(),
                pick(
                    v,
                    &[
                        "count",
                        "loss",
                        "exact_match",
                        "token_f1",
                        "numeric_match",
                        "semantic",
                    ],
                ),
            );
        }
    }
    Value::Object(out)
}
fn behavior_counts(summary: &Value) -> Value {
    let mut signals = serde_json::Map::new();
    for name in BEHAVIOR_SIGNALS {
        if let Some(count) = summary["signals"][name]["count"].as_u64() {
            signals.insert(name.into(), json!(count));
        }
    }
    json!({"sample_count":summary["sample_count"].as_u64(),"diagnosed_count":summary["diagnosed_count"].as_u64(),"signals":signals})
}
pub fn history_summary(rows: &[Value], limit: usize) -> Vec<Value> {
    rows.iter().rev().take(limit).map(|r| json!({"turn":r["turn"], "accepted":r["accepted"], "strategy":short(&r["proposal"]["strategy"],80), "operation":r["proposal"]["operation"], "strength":r["proposal"]["strength"], "metrics":measurements(&r["metrics"]), "decision":short(&r["decision_reason"],256),"health_passed":r["health_passed"],"behavior":behavior_counts(&r["behavior"]),"controls":pick(&r["controls"],&["status","retention","checked","applicable"])})).collect()
}
pub fn health_comparison(current: &Value, baseline: &Value) -> Value {
    let mut rows:Vec<_>=current.as_object().into_iter().flatten().filter_map(|(name,row)| {
        let score=row["score"].as_f64()?;
        let before=baseline[name]["score"].as_f64();
        Some(json!({"check":name.chars().take(120).collect::<String>(),"baseline":before,"current":score,"change":before.map(|b|score-b),"count":row["count"]}))
    }).collect();
    rows.sort_by(|a, b| {
        a["change"]
            .as_f64()
            .unwrap_or(0.)
            .total_cmp(&b["change"].as_f64().unwrap_or(0.))
    });
    json!({"current_checks":current.as_object().map_or(0,|o|o.len()),"baseline_checks":baseline.as_object().map_or(0,|o|o.len()),"largest_drops":rows.into_iter().take(4).collect::<Vec<_>>()})
}
fn raw_rollouts(best: &Value) -> Vec<Value> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    best["samples"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|r| {
            matches!(
                r["split"].as_str(),
                Some("edit" | "validation" | "preserve")
            )
        })
        .map(|r| {
            let split = r["split"].as_str().unwrap();
            let index = counts.entry(split.to_owned()).or_default();
            let mut row = pick(r, &ROLLOUT_FIELDS);
            let supplied = r["row_id"].as_str().filter(|id| {
                id.strip_prefix(&format!("{split}:")).is_some_and(|n| {
                    !n.is_empty() && n.len() <= 10 && n.bytes().all(|b| b.is_ascii_digit())
                })
            });
            row["row_id"] = json!(supplied
                .map(str::to_owned)
                .unwrap_or_else(|| format!("{split}:{index}")));
            *index += 1;
            for field in ["prompt", "target", "prediction"] {
                if let Some(text) = r[field].as_str() {
                    row[field] = json!(text);
                }
            }
            row
        })
        .collect()
}
fn rollout_preview(row: &Value) -> Value {
    let mut preview = pick(row, &ROLLOUT_FIELDS);
    for field in ["prompt", "target", "prediction"] {
        if let Some(text) = row[field].as_str() {
            let count = text.chars().count();
            preview[format!("{field}_chars")] = json!(count);
            preview[format!("{field}_truncated")] = json!(count > 768);
        }
    }
    preview
}
/// Rebuild the compact summary from only non-test rows, never forward a manifest
/// or blindly forward a caller-provided aggregate which may include test data.
pub fn behavior_summary(evaluation: &Value) -> Value {
    let rows = raw_rollouts(evaluation);
    let mut signals = serde_json::Map::new();
    let mut by_split = serde_json::Map::new();
    for name in BEHAVIOR_SIGNALS {
        signals.insert(
            name.into(),
            json!({"count":0,"row_ids":[],"row_ids_truncated":false}),
        );
    }
    for split in ["edit", "validation", "preserve"] {
        let counts: serde_json::Map<String, Value> = BEHAVIOR_SIGNALS
            .iter()
            .map(|name| ((*name).into(), json!(0)))
            .collect();
        by_split.insert(split.into(), json!({"count":0,"signals":counts}));
    }
    let mut diagnosed = 0;
    let mut reference_count = 0;
    for row in &rows {
        let split = row["split"].as_str().unwrap();
        by_split[split]["count"] = json!(by_split[split]["count"].as_u64().unwrap() + 1);
        if row["mode"].as_str().unwrap_or("paired") == "paired"
            && row["exact_match"].is_boolean()
            && row["target"].is_string()
        {
            reference_count += 1;
        }
        let Some(observed) = row["behavior"]["signals"].as_array() else {
            continue;
        };
        if row["behavior"]["evidence"]["status"] == "no_prediction" {
            continue;
        }
        diagnosed += 1;
        for name in BEHAVIOR_SIGNALS {
            if !observed.iter().any(|value| value.as_str() == Some(name)) {
                continue;
            }
            let signal = signals.get_mut(name).unwrap();
            let count = signal["count"].as_u64().unwrap() + 1;
            signal["count"] = json!(count);
            if count <= PAGE as u64 {
                signal["row_ids"]
                    .as_array_mut()
                    .unwrap()
                    .push(row["row_id"].clone());
            }
            signal["row_ids_truncated"] = json!(count > PAGE as u64);
            by_split[split]["signals"][name] =
                json!(by_split[split]["signals"][name].as_u64().unwrap() + 1);
        }
    }
    json!({"schema_version":1,"scope":"edit_validation_preserve","sample_count":rows.len(),"diagnosed_count":diagnosed,"reference_count":reference_count,"signals":signals,"by_split":by_split,"limitations":BEHAVIOR_LIMITATIONS})
}
pub fn discovery(catalogue: &Value) -> Value {
    let cards: Vec<_> = catalogue["cards"].as_array().into_iter().flatten().map(|c| json!({"id":c["id"],"operations":c["operations"],"use_when":short(&c["when_to_use"],160),"available":true})).collect();
    json!({"skills":cards,"available_operations":catalogue["available_operations"],"unavailable":[{"id":"rome","reason":"Not implemented: requires causal localisation, covariance statistics and optimised latent targets."},{"id":"memit","reason":"Not implemented: requires multi-layer covariance solves and optimised latent targets."},{"id":"patch_arithmetic","reason":"Agent-directed patch composition/negation is not implemented. Saved journal patches can be replayed by the materialize-patch utility; repeating an edit is not patch arithmetic."}],"tools":[{"name":"read_skill","purpose":"Load optional focused guidance for one suggested strategy."},{"name":"inspect_targets","purpose":"Read eight supported tensors; query matches name, alias, role or operation."},{"name":"inspect_probe_modules","purpose":"Read eight supported ordinary Linear modules and dimensions. Query filters module names. Probe activations on explicit non-test row IDs, source (prompt/reference/generation) and bounded token positions."},{"name":"probe_activations","purpose":"Measure selected module activations for up to 8 non-test contexts and optional contrasts. Read-only, produces bounded statistics and reusable direction handles, not full vectors. Set the structured probe field; null for other actions."},{"name":"inspect_tokens","purpose":"Inspect tokenizer-grounded IDs and norms; optional query tokenizes one text up to 256 characters; offset pages eight candidates."},{"name":"read_history","purpose":"Read eight measured candidate outcomes and preservation/control decisions."},{"name":"read_rollouts","purpose":"Empty query: eight non-test previews with row IDs, generation counts and finish reasons. Query a row ID, optionally /prompt, /target or /prediction (default), to page full text in 2048-character chunks; offset is then a character offset."},{"name":"read_behavior","purpose":"Read eight non-test row diagnostics and evidence. Query a signal name, split or row ID to filter. Counts describe observations, not semantic judgments or acceptance scores."},{"name":"propose","purpose":"Submit one bounded edit for controller validation, application, testing and keep/revert."},{"name":"stop","purpose":"End search without another edit, explaining why."}],"operation_contracts":{"scale":"MLP: W*(1+strength).","low_rank":"MLP: seeded low-rank direction, rank1–16, relative norm abs(strength).","noise":"MLP: seeded full-rank direction, relative norm abs(strength).","activation_rank_one":"MLP: forward reference keys with target-minus-predicted embedding directions; rank selects examples.","activation_preserve":"MLP: contrastive reference direction projected off measured preserve-key span; requires edit/preserve references and positive strength.","direction_projection":"One inspected eligible MLP weight and matching measured direction_id: strength0.000001–16, rank1, and1–256 inspected column indices. Actual stored relative norm after dtype rounding must satisfy the run max_relative_norm. Handles are evidence, not proof the direction causes the failure.","token_row_scale":"One table: multiply inspected non-special destination rows by1+strength; empty sources.","token_row_interpolate":"One table: positive bounded blend toward paired inspected source IDs; distinct source/destination; stored-row norm bound may reduce fraction.","token_row_residual":"Supported output table: positive measured target-minus-probability head-input residual on inspected edit_reference rows; rank selects examples; empty sources."},"edit_contract":"Only available_operations may be used, and inspected targets must support them. Other operation contracts describe capabilities that this model/dataset may not have. Except direction_projection, strength magnitude0.000001–0.1 and rank1–16. Direction_projection requires strength0.000001–16, rank1, one inspected target, a matching measured direction handle and1–256 selected in-range columns; actual stored relative norm after dtype rounding must satisfy the run max_relative_norm. MLP operations use empty token selectors. Row operations one canonical table and1–32 inspected non-special IDs. Tied aliases are one physical edit affecting both roles. Proposed writes are tested before retention. Use provided tools before considering unsupported custom gradient optimisers; arbitrary code/autograd/training is not available.","instructions":"Skills are optional suggestions, not a strategy whitelist. Inspect needed targets/tokens, then propose an available operation with your own concise strategy, plan and measured rationale. Read a skill only when its guidance helps. Tools are read-only and paginated (8 entries). Use query to filter target names/aliases/roles/operations, or inspect one token text (up to 256 characters) with the current tokenizer. An empty token query pages the current view; a new text replaces it. Use offset to page. Start with the measured behavior summary, then read_behavior and full-text rollout pages to inspect failures. Choose probes and model-level changes from that evidence, not a fixed task menu. Explain the public evidence and hypothesis concisely, without private chain-of-thought. Read history for previous outcomes. Do not request the entire manifest. Tool calls consume frontier budget; stop if no justified candidate remains. The selected skill and latest target/token pages remain in context, alongside bounded recent evidence and request receipts. Evidence previews may be truncated; fetch a specific page when needed. Combine behavior and rollout evidence rather than alternating identical reads. Identical same-state requests reuse results; two consecutive reused requests stop inspection as no progress. New justified queries and pages remain available."})
}

pub fn validate_action(action: &Value) -> Result<()> {
    let fields = [
        "action", "skill_id", "query", "offset", "summary", "proposal", "probe",
    ];
    let obj = action
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Expected an agent action object"))?;
    if obj.len() != fields.len() || fields.iter().any(|key| !obj.contains_key(*key)) {
        bail!("Unexpected agent action shape");
    }
    for (field, max) in [
        ("action", 32),
        ("skill_id", 80),
        ("query", 256),
        ("summary", 512),
    ] {
        if action[field]
            .as_str()
            .is_none_or(|s| s.chars().count() > max)
        {
            bail!("Invalid or oversized agent action field: {field}");
        }
    }
    if ![
        "read_skill",
        "inspect_targets",
        "inspect_tokens",
        "read_history",
        "read_rollouts",
        "read_behavior",
        "inspect_probe_modules",
        "probe_activations",
        "propose",
        "stop",
    ]
    .contains(&action["action"].as_str().unwrap_or(""))
        || action["offset"]
            .as_u64()
            .is_none_or(|n| n > MAX_OFFSET as u64)
    {
        bail!("Invalid agent action or offset");
    }
    if action["action"] == "probe_activations" {
        let probe = action["probe"].as_object().ok_or_else(|| {
            anyhow::anyhow!("Activation probes require a structured probe object")
        })?;
        if probe.len() != 4
            || ["targets", "contexts", "contrast_contexts", "top_k"]
                .iter()
                .any(|key| !probe.contains_key(*key))
        {
            bail!("Unexpected activation probe shape");
        }
        if action["probe"]["targets"]
            .as_array()
            .is_none_or(|a| a.is_empty() || a.len() > 4)
            || action["probe"]["contexts"]
                .as_array()
                .is_none_or(|a| a.is_empty() || a.len() > 8)
            || action["probe"]["contrast_contexts"]
                .as_array()
                .is_none_or(|a| a.len() > 8)
            || action["probe"]["top_k"]
                .as_u64()
                .is_none_or(|n| !(1..=128).contains(&n))
        {
            bail!("Activation probe exceeds bounded target, context or top-k limits");
        }
    } else if !action["probe"].is_null() {
        bail!("Only activation probes may have non-null probe arguments");
    }
    if action["action"] == "propose" {
        if !action["proposal"].is_object() {
            bail!("Proposal action requires an edit object");
        }
    } else if !action["proposal"].is_null() {
        bail!("Read-only and stop actions must have a null proposal");
    }
    Ok(())
}

// Payload retention is bounded independently of the number of editor calls.
const RECEIPTS: usize = 12;
const CACHE_ENTRIES: usize = 8;
const CACHE_BYTES: usize = 256 * 1024;
const EVIDENCE_BYTES: usize = 6000;

fn evidence_preview(result: &Value) -> Value {
    if result.to_string().len() <= EVIDENCE_BYTES {
        return result.clone();
    }
    let mut preview = short(result, 256);
    while preview.to_string().len() > EVIDENCE_BYTES {
        if let Some(items) = preview.get_mut("items").and_then(Value::as_array_mut) {
            if items.pop().is_some() {
                continue;
            }
        }
        return json!({"excerpt": result.to_string().chars().take(1024).collect::<String>(), "truncated": true});
    }
    if &preview != result {
        preview["truncated"] = json!(true);
    }
    preview
}

pub struct Session {
    pub catalogue: Value,
    pub inspection: Value,
    editable: Value,
    history: Vec<Value>,
    rollouts: Vec<Value>,
    behavior: Value,
    directions: Vec<Value>,
    pub selected: Value,
    pub last_result: Value,
    pub token_query: String,
    evidence: BTreeMap<String, Value>,
    visible_requests: BTreeMap<String, String>,
    receipts: Vec<Value>,
    cache: Vec<(String, Value)>,
    consecutive_repeats: usize,
    working_targets: Value,
    working_tokens: Value,
    seen_targets: BTreeSet<String>,
    seen_tokens: BTreeSet<u64>,
}
impl Session {
    pub fn new(
        catalogue: Value,
        inspection: Value,
        editable: Value,
        history: &[Value],
        best: &Value,
    ) -> Self {
        Self {
            catalogue,
            inspection,
            editable,
            history: history_summary(history, history.len()),
            rollouts: raw_rollouts(best),
            behavior: behavior_summary(best),
            directions: vec![],
            selected: Value::Null,
            last_result: Value::Null,
            token_query: String::new(),
            evidence: BTreeMap::new(),
            visible_requests: BTreeMap::new(),
            receipts: vec![],
            cache: vec![],
            consecutive_repeats: 0,
            working_targets: json!([]),
            working_tokens: json!([]),
            seen_targets: BTreeSet::new(),
            seen_tokens: BTreeSet::new(),
        }
    }
    // A Session belongs to one retained checkpoint/candidate turn. Summary prose
    // is not part of tool identity. Token pages additionally depend on their view.
    fn request_key(&self, request: &Value) -> String {
        let action = request["action"].as_str().unwrap_or("");
        let identity = json!({
            "action": action,
            "query": request["query"].as_str().unwrap_or(""),
            "offset": request["offset"].as_u64().unwrap_or(0),
            "skill_id": if action == "read_skill" { request["skill_id"].as_str().unwrap_or("") } else { "" },
            "probe": if action == "probe_activations" { request["probe"].clone() } else { Value::Null },
            "token_view": if action == "inspect_tokens" { &self.token_query } else { "" },
        });
        format!("{:x}", Sha256::digest(identity.to_string().as_bytes()))
    }
    /// Check before worker dispatch, so identical probes do not run twice.
    /// Cheap view tools are re-applied to restore their selection/working page.
    pub fn repeated_result(&mut self, request: &Value) -> Option<(Value, bool)> {
        let key = self.request_key(request);
        let result = self.cache.iter().find(|(k, _)| k == &key)?.1.clone();
        let action = request["action"].as_str().unwrap_or("");
        let visible = self.visible_requests.get(action) == Some(&key)
            && (self
                .evidence
                .get(action)
                .is_none_or(|e| e["result"].get("truncated").is_none())
                || self.last_result == result);
        if matches!(action, "read_skill" | "inspect_targets" | "inspect_tokens") {
            return self
                .execute(request)
                .ok()
                .map(|restored| (restored, visible));
        }
        Some((result, visible))
    }
    pub fn record_result(&mut self, request: &Value, result: Value, repeated: bool) {
        let action = request["action"].as_str().unwrap_or("");
        let key = self.request_key(request);
        self.consecutive_repeats = if repeated {
            self.consecutive_repeats + 1
        } else {
            0
        };
        // A rejected proposal isn't a read and must be revalidated against newly
        // inspected targets/tokens/handles. Errors are not reusable evidence.
        if result.get("error").is_none() && action != "propose" {
            self.visible_requests.insert(action.into(), key.clone());
            if !repeated {
                self.cache.retain(|(old, _)| old != &key);
                if result.to_string().len() <= CACHE_BYTES / CACHE_ENTRIES {
                    self.cache.push((key.clone(), result.clone()));
                    if self.cache.len() > CACHE_ENTRIES {
                        self.cache.remove(0);
                    }
                }
            }
            if matches!(
                action,
                "read_behavior"
                    | "read_rollouts"
                    | "read_history"
                    | "probe_activations"
                    | "inspect_probe_modules"
            ) {
                self.evidence.insert(
                    action.into(),
                    json!({
                        "request_id": key, "query": request["query"],
                        "offset": request["offset"], "result": evidence_preview(&result),
                    }),
                );
            }
        }
        self.receipts.push(json!({
            "request_id": key, "action": action, "query": short(&request["query"], 256),
            "offset": request["offset"], "summary": short(&request["summary"], 160),
            "status": if result.get("error").is_some() { "rejected" } else if repeated { "reused" } else { "completed" },
            "count": result["count"], "next_offset": result["next_offset"],
        }));
        if self.receipts.len() > RECEIPTS {
            self.receipts.remove(0);
        }
        self.last_result = result;
    }
    pub fn repeated_without_progress(&self) -> bool {
        self.consecutive_repeats >= 2
    }

    /// Retain only compact measured handles when later read tools replace the
    /// current tool result. Sessions are recreated after each candidate turn.
    pub fn remember_probe_result(&mut self, result: &Value) {
        // New measurements can evict worker-side handles. Only the latest
        // measurement may be replayed; older requests must measure again.
        self.cache
            .retain(|(_, value)| value.get("probes").is_none());
        self.visible_requests.remove("probe_activations");
        for probe in result["probes"].as_array().into_iter().flatten().take(4) {
            let Some(id) = probe["direction_id"]
                .as_str()
                .filter(|id| !id.is_empty() && id.len() <= 160)
            else {
                continue;
            };
            if !probe["target"].is_string() {
                continue;
            }
            let mut retained = pick(
                probe,
                &[
                    "target",
                    "direction_id",
                    "dimension",
                    "direction_projection",
                ],
            );
            retained["method"] = short(&probe["evidence"]["method"], 80);
            retained["raw_direction_norm"] = probe["evidence"]["raw_direction_norm"].clone();
            retained["top_columns"] = json!(probe["top_columns"]
                .as_array()
                .into_iter()
                .flatten()
                .take(8)
                .map(|column| pick(
                    column,
                    &["column", "projection", "mean_abs_activation", "score"]
                ))
                .collect::<Vec<_>>());
            self.directions.retain(|old| old["direction_id"] != id);
            self.directions.push(retained);
            if self.directions.len() > 8 {
                self.directions.remove(0);
            }
        }
    }
    pub fn refresh(&mut self, inspection: Value, catalogue: Value, query: String) {
        self.inspection = inspection;
        self.catalogue = catalogue;
        self.token_query = query;
        self.seen_tokens.clear();
        self.working_tokens = json!([]);
        // Refresh may replace tokenizer/inspection metadata; never replay old
        // cached working pages or probes after this explicit state transition.
        self.cache.clear();
        self.visible_requests.clear();
        self.consecutive_repeats = 0;
    }
    pub fn context(&self, base: &Value) -> Value {
        let mut context = base.clone();
        context["discovery"] = discovery(&self.catalogue);
        context["behavior"] = self.behavior.clone();
        context["measured_directions"] = json!(self.directions);
        context["selected_skill"] = self.selected.clone();
        context["tool_result"] = self.last_result.clone();
        context["recent_evidence"] = json!(self.evidence);
        context["recent_requests"] = json!(self.receipts);
        context["inspection_progress"] = json!({
            "consecutive_reused_requests": self.consecutive_repeats,
            "policy": "Combine retained evidence before requesting more. Identical same-state requests reuse results and add no evidence. After a reused request, choose a new justified query/page/probe, propose, or stop. A second consecutive reused request stops inspection without changing weights or buying another editor call. This is a no-progress guard, not a candidate-turn limit."
        });
        context["working_set"] =
            json!({"targets":self.working_targets,"tokens":self.working_tokens});
        context
    }
    pub fn execute(&mut self, request: &Value) -> Result<Value> {
        let action = request["action"].as_str().unwrap_or("");
        let query = request["query"].as_str().unwrap_or("");
        if query.chars().count() > 256 {
            bail!("Tool query exceeds 256 characters");
        }
        let offset = request["offset"].as_u64().unwrap_or(0) as usize;
        if offset > MAX_OFFSET {
            bail!("Tool page offset is out of range");
        }
        if action == "read_skill" {
            let id = request["skill_id"].as_str().unwrap_or("");
            let card = self.catalogue["cards"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|c| c["id"] == id)
                .ok_or_else(|| {
                    anyhow::anyhow!("Skill is unavailable; choose an available discovery entry")
                })?;
            self.selected = pick(
                card,
                &[
                    "id",
                    "operations",
                    "when_to_use",
                    "plan",
                    "evidence",
                    "limitation",
                    "tensor_scope",
                ],
            );
            return Ok(json!({"tool":action,"skill_id":self.selected["id"],"available":true}));
        }
        if action == "read_rollouts" && !query.is_empty() {
            let parts: Vec<_> = query.split('/').collect();
            let field = parts.get(1).copied().unwrap_or("prediction");
            if parts.len() > 2 || !["prompt", "target", "prediction"].contains(&field) {
                bail!(
                    "Use a non-test row ID, optionally followed by /prompt, /target or /prediction"
                );
            }
            let row = self
                .rollouts
                .iter()
                .find(|r| r["row_id"] == parts[0])
                .ok_or_else(|| anyhow::anyhow!("Unknown or unavailable non-test row ID"))?;
            let text = row[field].as_str().unwrap_or("");
            let total = text.chars().count();
            if offset > total {
                bail!("Text offset exceeds this field's character count");
            }
            let page: String = text.chars().skip(offset).take(TEXT_PAGE).collect();
            let end = offset + page.chars().count();
            return Ok(
                json!({"tool":action,"row_id":row["row_id"],"split":row["split"],"field":field,"offset":offset,"count":1,"text":page,"total_chars":total,"next_offset":if end<total{json!(end)}else{Value::Null},"generated_tokens":row["generated_tokens"],"finish_reason":row["finish_reason"]}),
            );
        }
        let rows: Vec<Value> = match action {
            "inspect_targets" => self
                .editable
                .as_array()
                .into_iter()
                .flatten()
                .filter(|v| {
                    query.is_empty()
                        || ["name", "aliases", "roles", "operations"]
                            .iter()
                            .any(|key| match &v[*key] {
                                Value::String(name) => name.contains(query),
                                Value::Array(items) => items.iter().any(|item| {
                                    item.as_str().is_some_and(|name| name.contains(query))
                                }),
                                _ => false,
                            })
                })
                .map(|v| {
                    pick(
                        v,
                        &["name", "shape", "operations", "roles", "aliases", "tied"],
                    )
                })
                .collect(),
            "inspect_probe_modules" => self.inspection["probe_modules"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|row| {
                    query.is_empty()
                        || row["name"]
                            .as_str()
                            .is_some_and(|name| name.contains(query))
                })
                .map(|row| {
                    pick(
                        row,
                        &[
                            "name",
                            "input_width",
                            "output_width",
                            "direction_projection",
                        ],
                    )
                })
                .collect(),
            "inspect_tokens" => self.inspection["token_candidates"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|r| short(r, 256))
                .collect(),
            "read_history" => self.history.clone(),
            "read_rollouts" => self.rollouts.iter().map(rollout_preview).collect(),
            "read_behavior" => self
                .rollouts
                .iter()
                .filter(|row| {
                    query.is_empty()
                        || row["row_id"] == query
                        || row["split"] == query
                        || row["behavior"]["signals"]
                            .as_array()
                            .is_some_and(|signals| signals.iter().any(|signal| signal == query))
                })
                .map(|row| {
                    pick(
                        row,
                        &[
                            "row_id",
                            "split",
                            "generated_tokens",
                            "finish_reason",
                            "exact_match",
                            "behavior",
                        ],
                    )
                })
                .collect(),
            _ => bail!("Unknown read-only tool"),
        };
        let page: Vec<_> = rows.iter().skip(offset).take(PAGE).cloned().collect();
        if action == "inspect_targets" {
            self.working_targets = json!(page);
        }
        if action == "inspect_tokens" {
            self.working_tokens = json!(page);
        }
        for entry in &page {
            if action == "inspect_targets" {
                if let Some(name) = entry["name"].as_str() {
                    self.seen_targets.insert(name.into());
                }
            }
            if action == "inspect_tokens" {
                if let Some(id) = entry["id"].as_u64() {
                    self.seen_tokens.insert(id);
                }
            }
        }
        Ok(
            json!({"tool":action,"offset":offset,"count":page.len(),"total":rows.len(),"next_offset":if offset.saturating_add(page.len())<rows.len(){json!(offset+page.len())}else{Value::Null},"items":page}),
        )
    }
    pub fn validate(&self, proposal: &Value) -> Result<()> {
        if proposal["operation"] == "direction_projection" {
            let handle = self.directions.iter().find(|direction| {
                direction["direction_id"] == proposal["direction_id"]
                    && direction["direction_projection"] == true
                    && proposal["targets"].as_array().is_some_and(|targets| {
                        targets.len() == 1 && targets[0] == direction["target"]
                    })
            });
            if handle.is_none() {
                bail!("Probe and retain a matching editable direction handle before proposing a direction projection");
            }
        }
        for field in ["strategy", "plan", "rationale", "reflection"] {
            if proposal[field]
                .as_str()
                .is_none_or(|text| text.trim().is_empty())
            {
                bail!("A public {field} is required");
            }
        }
        if proposal["targets"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|t| t.as_str().is_none_or(|n| !self.seen_targets.contains(n)))
        {
            bail!("Inspect selected targets before proposing");
        }
        if ["token_ids", "source_token_ids"].iter().any(|field| {
            proposal[*field]
                .as_array()
                .into_iter()
                .flatten()
                .any(|id| id.as_u64().is_none_or(|id| !self.seen_tokens.contains(&id)))
        }) {
            bail!("Inspect token IDs in the current token view before proposing");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn session() -> Session {
        Session::new(
            json!({"cards":[{"id":"test","operations":["scale"],"plan":"guide","eligible_tables":["SECRET_MANIFEST"]}]}),
            json!({"token_candidates":[{"id":3,"text":"hi"}]}),
            json!([{"name":"layer.weight","operations":["scale"]}]),
            &[],
            &json!({"samples":[{"split":"test","prompt":"SEALED"},{"split":"edit","prompt":"visible"}]}),
        )
    }
    fn request(action: &str, query: &str, offset: usize) -> Value {
        json!({"action":action,"query":query,"offset":offset,"skill_id":"","summary":"Inspect evidence","probe":null,"proposal":null})
    }
    fn run_read(s: &mut Session, request: &Value) -> bool {
        let (result, repeated) = s
            .repeated_result(request)
            .unwrap_or_else(|| (s.execute(request).unwrap(), false));
        s.record_result(request, result, repeated);
        repeated
    }
    #[test]
    fn alternating_reads_retain_evidence_then_stop_only_on_no_progress() {
        let mut s = session();
        let behavior = request("read_behavior", "", 0);
        let rollouts = request("read_rollouts", "", 0);
        assert!(!run_read(&mut s, &behavior));
        assert!(!run_read(&mut s, &rollouts));
        let c = s.context(&json!({}));
        assert!(c["recent_evidence"]["read_behavior"]["result"]["items"].is_array());
        assert!(c["recent_evidence"]["read_rollouts"]["result"]["items"].is_array());
        assert!(!c.to_string().contains("SEALED"));
        assert!(run_read(&mut s, &behavior));
        assert!(!s.repeated_without_progress()); // one more decision after warning
        let mut differently_worded = rollouts.clone();
        differently_worded["summary"] = json!("Read again with a different explanation");
        assert!(run_read(&mut s, &differently_worded));
        assert!(s.repeated_without_progress()); // controller stops before next paid call
    }
    #[test]
    fn pages_queries_eviction_and_refresh_are_progress() {
        let mut s = session();
        let first = request("read_rollouts", "", 0);
        run_read(&mut s, &first);
        run_read(&mut s, &request("read_rollouts", "", 8));
        assert!(!run_read(&mut s, &first)); // old page no longer visible, restore it
        for offset in 1..30 {
            run_read(&mut s, &request("read_rollouts", "", offset));
        }
        assert!(s.repeated_result(&first).is_none()); // bounded cache eviction
        assert!(!run_read(&mut s, &first)); // still fetchable
        assert!(!run_read(&mut s, &request("read_behavior", "edit", 0)));
        let token = request("inspect_tokens", "", 0);
        run_read(&mut s, &token);
        assert!(run_read(&mut s, &token));
        s.refresh(
            json!({"token_candidates":[{"id":7}]}),
            json!({}),
            "changed".into(),
        );
        assert!(s.repeated_result(&token).is_none());
        assert!(!run_read(&mut s, &token));
        assert_eq!(s.last_result["items"][0]["id"], 7);
        assert!(s.seen_tokens.contains(&7));
        assert!(!s.seen_tokens.contains(&3));
        assert!(!s.repeated_without_progress());
        assert!(session().repeated_result(&first).is_none());
    }
    #[test]
    fn repeated_probe_is_reused_but_new_measurements_invalidate_handles_cache() {
        let mut s = session();
        let mut probe = request("probe_activations", "", 0);
        probe["probe"] = json!({"modules":["layer"],"positions":[-1]});
        let result = json!({"probes":[{"direction_id":"a","target":"layer.weight"}],"state_id":1});
        s.remember_probe_result(&result);
        s.record_result(&probe, result.clone(), false);
        assert_eq!(s.repeated_result(&probe), Some((result, true)));
        let mut new_probe = probe.clone();
        new_probe["probe"]["positions"] = json!([0]);
        assert!(s.repeated_result(&new_probe).is_none());
        s.remember_probe_result(&json!({"probes":[{"direction_id":"b","target":"layer.weight"}]}));
        assert!(s.repeated_result(&probe).is_none());
    }
    #[test]
    fn evidence_and_receipts_are_bounded_over_long_sessions() {
        let mut s = session();
        for n in 0..200 {
            let r = request("read_rollouts", "", n);
            let result = json!({"tool":"read_rollouts","items":vec![json!({"prediction":"界".repeat(10000) }); 8]});
            s.record_result(&r, result, false);
        }
        // Last result remains the caller's bounded page; historical retention
        // has a separate strict bound even for deliberately oversized fixtures.
        s.last_result = Value::Null;
        let c = s.context(&json!({}));
        assert!(c["recent_evidence"].to_string().len() < EVIDENCE_BYTES + 1024);
        assert_eq!(c["recent_requests"].as_array().unwrap().len(), RECEIPTS);
        assert!(s.cache.len() <= CACHE_ENTRIES);
        assert!(
            s.cache
                .iter()
                .map(|(_, v)| v.to_string().len())
                .sum::<usize>()
                <= CACHE_BYTES
        );
        assert!(c.to_string().len() < 30_000);
    }
    #[test]
    fn truncated_previous_evidence_can_be_recovered_without_stall() {
        let mut s = session();
        let r = request("read_rollouts", "", 0);
        let result = json!({"tool":"read_rollouts","text":"x".repeat(10000)});
        s.record_result(&r, result.clone(), false);
        run_read(&mut s, &request("read_behavior", "", 0));
        assert_eq!(s.repeated_result(&r), Some((result, false)));
    }
    #[test]
    fn initial_context_is_discovery_not_manifest() {
        let s = session();
        let c = s.context(&json!({}));
        assert!(!c.to_string().contains("SECRET_MANIFEST"));
        assert!(!c.to_string().contains("layer.weight"));
        assert!(!c.to_string().contains("SEALED"));
        assert!(c["selected_skill"].is_null());
    }
    #[test]
    fn custom_strategies_require_inspected_target_not_a_skill_whitelist() {
        let mut s = session();
        let p = json!({"strategy":"custom evidence-led choice","plan":"Small candidate","rationale":"Measured reference error","reflection":"No previous candidates","operation":"scale","targets":["layer.weight"],"token_ids":[],"source_token_ids":[]});
        assert!(s.validate(&p).is_err());
        s.execute(&json!({"action":"read_skill","skill_id":"test"}))
            .unwrap();
        assert!(s.validate(&p).is_err());
        s.execute(&json!({"action":"inspect_targets"})).unwrap();
        assert!(s.validate(&p).is_ok());
        assert!(!s.selected.to_string().contains("SECRET_MANIFEST"));
        s.selected = Value::Null;
        assert!(s.validate(&p).is_ok());
    }
    #[test]
    fn pages_are_bounded_and_test_rows_sealed() {
        let mut s = session();
        s.editable = json!((0..1000)
            .map(|n| json!({"name":format!("layer.{n}")}))
            .collect::<Vec<_>>());
        let p = s
            .execute(&json!({"action":"inspect_targets","offset":8}))
            .unwrap();
        assert_eq!(p["items"].as_array().unwrap().len(), 8);
        assert_eq!(p["next_offset"], 16);
        assert!(!s
            .execute(&json!({"action":"read_rollouts"}))
            .unwrap()
            .to_string()
            .contains("SEALED"));
        assert!(s.execute(&json!({"action":"shell"})).is_err());
    }
    #[test]
    fn context_does_not_accumulate_tool_payloads() {
        let mut s = session();
        s.last_result = json!({"old":"OLD"});
        s.last_result = json!({"new":"NEW"});
        let c = s.context(&json!({}));
        assert!(!c.to_string().contains("OLD"));
        assert!(c.to_string().contains("NEW"));
    }
    #[test]
    fn discovery_does_not_scale_with_manifest_or_full_history() {
        let mut s = session();
        let small = s.context(&json!({})).to_string();
        s.editable = json!((0..10000)
            .map(|n| json!({"name":format!("layer.{n}"),"aliases":["x".repeat(10000)]}))
            .collect::<Vec<_>>());
        s.history = vec![json!({"massive":"x".repeat(10000)}); 100];
        assert_eq!(small, s.context(&json!({})).to_string());
        let result = s.execute(&json!({"action":"inspect_targets"})).unwrap();
        assert!(result.to_string().len() < 8000);
    }
    #[test]
    fn target_queries_match_aliases_and_operations() {
        let mut s = session();
        s.editable = json!([{"name":"embed","aliases":["lm_head.weight"],"operations":["token_row_residual"]}]);
        for query in ["lm_head", "token_row_residual"] {
            assert_eq!(
                s.execute(&json!({"action":"inspect_targets","query":query}))
                    .unwrap()["count"],
                1
            );
        }
    }
    #[test]
    fn stop_is_a_valid_nonmutating_action_and_requests_are_bounded() {
        let a = json!({"action":"stop","skill_id":"","query":"","offset":0,"summary":"No justified candidate remains.","proposal":null,"probe":null});
        assert!(validate_action(&a).is_ok());
        let mut bad = a.clone();
        bad["proposal"] = json!({});
        assert!(validate_action(&bad).is_err());
        bad = a.clone();
        bad["query"] = json!("x".repeat(257));
        assert!(validate_action(&bad).is_err());
        bad = a;
        bad["action"] = json!("execute");
        assert!(validate_action(&bad).is_err());
    }
    #[test]
    fn stateless_requests_retain_only_latest_target_and_token_pages() {
        let mut s = session();
        s.execute(&json!({"action":"inspect_targets"})).unwrap();
        s.execute(&json!({"action":"inspect_tokens"})).unwrap();
        s.last_result = s.execute(&json!({"action":"read_history"})).unwrap();
        let c = s.context(&json!({}));
        assert_eq!(c["working_set"]["targets"][0]["name"], "layer.weight");
        assert_eq!(c["working_set"]["tokens"][0]["id"], 3);
        s.refresh(
            json!({"token_candidates":[]}),
            json!({"cards":[]}),
            "new".into(),
        );
        assert_eq!(s.context(&json!({}))["working_set"]["tokens"], json!([]));
    }
    #[test]
    fn health_and_control_summaries_preserve_regression_evidence_only() {
        let health = health_comparison(
            &json!({"copy":{"score":0.4,"count":10,"samples":["SECRET"]}}),
            &json!({"copy":{"score":0.9}}),
        );
        assert_eq!(health["largest_drops"][0]["change"], -0.5);
        assert!(!health.to_string().contains("SECRET"));
        let history = history_summary(
            &[json!({"health_passed":false,"controls":{"status":"failed","checks":["SECRET"]}})],
            3,
        );
        assert_eq!(history[0]["health_passed"], false);
        assert_eq!(history[0]["controls"]["status"], "failed");
        assert!(!json!(history).to_string().contains("SECRET"));
    }
    #[test]
    fn custom_strategy_cannot_bypass_shared_operation_validation() {
        let p = json!({"strategy":"custom measured hypothesis","plan":"Small edit","rationale":"Reference error","reflection":"No previous candidates","operation":"execute","targets":["layer.weight"],"token_ids":[],"source_token_ids":[],"strength":0.01,"rank":1,"seed":42});
        assert!(crate::search::validate_proposal(
            &p,
            &json!([{"name":"layer.weight","operations":["scale"]}])
        )
        .is_err());
        let mut supported = p;
        supported["operation"] = json!("scale");
        assert!(crate::search::validate_proposal(
            &supported,
            &json!([{"name":"layer.weight","operations":["scale"]}])
        )
        .is_ok());
    }
    #[test]
    fn behavior_summary_is_bounded_and_sealed_from_test_rows() {
        let mut samples = vec![
            json!({"split":"test","row_id":"test:0","prediction":"SEALED","behavior":{"signals":["repetition"]}}),
        ];
        samples.extend((0..1000).map(|_| json!({"split":"validation","prediction":"x".repeat(10000),"behavior":{"signals":["repetition","repetition","invented"],"evidence":{"private":"SECRET"}}})));
        let best = json!({"samples":samples,"behavior":{"untrusted":"SEALED"}});
        let summary = behavior_summary(&best);
        assert_eq!(summary["sample_count"], 1000);
        assert_eq!(summary["signals"]["repetition"]["count"], 1000);
        assert_eq!(
            summary["signals"]["repetition"]["row_ids"]
                .as_array()
                .unwrap()
                .len(),
            8
        );
        assert_eq!(summary["signals"]["repetition"]["row_ids_truncated"], true);
        let text = summary.to_string();
        assert!(text.len() < 6000);
        for secret in ["SEALED", "SECRET", "invented", "test:"] {
            assert!(!text.contains(secret));
        }
        let s = Session::new(json!({}), json!({}), json!([]), &[], &best);
        let context = s.context(&json!({})).to_string();
        assert!(context.len() < 15000);
        assert!(!context.contains("xxxx"));
    }
    #[test]
    fn behavior_details_filter_by_signal_or_row_without_raw_traces() {
        let best = json!({"samples":[
            {"split":"test","row_id":"test:0","prediction":"SEALED","behavior":{"signals":["repetition"]}},
            {"split":"validation","row_id":"validation:14","prediction":"FULL_TRACE","generated_tokens":200,"finish_reason":"length","behavior":{"signals":["repetition","length_truncation"],"evidence":{"repetition":{"copies":8}}}},
            {"split":"preserve","prediction":"other","behavior":{"signals":["terminated"]}}
        ]});
        let mut s = Session::new(json!({}), json!({}), json!([]), &[], &best);
        assert_eq!(
            s.behavior["signals"]["repetition"]["row_ids"][0],
            "validation:14"
        );
        for query in ["repetition", "validation:14", "validation"] {
            let result = s
                .execute(&json!({"action":"read_behavior","query":query}))
                .unwrap();
            assert_eq!(result["count"], 1);
            assert_eq!(
                result["items"][0]["behavior"]["evidence"]["repetition"]["copies"],
                8
            );
            assert!(!result.to_string().contains("FULL_TRACE"));
            assert!(!result.to_string().contains("SEALED"));
        }
        assert_eq!(
            s.execute(&json!({"action":"read_behavior","query":"test:0"}))
                .unwrap()["count"],
            0
        );
    }
    #[test]
    fn rollout_text_pages_are_complete_unicode_safe_and_bounded() {
        let text = "α🦀".repeat(2500);
        let best = json!({"samples":[{"split":"test","prediction":"SEALED"},{"split":"edit","prediction":text,"prompt":"question","target":"answer","generated_tokens":2048,"finish_reason":"length"}]});
        let mut s = Session::new(json!({}), json!({}), json!([]), &[], &best);
        let preview = s.execute(&json!({"action":"read_rollouts"})).unwrap();
        let row = &preview["items"][0];
        assert_eq!(row["row_id"], "edit:0");
        assert_eq!(row["generated_tokens"], 2048);
        assert_eq!(row["finish_reason"], "length");
        assert_eq!(row["prediction_chars"], 5000);
        assert_eq!(row["prediction_truncated"], true);
        assert_eq!(row["prediction"].as_str().unwrap().chars().count(), 768);
        let mut offset = 0;
        let mut rebuilt = String::new();
        loop {
            let page = s
                .execute(
                    &json!({"action":"read_rollouts","query":"edit:0/prediction","offset":offset}),
                )
                .unwrap();
            assert!(page["text"].as_str().unwrap().chars().count() <= TEXT_PAGE);
            rebuilt.push_str(page["text"].as_str().unwrap());
            match page["next_offset"].as_u64() {
                Some(next) => offset = next,
                None => break,
            }
        }
        assert_eq!(rebuilt, text);
        assert_eq!(
            s.execute(&json!({"action":"read_rollouts","query":"edit:0/target"}))
                .unwrap()["text"],
            "answer"
        );
        assert!(s
            .execute(&json!({"action":"read_rollouts","query":"test:0"}))
            .is_err());
        assert!(s
            .execute(&json!({"action":"read_rollouts","query":"edit:0/token_ids"}))
            .is_err());
        assert!(s
            .execute(&json!({"action":"read_rollouts","query":"edit:0","offset":5001}))
            .is_err());
    }
    #[test]
    fn missing_behavior_data_does_not_invent_clean_results() {
        let summary =
            behavior_summary(&json!({"samples":[{"split":"edit","prediction":"unknown"}]}));
        assert_eq!(summary["sample_count"], 1);
        assert_eq!(summary["diagnosed_count"], 0);
        assert_eq!(summary["signals"]["terminated"]["count"], 0);
    }
    #[test]
    fn probe_actions_require_bounded_structured_arguments() {
        let mut action = json!({"action":"probe_activations","skill_id":"","query":"","offset":0,"summary":"Measure a reference contrast","proposal":null,"probe":{"targets":["layer.weight"],"contexts":[{"row_id":"edit:0","source":"prompt","positions":[-1]}],"contrast_contexts":[],"top_k":64}});
        assert!(validate_action(&action).is_ok());
        action["probe"]["top_k"] = json!(129);
        assert!(validate_action(&action).is_err());
        action["probe"]["top_k"] = json!(64);
        action["action"] = json!("read_behavior");
        assert!(validate_action(&action).is_err());
        action["probe"] = Value::Null;
        assert!(validate_action(&action).is_ok());
        action.as_object_mut().unwrap().remove("probe");
        assert!(validate_action(&action).is_err());
    }
    #[test]
    fn history_keeps_diagnostic_counts_but_not_text_or_unbounded_ids() {
        let history = history_summary(
            &[
                json!({"behavior":{"sample_count":20,"diagnosed_count":20,"signals":{"repetition":{"count":3,"row_ids":["SECRET"]},"invented":{"count":999}},"samples":["SEALED"]}}),
            ],
            3,
        );
        assert_eq!(history[0]["behavior"]["signals"]["repetition"], 3);
        let text = json!(history).to_string();
        assert!(!text.contains("SECRET"));
        assert!(!text.contains("SEALED"));
        assert!(!text.contains("invented"));
    }
    #[test]
    fn measured_direction_handles_are_bounded_and_survive_read_tools() {
        let mut s = session();
        for i in 0..12 {
            s.remember_probe_result(&json!({"probes":[{"target":"layer.weight","direction_id":format!("direction-{i}"),"dimension":10000,"direction_projection":true,"vector":["SECRET_VECTOR"],"evidence":{"method":"output_mean","raw_direction_norm":12,"contexts":["SECRET_CONTEXT"]},"top_columns":(0..128).map(|column|json!({"column":column,"score":1,"vector":"SECRET_VECTOR"})).collect::<Vec<_>>()}]}));
        }
        s.last_result = s.execute(&json!({"action":"inspect_targets"})).unwrap();
        let context = s.context(&json!({}));
        assert_eq!(context["measured_directions"].as_array().unwrap().len(), 8);
        assert_eq!(
            context["measured_directions"][0]["direction_id"],
            "direction-4"
        );
        assert_eq!(
            context["measured_directions"][0]["top_columns"]
                .as_array()
                .unwrap()
                .len(),
            8
        );
        assert!(!context.to_string().contains("SECRET"));
        let mut proposal = json!({"operation":"direction_projection","direction_id":"direction-11","targets":["layer.weight"],"token_ids":[],"source_token_ids":[],"strategy":"measured direction","plan":"Test a small projection","rationale":"Observed generation contrast","reflection":"No retained candidate yet"});
        assert!(s.validate(&proposal).is_ok());
        proposal["direction_id"] = json!("invented");
        assert!(s.validate(&proposal).is_err());
        proposal["direction_id"] = json!("direction-11");
        proposal["targets"] = json!(["other.weight"]);
        assert!(s.validate(&proposal).is_err());
    }
    #[test]
    fn long_rollout_pages_support_offsets_beyond_one_hundred_thousand() {
        let mut action = json!({"action":"read_rollouts","skill_id":"","query":"edit:0/prediction","offset":100_001,"summary":"Read the long generation tail","proposal":null,"probe":null});
        assert!(validate_action(&action).is_ok());
        let text = "🦀".repeat(100_001) + "verified tail";
        let best = json!({"samples":[{"split":"edit","prediction":text,"generated_tokens":32768,"finish_reason":"length"}]});
        let mut s = Session::new(json!({}), json!({}), json!([]), &[], &best);
        let page = s.execute(&action).unwrap();
        assert_eq!(page["text"], "verified tail");
        assert!(page["next_offset"].is_null());
        action["offset"] = json!(MAX_OFFSET);
        assert!(validate_action(&action).is_ok());
        action["offset"] = json!(MAX_OFFSET + 1);
        assert!(validate_action(&action).is_err());
        assert!(s.execute(&action).is_err());
    }
    #[test]
    fn discovery_distinguishes_direction_projection_bounds_from_legacy_operations() {
        let card = discovery(&json!({}));
        let contract = card["edit_contract"].as_str().unwrap();
        assert!(contract.contains("Except direction_projection"));
        assert!(contract.contains("strength0.000001–16, rank1"));
        assert!(contract.contains("1–256 selected in-range columns"));
        assert!(contract.contains("after dtype rounding"));
    }
}
