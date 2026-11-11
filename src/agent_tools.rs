//! Small discovery surface and bounded, read-only tools shared by CLI and Space.
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::collections::BTreeSet;

const PAGE: usize = 8;
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
                pick(v, &["count", "loss", "exact_match", "token_f1", "semantic"]),
            );
        }
    }
    Value::Object(out)
}
pub fn history_summary(rows: &[Value], limit: usize) -> Vec<Value> {
    rows.iter().rev().take(limit).map(|r| json!({"turn":r["turn"], "accepted":r["accepted"], "strategy":short(&r["proposal"]["strategy"],80), "operation":r["proposal"]["operation"], "strength":r["proposal"]["strength"], "metrics":measurements(&r["metrics"]), "decision":short(&r["decision_reason"],256),"health_passed":r["health_passed"],"controls":pick(&r["controls"],&["status","retention","checked","applicable"])})).collect()
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
pub fn rollouts(best: &Value) -> Vec<Value> {
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
            pick(
                r,
                &[
                    "split",
                    "prompt",
                    "target",
                    "prediction",
                    "exact_match",
                    "token_f1",
                    "semantic",
                    "loss",
                ],
            )
        })
        .collect()
}
pub fn discovery(catalogue: &Value) -> Value {
    let cards: Vec<_> = catalogue["cards"].as_array().into_iter().flatten().map(|c| json!({"id":c["id"],"operations":c["operations"],"use_when":short(&c["when_to_use"],160),"available":true})).collect();
    json!({"skills":cards,"available_operations":catalogue["available_operations"],"unavailable":[{"id":"rome","reason":"Not implemented: requires causal localisation, covariance statistics and optimised latent targets."},{"id":"memit","reason":"Not implemented: requires multi-layer covariance solves and optimised latent targets."},{"id":"patch_arithmetic","reason":"Agent-directed patch composition/negation is not implemented. Saved journal patches can be replayed by the materialize-patch utility; repeating an edit is not patch arithmetic."}],"tools":[{"name":"read_skill","purpose":"Load optional focused guidance for one suggested strategy."},{"name":"inspect_targets","purpose":"Read eight supported tensors; query matches name, alias, role or operation."},{"name":"inspect_tokens","purpose":"Inspect tokenizer-grounded IDs and norms; optional query tokenizes one text up to 256 characters; offset pages eight candidates."},{"name":"read_history","purpose":"Read eight measured candidate outcomes and preservation/control decisions."},{"name":"read_rollouts","purpose":"Read eight current non-test examples, outputs and measured scores."},{"name":"propose","purpose":"Submit one bounded edit for controller validation, application, testing and keep/revert."},{"name":"stop","purpose":"End search without another edit, explaining why."}],"operation_contracts":{"scale":"MLP: W*(1+strength).","low_rank":"MLP: seeded low-rank direction, rank1–16, relative norm abs(strength).","noise":"MLP: seeded full-rank direction, relative norm abs(strength).","activation_rank_one":"MLP: forward reference keys with target-minus-predicted embedding directions; rank selects examples.","activation_preserve":"MLP: contrastive reference direction projected off measured preserve-key span; requires edit/preserve references and positive strength.","token_row_scale":"One table: multiply inspected non-special destination rows by1+strength; empty sources.","token_row_interpolate":"One table: positive bounded blend toward paired inspected source IDs; distinct source/destination; stored-row norm bound may reduce fraction.","token_row_residual":"Supported output table: positive measured target-minus-probability head-input residual on inspected edit_reference rows; rank selects examples; empty sources."},"edit_contract":"Only available_operations may be used, and inspected targets must support them. Other operation contracts describe capabilities that this model/dataset may not have. Strength magnitude0.000001–0.1, rank1–16; MLP operations empty token selectors. Row operations one canonical table and1–32 inspected non-special IDs. Tied aliases are one physical edit affecting both roles. Proposed writes are tested before retention. Use provided tools before considering unsupported custom gradient optimisers; arbitrary code/autograd/training is not available.","instructions":"Skills are optional suggestions, not a strategy whitelist. Inspect needed targets/tokens, then propose an available operation with your own concise strategy, plan and measured rationale. Read a skill only when its guidance helps. Tools are read-only and paginated (8 entries). Use query to filter target names/aliases/roles/operations, or inspect one token text (up to 256 characters) with the current tokenizer. An empty token query pages the current view; a new text replaces it. Use offset to page. Read history/rollouts for measured evidence. Do not request the entire manifest. Tool calls consume frontier budget; stop if no justified candidate remains. The selected skill, latest target/token pages and latest other tool result remain in context. Older pages are not replayed; fetch another page when needed."})
}

pub fn validate_action(action: &Value) -> Result<()> {
    let fields = [
        "action", "skill_id", "query", "offset", "summary", "proposal",
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
        "propose",
        "stop",
    ]
    .contains(&action["action"].as_str().unwrap_or(""))
        || action["offset"].as_u64().is_none_or(|n| n > 100_000)
    {
        bail!("Invalid agent action or offset");
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

pub struct Session {
    pub catalogue: Value,
    pub inspection: Value,
    editable: Value,
    history: Vec<Value>,
    rollouts: Vec<Value>,
    pub selected: Value,
    pub last_result: Value,
    pub token_query: String,
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
            rollouts: rollouts(best),
            selected: Value::Null,
            last_result: Value::Null,
            token_query: String::new(),
            working_targets: json!([]),
            working_tokens: json!([]),
            seen_targets: BTreeSet::new(),
            seen_tokens: BTreeSet::new(),
        }
    }
    pub fn refresh(&mut self, inspection: Value, catalogue: Value, query: String) {
        self.inspection = inspection;
        self.catalogue = catalogue;
        self.token_query = query;
        self.seen_tokens.clear();
        self.working_tokens = json!([]);
    }
    pub fn context(&self, base: &Value) -> Value {
        let mut context = base.clone();
        context["discovery"] = discovery(&self.catalogue);
        context["selected_skill"] = self.selected.clone();
        context["tool_result"] = self.last_result.clone();
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
        if offset > 100_000 {
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
            "inspect_tokens" => self.inspection["token_candidates"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|r| short(r, 256))
                .collect(),
            "read_history" => self.history.clone(),
            "read_rollouts" => self.rollouts.clone(),
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
        let a = json!({"action":"stop","skill_id":"","query":"","offset":0,"summary":"No justified candidate remains.","proposal":null});
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
}
