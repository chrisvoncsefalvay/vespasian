use anyhow::{bail, Result};
use serde_json::Value;
use std::collections::HashSet;

pub fn validate_proposal(p: &Value, editable: &Value) -> Result<()> {
    let keys: [&str; 13] = [
        "targets",
        "operation",
        "strength",
        "rank",
        "seed",
        "rationale",
        "strategy",
        "plan",
        "reflection",
        "token_ids",
        "source_token_ids",
        "direction_id",
        "columns",
    ];
    let obj = p
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Editor proposal must be an object"))?;
    if keys[..6].iter().any(|key| !obj.contains_key(*key))
        || obj.keys().any(|key| !keys.contains(&key.as_str()))
    {
        bail!("Editor returned an unexpected proposal shape")
    }
    let targets = p["targets"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Proposal targets must be a list"))?;
    if targets.is_empty() || targets.len() > 4 {
        bail!("Proposal must select one to four tensors")
    }
    let allowed: HashSet<&str> = editable
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|v| v["name"].as_str())
        .collect();
    let mut seen = HashSet::new();
    for t in targets {
        let name = t
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid tensor name"))?;
        if !allowed.contains(name) || !seen.insert(name) {
            bail!("Proposal selected an unavailable or duplicate tensor: {name}")
        }
    }
    if ![
        "scale",
        "low_rank",
        "noise",
        "activation_rank_one",
        "activation_preserve",
        "direction_projection",
        "token_row_residual",
        "token_row_scale",
        "token_row_interpolate",
    ]
    .contains(&p["operation"].as_str().unwrap_or(""))
    {
        bail!("Unknown tensor operation")
    }
    let operation = p["operation"].as_str().unwrap_or("");
    if matches!(operation, "activation_preserve" | "token_row_residual")
        && p["strength"].as_f64().is_none_or(|s| s <= 0.)
    {
        bail!("Measured preservation/residual edits require positive strength");
    }
    let row_operation = operation.starts_with("token_row_");
    for target in targets {
        let descriptor = editable
            .as_array()
            .into_iter()
            .flatten()
            .find(|entry| entry["name"] == *target)
            .unwrap();
        if let Some(operations) = descriptor["operations"].as_array() {
            if !operations.contains(&p["operation"]) {
                bail!("The selected tensor does not support this operation");
            }
        } else if row_operation {
            bail!("Token-row edits require an explicitly supported token table");
        }
    }
    let ids = |field: &str| -> Result<Vec<u64>> {
        match p.get(field) {
            None => Ok(vec![]),
            Some(Value::Array(values)) if values.len() <= 32 => values
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| anyhow::anyhow!("Token IDs must be nonnegative integers"))
                })
                .collect(),
            _ => bail!("Token selectors must be arrays of at most 32 IDs"),
        }
    };
    let tokens = ids("token_ids")?;
    let sources = ids("source_token_ids")?;
    if row_operation {
        if targets.len() != 1
            || tokens.is_empty()
            || tokens.iter().collect::<HashSet<_>>().len() != tokens.len()
        {
            bail!("Token-row edits require one table and one to 32 distinct destination IDs");
        }
        let descriptor = editable
            .as_array()
            .into_iter()
            .flatten()
            .find(|entry| entry["name"] == targets[0])
            .unwrap();
        let rows = descriptor["shape"][0].as_u64().unwrap_or(0);
        if tokens.iter().chain(&sources).any(|id| *id >= rows) {
            bail!("A token ID is outside the table row range");
        }
        if operation == "token_row_interpolate" {
            if sources.len() != tokens.len()
                || tokens.iter().zip(&sources).any(|(a, b)| a == b)
                || p["strength"].as_f64().is_none_or(|v| v <= 0.)
            {
                bail!("Row interpolation requires positive strength and a different source ID for every destination");
            }
        } else if !sources.is_empty() {
            bail!("Row scaling does not accept source token IDs");
        }
    } else if !tokens.is_empty() || !sources.is_empty() {
        bail!("MLP edits must have empty token selectors");
    }
    let strength = p["strength"].as_f64().unwrap_or(f64::NAN);
    if operation == "direction_projection" {
        if targets.len() != 1 || p["rank"].as_u64() != Some(1) {
            bail!("Direction projection requires one tensor and rank one");
        }
        if !strength.is_finite() || !(1e-6..=16.).contains(&strength) {
            bail!("Direction projection strength must be between 0.000001 and 16");
        }
        let handle = p["direction_id"].as_str().unwrap_or("");
        if !handle.starts_with("direction_")
            || handle.len() != 42
            || !handle[10..].bytes().all(|c| c.is_ascii_hexdigit())
        {
            bail!("Direction projection requires a measured direction handle");
        }
        let columns = p["columns"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Direction projection requires columns"))?;
        let descriptor = editable
            .as_array()
            .into_iter()
            .flatten()
            .find(|entry| entry["name"] == targets[0])
            .unwrap();
        let width = descriptor["shape"][1].as_u64().unwrap_or(0);
        let mut selected = HashSet::new();
        if columns.is_empty()
            || columns.len() > 256
            || columns.iter().any(|column| {
                column
                    .as_u64()
                    .is_none_or(|value| value >= width || !selected.insert(value))
            })
        {
            bail!("Select one to 256 distinct in-range input columns");
        }
    } else {
        if p.get("direction_id")
            .is_some_and(|value| !value.is_null() && value.as_str() != Some(""))
            || p.get("columns")
                .is_some_and(|value| value.as_array().is_none_or(|items| !items.is_empty()))
        {
            bail!("Only direction_projection accepts direction_id and columns");
        }
        if !strength.is_finite() || !(1e-6..=0.1).contains(&strength.abs()) {
            bail!("Mutation strength must have magnitude between 0.000001 and 0.1")
        }
    }
    if !(1..=16).contains(&p["rank"].as_u64().unwrap_or(0)) {
        bail!("Mutation rank must be between 1 and 16")
    }
    if p["seed"].as_u64().is_none_or(|s| s > 2147483647) {
        bail!("Mutation seed must be between 0 and 2147483647")
    }
    if p["rationale"].as_str().is_none_or(|s| s.trim().is_empty()) {
        bail!("Editor must explain its proposed alteration")
    }
    for key in ["strategy", "plan", "reflection"] {
        if let Some(value) = obj.get(key) {
            if value.as_str().is_none_or(|text| text.trim().is_empty()) {
                bail!("Editor {key} must be a nonempty summary");
            }
        }
    }
    Ok(())
}
/// Token selection must be grounded in this turn's read-only model inspection.
pub fn validate_inspected_rows(proposal: &Value, inspection: &Value) -> Result<()> {
    if !proposal["operation"]
        .as_str()
        .unwrap_or("")
        .starts_with("token_row_")
    {
        return Ok(());
    }
    let allowed: HashSet<u64> = inspection["token_candidates"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|row| row["id"].as_u64())
        .collect();
    for field in ["token_ids", "source_token_ids"] {
        for id in proposal[field].as_array().into_iter().flatten() {
            if id.as_u64().is_none_or(|value| !allowed.contains(&value)) {
                bail!("Token-row edits must use IDs from the current model inspection");
            }
        }
    }
    Ok(())
}

pub fn metric(v: &Value, split: &str, key: &str) -> Result<f64> {
    let f = v[split][key]
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("Missing {split} {key} measurement"))?;
    if !f.is_finite() {
        bail!("Non-finite {split} {key} measurement")
    }
    Ok(f)
}
pub fn optional_metric(v: &Value, split: &str, key: &str) -> Result<Option<f64>> {
    if v[split][key].is_null() {
        Ok(None)
    } else {
        metric(v, split, key).map(Some)
    }
}
pub fn improves(
    candidate: &Value,
    best: &Value,
    baseline: &Value,
    key: &str,
    tolerance: f64,
) -> Result<bool> {
    if !baseline["preserve"].is_null() {
        if metric(candidate, "preserve", key)? + tolerance + 1e-9
            < metric(baseline, "preserve", key)?
        {
            return Ok(false);
        }
        if let Some(baseline_loss) = optional_metric(baseline, "preserve", "loss")? {
            if metric(candidate, "preserve", "loss")? > baseline_loss * 1.25 + 1e-6 {
                return Ok(false);
            }
        }
    }
    let a = metric(candidate, "validation", key)?;
    let b = metric(best, "validation", key)?;
    let lower_loss = match (
        optional_metric(candidate, "validation", "loss")?,
        optional_metric(best, "validation", "loss")?,
    ) {
        (Some(candidate), Some(best)) => candidate < best - 1e-6,
        _ => false,
    };
    Ok(a > b + 1e-9 || ((a - b).abs() <= 1e-9 && lower_loss))
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn token_rows_require_supported_grounded_bounded_selectors() {
        let editable = json!([{"name":"embed.weight","shape":[10,4],"operations":["token_row_scale","token_row_interpolate"]}]);
        let inspection = json!({"token_candidates":[{"id":1},{"id":2}]});
        let p = json!({"targets":["embed.weight"],"operation":"token_row_interpolate","strength":0.1,"rank":1,"seed":1,"rationale":"Measured rows","token_ids":[1],"source_token_ids":[2]});
        validate_proposal(&p, &editable).unwrap();
        validate_inspected_rows(&p, &inspection).unwrap();
        for (key, value) in [
            ("token_ids", json!([1, 1])),
            ("token_ids", json!([10])),
            ("source_token_ids", json!([1])),
            ("strength", json!(-0.1)),
            ("targets", json!(["embed.weight", "embed.weight"])),
        ] {
            let mut invalid = p.clone();
            invalid[key] = value;
            assert!(validate_proposal(&invalid, &editable).is_err(), "{invalid}");
        }
        let mut ungrounded = p.clone();
        ungrounded["token_ids"] = json!([3]);
        assert!(validate_inspected_rows(&ungrounded, &inspection).is_err());
        let mut generic = p.clone();
        generic["operation"] = json!("scale");
        assert!(validate_proposal(&generic, &editable).is_err());
        assert!(validate_proposal(&p, &json!([{"name":"embed.weight","shape":[10,4]}])).is_err());
    }
    #[test]
    fn measured_projection_requires_rank_one_handle_and_unique_columns() {
        let editable =
            json!([{"name":"layer.weight","shape":[24,48],"operations":["direction_projection"]}]);
        let proposal = json!({"targets":["layer.weight"],"operation":"direction_projection","strength":6.,"rank":1,"seed":1,"rationale":"Measured direction","direction_id":"direction_0123456789abcdef0123456789abcdef","columns":[0,7,47]});
        validate_proposal(&proposal, &editable).unwrap();
        for (key, value) in [
            ("strength", json!(-1)),
            ("strength", json!(17)),
            ("rank", json!(2)),
            ("columns", json!([])),
            ("columns", json!([1, 1])),
            ("columns", json!([48])),
            ("columns", json!([-1])),
            ("direction_id", json!("unknown")),
        ] {
            let mut invalid = proposal.clone();
            invalid[key] = value;
            assert!(validate_proposal(&invalid, &editable).is_err(), "{invalid}");
        }
    }

    fn m(score: f64, loss: f64, p: f64) -> Value {
        json!({"validation":{"exact_match":score,"loss":loss},"preserve":{"exact_match":p,"loss":1.0}})
    }
    #[test]
    fn proposal_explanations_are_validated_without_breaking_legacy_receipts() {
        let mut proposal = serde_json::json!({"targets":["layer.weight"],"operation":"scale","strength":0.01,"rank":1,"seed":1,"rationale":"Measured change"});
        let editable = serde_json::json!([{"name":"layer.weight"}]);
        validate_proposal(&proposal, &editable).unwrap();
        for field in ["strategy", "plan", "reflection"] {
            proposal[field] = serde_json::json!("A concise measured explanation");
        }
        validate_proposal(&proposal, &editable).unwrap();
        proposal["reflection"] = serde_json::json!("");
        assert!(validate_proposal(&proposal, &editable).is_err());
        proposal["reflection"] = serde_json::json!({"private":"data"});
        assert!(validate_proposal(&proposal, &editable).is_err());
    }

    #[test]
    fn only_climbs_and_preserves() {
        let b = m(0.5, 2., 1.);
        assert!(improves(&m(0.6, 3., 1.), &b, &b, "exact_match", 0.).unwrap());
        assert!(!improves(&m(0.6, 1., 0.), &b, &b, "exact_match", 0.).unwrap());
        assert!(!improves(&m(0.4, 1., 1.), &b, &b, "exact_match", 0.).unwrap());
        assert!(improves(&m(0.5, 1.9, 1.), &b, &b, "exact_match", 0.).unwrap());
        assert!(!improves(&b, &b, &b, "exact_match", 0.).unwrap());
    }
    #[test]
    fn prevents_path_or_tensor_injection() {
        let p = json!({"targets":["/tmp/weights"],"operation":"scale","strength":0.01,"rank":1,"seed":1,"rationale":"test"});
        assert!(validate_proposal(
            &p,
            &json!([{ "name":"model.layers.0.mlp.down_proj.weight"}])
        )
        .is_err());
    }
    #[test]
    fn reference_free_search_uses_scores_without_a_fabricated_loss_tie() {
        let baseline = json!({"validation":{"semantic":0.5},"preserve":{"semantic":1.0}});
        let better = json!({"validation":{"semantic":0.6},"preserve":{"semantic":1.0}});
        assert!(improves(&better, &baseline, &baseline, "semantic", 0.0).unwrap());
        assert!(!improves(&baseline, &baseline, &baseline, "semantic", 0.0).unwrap());
        let missing = json!({"validation":{},"preserve":{"semantic":1.0}});
        assert!(improves(&missing, &baseline, &baseline, "semantic", 0.0).is_err());
        let mut regression = better;
        regression["preserve"]["semantic"] = json!(0.9);
        assert!(!improves(&regression, &baseline, &baseline, "semantic", 0.0).unwrap());
    }
    #[test]
    fn measured_reference_losses_remain_required_for_preservation() {
        let baseline = json!({"validation":{"semantic":0.5,"loss":2.0},"preserve":{"semantic":1.0,"loss":2.0}});
        let missing = json!({"validation":{"semantic":0.6},"preserve":{"semantic":1.0}});
        assert!(improves(&missing, &baseline, &baseline, "semantic", 0.0).is_err());
        let mut nonfinite_shape = baseline.clone();
        nonfinite_shape["validation"]["loss"] = json!("not measured");
        assert!(improves(&nonfinite_shape, &baseline, &baseline, "semantic", 0.0).is_err());
    }
}
