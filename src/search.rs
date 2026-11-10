use anyhow::{bail, Result};
use serde_json::Value;
use std::collections::HashSet;

pub fn validate_proposal(p: &Value, editable: &Value) -> Result<()> {
    let keys: [&str; 7] = [
        "targets",
        "operation",
        "strength",
        "rank",
        "seed",
        "rationale",
        "",
    ];
    let obj = p
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Editor proposal must be an object"))?;
    if obj.len() != 6 || obj.keys().any(|k| !keys[..6].contains(&k.as_str())) {
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
    if !["scale", "low_rank", "noise", "activation_rank_one"]
        .contains(&p["operation"].as_str().unwrap_or(""))
    {
        bail!("Unknown tensor operation")
    }
    let strength = p["strength"].as_f64().unwrap_or(f64::NAN);
    if !strength.is_finite() || !(1e-6..=0.1).contains(&strength.abs()) {
        bail!("Mutation strength must have magnitude between 0.000001 and 0.1")
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
    fn m(score: f64, loss: f64, p: f64) -> Value {
        json!({"validation":{"exact_match":score,"loss":loss},"preserve":{"exact_match":p,"loss":1.0}})
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
