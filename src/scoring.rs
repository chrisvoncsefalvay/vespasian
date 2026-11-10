use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

pub type Cache = HashMap<String, (f64, String)>;
pub fn cache_key(row: &Value) -> String {
    json!([row["prompt"], row["target"], row["prediction"]]).to_string()
}
pub fn validate(response: &Value, expected: usize) -> Result<Vec<(f64, String)>> {
    let rows = response["scores"]
        .as_array()
        .context("Judge omitted its scores")?;
    if rows.len() != expected {
        bail!(
            "Judge returned {} scores for {expected} examples",
            rows.len()
        )
    }
    let mut output = vec![None; expected];
    for row in rows {
        let id = row["id"]
            .as_u64()
            .context("Judge returned an invalid example ID")? as usize;
        let score = row["score"]
            .as_f64()
            .context("Judge omitted an example score")?;
        if id >= expected
            || output[id].is_some()
            || !score.is_finite()
            || !(0.0..=1.0).contains(&score)
        {
            bail!("Judge returned duplicate, missing or invalid example scores")
        };
        output[id] = Some((
            score,
            row["reason"]
                .as_str()
                .context("Judge omitted its reason")?
                .into(),
        ));
    }
    output
        .into_iter()
        .map(|v| v.context("Judge omitted an example"))
        .collect()
}
pub fn apply(evaluation: &mut Value, cache: &Cache) -> Result<()> {
    let mut sums: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for row in evaluation["samples"]
        .as_array_mut()
        .context("Missing evaluation samples")?
    {
        let (score, reason) = cache
            .get(&cache_key(row))
            .context("Missing semantic judgement")?;
        row["semantic"] = json!(score);
        row["judgement"] = json!(reason);
        let entry = sums
            .entry(
                row["split"]
                    .as_str()
                    .context("Missing sample split")?
                    .into(),
            )
            .or_default();
        entry.0 += score;
        entry.1 += 1;
    }
    for (split, (sum, n)) in sums {
        evaluation["metrics"][&split]["semantic"] = json!(sum / n as f64);
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_duplicate_and_out_of_bounds() {
        assert!(validate(&json!({"scores":[{"id":0,"score":0.5,"reason":"a"},{"id":0,"score":0.2,"reason":"b"}]}),2).is_err());
        assert!(validate(&json!({"scores":[{"id":0,"score":2.,"reason":"a"}]}), 1).is_err());
    }
    #[test]
    fn orders_by_id() {
        let s=validate(&json!({"scores":[{"id":1,"score":0.1,"reason":"b"},{"id":0,"score":0.9,"reason":"a"}]}),2).unwrap();
        assert_eq!(s[0].0, 0.9);
    }
    #[test]
    fn cached_judgements_follow_exact_outputs() {
        let mut e = json!({"metrics":{"validation":{}},"samples":[{"split":"validation","prompt":"q","target":"a","prediction":"a"}]});
        let mut cache = Cache::new();
        cache.insert(cache_key(&e["samples"][0]), (1., "correct".into()));
        apply(&mut e, &cache).unwrap();
        assert_eq!(e["metrics"]["validation"]["semantic"], 1.);
        e["samples"][0]["prediction"] = json!("changed");
        assert!(apply(&mut e, &cache).is_err());
    }
    #[test]
    fn unpaired_judgements_preserve_only_measured_metric_fields() {
        let mut corpus = json!({"metrics":{"validation":{"loss":2.0,"count":2}},"samples":[
            {"mode":"corpus","split":"validation","prompt":"","target":"First style example","prediction":"Generated passage"},
            {"mode":"corpus","split":"validation","prompt":"","target":"Second style example","prediction":"Generated passage"}
        ]});
        let mut cache = Cache::new();
        cache.insert(
            cache_key(&corpus["samples"][0]),
            (0.8, "Style retained".into()),
        );
        cache.insert(
            cache_key(&corpus["samples"][1]),
            (0.4, "Tone differs".into()),
        );
        assert_eq!(cache.len(), 2);
        apply(&mut corpus, &cache).unwrap();
        assert!(
            (corpus["metrics"]["validation"]["semantic"]
                .as_f64()
                .unwrap()
                - 0.6)
                .abs()
                < 1e-9
        );
        assert_eq!(corpus["metrics"]["validation"]["loss"], 2.0);
        assert!(corpus["metrics"]["validation"].get("exact_match").is_none());
        let mut prompts = json!({"metrics":{"validation":{"count":1}},"samples":[{"mode":"prompts","split":"validation","prompt":"Write a greeting","target":"","prediction":"Hello"}]});
        cache.insert(
            cache_key(&prompts["samples"][0]),
            (1.0, "Satisfies rubric".into()),
        );
        apply(&mut prompts, &cache).unwrap();
        assert_eq!(prompts["metrics"]["validation"]["semantic"], 1.0);
        assert!(prompts["metrics"]["validation"].get("loss").is_none());
        assert!(prompts["metrics"]["validation"].get("token_f1").is_none());
    }
}
