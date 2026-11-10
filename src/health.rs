use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, path::Path, process::Stdio, time::Duration};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Selection {
    pub check: String,
    pub examples: u32,
}

#[derive(Debug, Clone)]
pub struct Definition {
    pub id: &'static str,
    pub label: &'static str,
    pub group: &'static str,
}

pub fn catalogue() -> Vec<Definition> {
    vec![
        Definition {
            id: "copying",
            label: "Copying",
            group: "Core",
        },
        Definition {
            id: "arithmetic",
            label: "Arithmetic",
            group: "Core",
        },
        Definition {
            id: "json",
            label: "JSON output",
            group: "Core",
        },
        Definition {
            id: "mmlu:abstract_algebra",
            label: "MMLU · abstract algebra",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:anatomy",
            label: "MMLU · anatomy",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:astronomy",
            label: "MMLU · astronomy",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:business_ethics",
            label: "MMLU · business ethics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:clinical_knowledge",
            label: "MMLU · clinical knowledge",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:college_biology",
            label: "MMLU · college biology",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:college_chemistry",
            label: "MMLU · college chemistry",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:college_computer_science",
            label: "MMLU · college computer science",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:college_mathematics",
            label: "MMLU · college mathematics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:college_medicine",
            label: "MMLU · college medicine",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:college_physics",
            label: "MMLU · college physics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:computer_security",
            label: "MMLU · computer security",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:conceptual_physics",
            label: "MMLU · conceptual physics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:econometrics",
            label: "MMLU · econometrics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:electrical_engineering",
            label: "MMLU · electrical engineering",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:elementary_mathematics",
            label: "MMLU · elementary mathematics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:formal_logic",
            label: "MMLU · formal logic",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:global_facts",
            label: "MMLU · global facts",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_biology",
            label: "MMLU · high school biology",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_chemistry",
            label: "MMLU · high school chemistry",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_computer_science",
            label: "MMLU · high school computer science",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_european_history",
            label: "MMLU · high school european history",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_geography",
            label: "MMLU · high school geography",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_government_and_politics",
            label: "MMLU · high school government and politics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_macroeconomics",
            label: "MMLU · high school macroeconomics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_mathematics",
            label: "MMLU · high school mathematics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_microeconomics",
            label: "MMLU · high school microeconomics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_physics",
            label: "MMLU · high school physics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_psychology",
            label: "MMLU · high school psychology",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_statistics",
            label: "MMLU · high school statistics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_us_history",
            label: "MMLU · high school us history",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:high_school_world_history",
            label: "MMLU · high school world history",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:human_aging",
            label: "MMLU · human aging",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:human_sexuality",
            label: "MMLU · human sexuality",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:international_law",
            label: "MMLU · international law",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:jurisprudence",
            label: "MMLU · jurisprudence",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:logical_fallacies",
            label: "MMLU · logical fallacies",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:machine_learning",
            label: "MMLU · machine learning",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:management",
            label: "MMLU · management",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:marketing",
            label: "MMLU · marketing",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:medical_genetics",
            label: "MMLU · medical genetics",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:miscellaneous",
            label: "MMLU · miscellaneous",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:moral_disputes",
            label: "MMLU · moral disputes",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:moral_scenarios",
            label: "MMLU · moral scenarios",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:nutrition",
            label: "MMLU · nutrition",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:philosophy",
            label: "MMLU · philosophy",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:prehistory",
            label: "MMLU · prehistory",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:professional_accounting",
            label: "MMLU · professional accounting",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:professional_law",
            label: "MMLU · professional law",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:professional_medicine",
            label: "MMLU · professional medicine",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:professional_psychology",
            label: "MMLU · professional psychology",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:public_relations",
            label: "MMLU · public relations",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:security_studies",
            label: "MMLU · security studies",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:sociology",
            label: "MMLU · sociology",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:us_foreign_policy",
            label: "MMLU · us foreign policy",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:virology",
            label: "MMLU · virology",
            group: "MMLU",
        },
        Definition {
            id: "mmlu:world_religions",
            label: "MMLU · world religions",
            group: "MMLU",
        },
    ]
}

fn metric(check: &str) -> Result<&'static str> {
    match check {
        "copying" => Ok("exact_text"),
        "arithmetic" => Ok("numeric_exact"),
        "json" => Ok("json_exact"),
        name if name.starts_with("mmlu:") && catalogue().iter().any(|entry| entry.id == name) => {
            Ok("choice_logprob_accuracy")
        }
        _ => bail!("Unknown health check. Use copying, arithmetic, json or mmlu:subject"),
    }
}

pub fn parse(value: &str) -> Result<Selection> {
    let (check, examples) = value
        .rsplit_once(':')
        .context("Use check:count, for example copying:20 or mmlu:anatomy:20")?;
    if examples.is_empty() || !examples.bytes().all(|ch| ch.is_ascii_digit()) {
        bail!("The health example count must be a whole number")
    }
    let selection = Selection {
        check: check.into(),
        examples: examples
            .parse()
            .context("The health example count is too large")?,
    };
    validate(std::slice::from_ref(&selection), None)?;
    Ok(selection)
}

pub fn validate(selections: &[Selection], max_drop: Option<f64>) -> Result<()> {
    if max_drop.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        bail!("The health regression limit must be between 0 and 1")
    }
    if selections.is_empty() && max_drop.is_some() {
        bail!("Select a health check before setting a regression limit")
    }
    let mut seen = BTreeSet::new();
    for selection in selections {
        metric(&selection.check)?;
        if !(1..=1000).contains(&selection.examples) {
            bail!("Choose between 1 and 1000 examples per health check")
        }
        if !seen.insert(&selection.check) {
            bail!("Each health check can be selected only once")
        }
    }
    Ok(())
}

pub async fn prepare(
    python: &Path,
    output: &Path,
    selections: &[Selection],
    seed: u64,
    token: Option<&str>,
) -> Result<Value> {
    validate(selections, None)?;
    if selections.is_empty() {
        bail!("Select at least one health check")
    }
    if seed > u32::MAX as u64 {
        bail!("The health seed must be between 0 and 4294967295")
    }
    let mut command = tokio::process::Command::new(python);
    command
        .args(["-u", "-c", include_str!("../python/health.py"), "--output"])
        .arg(output.join("health.json"))
        .arg("--seed")
        .arg(seed.to_string())
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        .stdin(Stdio::null())
        .kill_on_drop(true);
    if let Some(token) = token.filter(|token| !token.is_empty()) {
        command.env("HF_TOKEN", token);
    }
    for selection in selections {
        command
            .arg("--selection")
            .arg(format!("{}:{}", selection.check, selection.examples));
    }
    let result = tokio::time::timeout(Duration::from_secs(300), command.output())
        .await
        .context(
            "Health dataset preparation timed out. Check the connection or select local checks",
        )?
        .context("Could not start health dataset preparation")?;
    if !result.status.success() {
        let diagnostics = String::from_utf8_lossy(&result.stderr);
        let message = diagnostics
            .lines()
            .rev()
            .find_map(|line| line.strip_prefix("Vespasian: "))
            .unwrap_or("Could not prepare the health examples");
        let message = match token.filter(|token| !token.is_empty()) {
            Some(token) => message.replace(token, "[redacted]"),
            None => message.into(),
        };
        bail!("{message}")
    }
    let summary: Value = serde_json::from_slice(&result.stdout)
        .context("Health preparation returned an invalid receipt")?;
    let checks = summary["checks"]
        .as_array()
        .context("The health receipt has no checks")?;
    if summary["schema_version"] != 1
        || checks.len() != selections.len()
        || summary["count"].as_u64()
            != Some(selections.iter().map(|entry| entry.examples as u64).sum())
        || summary["seed"] != seed
        || checks.iter().zip(selections).any(|(check, selected)| {
            check["check"] != selected.check || check["count"] != selected.examples
        })
    {
        bail!("The prepared health checks do not match the selected examples")
    }
    Ok(summary)
}

fn checked_metrics(value: &Value) -> Result<&serde_json::Map<String, Value>> {
    let map = value
        .as_object()
        .context("Health measurements must be an object")?;
    if map.is_empty() {
        bail!("The enabled health checks returned no measurements")
    }
    for (check, row) in map {
        let expected = metric(check)?;
        let score = row["score"]
            .as_f64()
            .context("A health measurement has no score")?;
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            bail!("A health score is outside 0 to 1")
        }
        let count = row["count"]
            .as_u64()
            .context("A health measurement has no example count")?;
        if !(1..=1000).contains(&count)
            || row["metric"] != expected
            || row["label"].as_str().is_none_or(str::is_empty)
        {
            bail!("A health measurement has an invalid definition")
        }
    }
    Ok(map)
}

pub fn passes(baseline: &Value, candidate: &Value, max_drop: Option<f64>) -> Result<bool> {
    if max_drop.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        bail!("The health regression limit must be between 0 and 1")
    }
    let before = checked_metrics(baseline)?;
    let after = checked_metrics(candidate)?;
    if before.keys().ne(after.keys()) {
        bail!("Candidate health checks differ from the baseline")
    }
    let mut retained = true;
    for (check, old) in before {
        let new = &after[check];
        if old["count"] != new["count"]
            || old["metric"] != new["metric"]
            || old["label"] != new["label"]
        {
            bail!("Candidate health examples differ from the baseline")
        }
        if let Some(limit) = max_drop {
            retained &=
                old["score"].as_f64().unwrap() - new["score"].as_f64().unwrap() <= limit + 1e-12;
        }
    }
    Ok(retained)
}

fn verify_samples(evaluation: &Value) -> Result<()> {
    let metrics = checked_metrics(&evaluation["metrics"])?;
    let samples = evaluation["samples"]
        .as_array()
        .context("Missing health samples")?;
    let expected: u64 = metrics
        .values()
        .map(|row| row["count"].as_u64().unwrap())
        .sum();
    if expected != samples.len() as u64 {
        bail!("Health sample count does not match its measurements")
    }
    let mut ids = BTreeSet::new();
    for sample in samples {
        let id = sample["id"]
            .as_str()
            .context("A health rollout has no example ID")?;
        let check = sample["check"]
            .as_str()
            .context("A health rollout has no check")?;
        if !ids.insert(id)
            || !metrics.contains_key(check)
            || !sample["passed"].is_boolean()
            || ["prompt", "target", "prediction", "finish_reason"]
                .iter()
                .any(|key| !sample[key].is_string())
        {
            bail!("A health rollout has missing or duplicated fields")
        }
    }
    for (check, metric) in metrics {
        if samples.iter().filter(|row| row["check"] == *check).count() as u64
            != metric["count"].as_u64().unwrap()
        {
            bail!("Health samples do not match their check counts")
        }
    }
    Ok(())
}

pub fn same_rollouts(before: &Value, after: &Value) -> Result<bool> {
    verify_samples(before)?;
    verify_samples(after)?;
    passes(&before["metrics"], &after["metrics"], None)?;
    let digest = before["rows_sha256"]
        .as_str()
        .context("Health rollouts have no example hash")?;
    if digest.len() != 64
        || !digest.bytes().all(|ch| ch.is_ascii_hexdigit())
        || before["rows_sha256"] != after["rows_sha256"]
    {
        return Ok(false);
    }
    let old = before["samples"]
        .as_array()
        .context("Missing original health samples")?;
    let new = after["samples"]
        .as_array()
        .context("Missing reloaded health samples")?;
    if old.len() != new.len() {
        return Ok(false);
    }
    let mut ids = BTreeSet::new();
    for left in old {
        let id = left["id"]
            .as_str()
            .context("A health rollout has no example ID")?;
        if !ids.insert(id) {
            bail!("Health rollout IDs must be unique")
        }
        let matches: Vec<_> = new.iter().filter(|right| right["id"] == id).collect();
        if matches.len() != 1 {
            return Ok(false);
        }
        let right = matches[0];
        for key in [
            "check",
            "prompt",
            "target",
            "prediction",
            "passed",
            "token_ids",
            "choice_token_ids",
            "finish_reason",
        ] {
            if left[key] != right[key] {
                return Ok(false);
            }
        }
        if !left["choice_logprobs"].is_null() || !right["choice_logprobs"].is_null() {
            let l = left["choice_logprobs"]
                .as_object()
                .context("Missing health choice likelihoods")?;
            let r = right["choice_logprobs"]
                .as_object()
                .context("Missing reloaded choice likelihoods")?;
            if l.keys().ne(r.keys()) {
                return Ok(false);
            }
            for (choice, score) in l {
                let x = score.as_f64().context("Invalid health choice likelihood")?;
                let y = r[choice]
                    .as_f64()
                    .context("Invalid reloaded choice likelihood")?;
                if !x.is_finite() || !y.is_finite() || (x - y).abs() > 1e-5 {
                    return Ok(false);
                }
            }
        }
    }
    for (check, score) in before["metrics"].as_object().unwrap() {
        if (score["score"].as_f64().unwrap() - after["metrics"][check]["score"].as_f64().unwrap())
            .abs()
            > 1e-12
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn scores(value: f64) -> Value {
        json!({"copying":{"label":"Copying","metric":"exact_text","score":value,"count":20}})
    }

    #[test]
    fn selections_parse_subjects_and_validate_limits() {
        assert_eq!(
            parse("mmlu:anatomy:12").unwrap(),
            Selection {
                check: "mmlu:anatomy".into(),
                examples: 12
            }
        );
        assert_eq!(catalogue().len(), 60);
        for value in [
            "copying",
            "copying:0",
            "copying:1001",
            "copying:+2",
            "mmlu:not_a_subject:3",
        ] {
            assert!(parse(value).is_err(), "{value}");
        }
        let selected = parse("copying:2").unwrap();
        assert!(validate(&[selected.clone(), selected], None).is_err());
        assert!(validate(&[], Some(0.1)).is_err());
        assert!(validate(&[], None).is_ok());
    }

    #[test]
    fn monitoring_accepts_drop_and_explicit_limits_reject_it() {
        let baseline = scores(0.8);
        let candidate = scores(0.6);
        assert!(passes(&baseline, &candidate, None).unwrap());
        assert!(!passes(&baseline, &candidate, Some(0.1)).unwrap());
        assert!(passes(&baseline, &candidate, Some(0.2)).unwrap());
        assert!(passes(&baseline, &baseline, Some(0.0)).unwrap());
        assert!(passes(&baseline, &scores(0.9), Some(0.0)).unwrap());
    }

    #[test]
    fn missing_or_changed_measurements_never_pass_silently() {
        let baseline = scores(0.8);
        assert!(passes(&baseline, &json!({}), None).is_err());
        let mut candidate = scores(0.9);
        candidate["copying"]["count"] = json!(19);
        assert!(passes(&baseline, &candidate, None).is_err());
        candidate = scores(1.1);
        assert!(passes(&baseline, &candidate, None).is_err());
        assert!(passes(&baseline, &baseline, Some(f64::NAN)).is_err());
        candidate = scores(0.9);
        candidate["copying"]["metric"] = json!("semantic");
        assert!(passes(&baseline, &candidate, None).is_err());
    }

    #[test]
    fn reload_verifies_answers_tokens_and_choice_likelihoods() {
        let before = json!({"metrics":{"copying":{"label":"Copying","metric":"exact_text","score":1.0,"count":1}}, "rows_sha256":"a".repeat(64), "samples":[{"id":"copying:0000","check":"copying","prompt":"Copy A","target":"A","prediction":"A","passed":true,"token_ids":[3,2],"finish_reason":"stop"}]});
        assert!(same_rollouts(&before, &before).unwrap());
        let mut after = before.clone();
        after["samples"][0]["token_ids"] = json!([4, 2]);
        assert!(!same_rollouts(&before, &after).unwrap());
        after = before.clone();
        after["rows_sha256"] = json!("b".repeat(64));
        assert!(!same_rollouts(&before, &after).unwrap());
        let mut choice = before.clone();
        choice["samples"][0]["choice_logprobs"] = json!({"A":-2.0,"B":-3.0});
        after = choice.clone();
        after["samples"][0]["choice_logprobs"]["A"] = json!(-2.000001);
        assert!(same_rollouts(&choice, &after).unwrap());
        after["samples"][0]["choice_logprobs"]["A"] = json!(-2.1);
        assert!(!same_rollouts(&choice, &after).unwrap());
    }

    #[tokio::test]
    async fn local_preparation_needs_no_hugging_face_runtime_or_token() {
        let directory = tempfile::tempdir().unwrap();
        let summary = prepare(
            &crate::platform::test_python(),
            directory.path(),
            &[parse("copying:3").unwrap(), parse("json:2").unwrap()],
            42,
            None,
        )
        .await
        .unwrap();
        assert_eq!(summary["count"], 5);
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(directory.path().join("health.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["rows"].as_array().unwrap().len(), 5);
        assert_eq!(summary["rows_sha256"], manifest["rows_sha256"]);
        assert!(prepare(
            &crate::platform::test_python(),
            directory.path(),
            &[parse("copying:3").unwrap()],
            42,
            None
        )
        .await
        .is_err());
    }
}
