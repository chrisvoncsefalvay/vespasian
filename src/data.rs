use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub prompt: String,
    pub target: String,
    pub split: String,
    #[serde(default = "paired", skip_serializing_if = "is_paired")]
    pub mode: String,
}
#[derive(Debug)]
pub struct Dataset {
    pub rows: Vec<Example>,
    pub sha256: String,
    pub counts: BTreeMap<String, usize>,
    pub mode: String,
}

pub fn normalise(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}
fn paired() -> String {
    "paired".into()
}
fn is_paired(mode: &String) -> bool {
    mode == "paired"
}

fn field<'a>(row: &'a Value, selector: &str) -> Option<&'a Value> {
    if let Some(value) = row.get(selector) {
        return Some(value);
    }
    let mut value = row;
    for component in selector.split('.') {
        value = match value {
            Value::Array(values) => values.get(component.parse::<usize>().ok()?)?,
            _ => value.get(component)?,
        };
    }
    Some(value)
}

pub fn load(path: &Path) -> Result<Dataset> {
    load_with_mode(path, "paired", None, None)
}

pub fn load_with_mode(
    path: &Path,
    mode: &str,
    text_field: Option<&str>,
    prompt_field: Option<&str>,
) -> Result<Dataset> {
    if !["paired", "corpus", "prompts"].contains(&mode) {
        bail!("Dataset mode must be paired, corpus or prompts");
    }
    let raw = fs::read(path).with_context(|| format!("Cannot read {}", path.display()))?;
    if raw.len() > 32 * 1024 * 1024 {
        bail!("Dataset exceeds 32 MiB. Use a focused editing dataset");
    }
    let text = std::str::from_utf8(&raw).context("Dataset must be UTF-8 JSONL")?;
    let mut rows = Vec::new();
    let mut identities = HashSet::new();
    let mut dataset_mode: Option<String> = None;
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line)
            .with_context(|| format!("Invalid JSON at line {}", n + 1))?;
        let row_mode = match v.get("mode") {
            None => mode,
            Some(Value::String(value))
                if ["paired", "corpus", "prompts"].contains(&value.as_str()) =>
            {
                value
            }
            _ => bail!("Invalid dataset mode at line {}", n + 1),
        };
        if mode != "paired" && row_mode != mode {
            bail!(
                "Line {} has a different mode from the selected dataset mode",
                n + 1
            );
        }
        if dataset_mode
            .as_deref()
            .is_some_and(|selected| selected != row_mode)
        {
            bail!("Use one dataset mode throughout the file");
        }
        dataset_mode = Some(row_mode.into());
        let (prompt, target) = match row_mode {
            "corpus" => {
                let selector = text_field.unwrap_or("text");
                let value = if v.get("mode").is_some() {
                    v.get("target").or_else(|| field(&v, selector))
                } else {
                    field(&v, selector)
                };
                let target = value
                    .and_then(Value::as_str)
                    .with_context(|| format!("Line {} needs a text field {selector:?}", n + 1))?;
                if v.get("mode").is_some()
                    && v.get("prompt")
                        .and_then(Value::as_str)
                        .is_some_and(|text| !text.is_empty())
                {
                    bail!("Corpus rows must have an empty prompt at line {}", n + 1);
                }
                ("", target)
            }
            "prompts" => {
                let selector = prompt_field.unwrap_or("prompt");
                let prompt = field(&v, selector)
                    .and_then(Value::as_str)
                    .with_context(|| format!("Line {} needs a prompt field {selector:?}", n + 1))?;
                if v.get("mode").is_some()
                    && v.get("target")
                        .and_then(Value::as_str)
                        .is_some_and(|text| !text.is_empty())
                {
                    bail!(
                        "Prompt-only rows must have an empty target at line {}",
                        n + 1
                    );
                }
                (prompt, "")
            }
            _ => {
                let prompt = v.get("prompt").or(v.get("input")).and_then(Value::as_str);
                let target = v
                    .get("target")
                    .or(v.get("output"))
                    .or(v.get("response"))
                    .and_then(Value::as_str);
                let (Some(prompt), Some(target)) = (prompt, target) else {
                    bail!(
                        "Line {} needs text fields prompt and target (or input and output)",
                        n + 1
                    );
                };
                (prompt, target)
            }
        };
        if (row_mode != "corpus" && prompt.trim().is_empty())
            || (row_mode != "prompts" && target.trim().is_empty())
        {
            bail!("Empty example text at line {}", n + 1);
        }
        let identity = if row_mode == "corpus" { target } else { prompt };
        if !identities.insert(normalise(identity)) {
            bail!(
                "Duplicate {} at line {}. Each example must belong to one split",
                if row_mode == "corpus" {
                    "text"
                } else {
                    "prompt"
                },
                n + 1
            );
        }
        let split = match v.get("split") {
            None => "",
            Some(Value::String(s)) => s,
            _ => bail!("Invalid split at line {}", n + 1),
        };
        if !["", "edit", "validation", "preserve", "test"].contains(&split) {
            bail!("Unknown split at line {}: {}", n + 1, split);
        }
        rows.push(Example {
            prompt: prompt.into(),
            target: target.into(),
            split: split.into(),
            mode: row_mode.into(),
        });
    }
    if rows.iter().any(|r| r.split.is_empty()) {
        if rows.iter().any(|r| !r.split.is_empty()) {
            bail!("Provide split on every row, or omit it on every row");
        }
        if rows.len() < 5 {
            bail!("At least five rows are required for an automatic edit/validation split");
        }
        // Hash ordering makes the split stable when input lines are reordered.
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| {
            format!(
                "{:x}",
                Sha256::digest(if rows[i].mode == "corpus" {
                    rows[i].target.as_bytes()
                } else {
                    rows[i].prompt.as_bytes()
                })
            )
        });
        let n_validation = (rows.len() / 5).max(1);
        for (j, i) in order.into_iter().enumerate() {
            rows[i].split = if j < n_validation {
                "validation"
            } else {
                "edit"
            }
            .into();
        }
    }
    let mut counts = BTreeMap::new();
    for row in &rows {
        *counts.entry(row.split.clone()).or_insert(0) += 1;
    }
    if counts.get("edit").copied().unwrap_or(0) == 0
        || counts.get("validation").copied().unwrap_or(0) == 0
    {
        bail!("Dataset needs both edit and validation examples");
    }
    Ok(Dataset {
        rows,
        sha256: format!("{:x}", Sha256::digest(&raw)),
        counts,
        mode: dataset_mode.unwrap_or_else(paired),
    })
}

pub fn write(path: &Path, rows: &[Example]) -> Result<()> {
    let mut s = String::new();
    for row in rows {
        s.push_str(&serde_json::to_string(row)?);
        s.push('\n');
    }
    fs::write(path, s)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn file(s: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(s.as_bytes()).unwrap();
        f
    }
    #[test]
    fn prevents_split_leakage() {
        let f=file("{\"prompt\":\"A?\",\"target\":\"a\",\"split\":\"edit\"}\n{\"prompt\":\" a? \",\"target\":\"b\",\"split\":\"validation\"}");
        assert!(load(f.path())
            .unwrap_err()
            .to_string()
            .contains("Duplicate"));
    }
    #[test]
    fn rejects_mixed_splits() {
        let f=file("{\"prompt\":\"a\",\"target\":\"x\",\"split\":\"edit\"}\n{\"prompt\":\"b\",\"target\":\"y\"}");
        assert!(load(f.path()).is_err());
    }
    #[test]
    fn stable_auto_split() {
        let lines: Vec<String> = (0..10)
            .map(|i| format!("{{\"input\":\"q{i}\",\"output\":\"a{i}\"}}"))
            .collect();
        let a = file(&lines.join("\n"));
        let b = file(&lines.iter().rev().cloned().collect::<Vec<_>>().join("\n"));
        let map = |d: Dataset| {
            d.rows
                .into_iter()
                .map(|r| (r.prompt, r.split))
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(map(load(a.path()).unwrap()), map(load(b.path()).unwrap()));
    }
    #[test]
    fn paired_serialization_preserves_existing_row_shape() {
        let input = file("{\"prompt\":\"a\",\"target\":\"x\",\"split\":\"edit\"}\n{\"prompt\":\"b\",\"target\":\"y\",\"split\":\"validation\"}");
        let data = load(input.path()).unwrap();
        assert_eq!(data.mode, "paired");
        assert!(serde_json::to_value(&data.rows[0])
            .unwrap()
            .get("mode")
            .is_none());
    }
    #[test]
    fn corpus_text_defines_identity_and_stable_split_without_prompts() {
        let lines: Vec<_> = (0..10)
            .map(|i| format!("{{\"body\":{{\"text\":\"Reference passage {i}\"}}}}"))
            .collect();
        let a = file(&lines.join("\n"));
        let b = file(&lines.iter().rev().cloned().collect::<Vec<_>>().join("\n"));
        let left = load_with_mode(a.path(), "corpus", Some("body.text"), None).unwrap();
        let right = load_with_mode(b.path(), "corpus", Some("body.text"), None).unwrap();
        assert_eq!(left.mode, "corpus");
        assert!(left
            .rows
            .iter()
            .all(|row| row.prompt.is_empty() && row.mode == "corpus"));
        let map = |data: Dataset| {
            data.rows
                .into_iter()
                .map(|row| (row.target, row.split))
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(map(left), map(right));
        let duplicate = file("{\"text\":\"Same text\",\"split\":\"edit\"}\n{\"text\":\" SAME  text \",\"split\":\"validation\"}");
        assert!(load_with_mode(duplicate.path(), "corpus", None, None)
            .unwrap_err()
            .to_string()
            .contains("Duplicate"));
    }
    #[test]
    fn canonical_modes_roundtrip_and_prompt_only_does_not_require_answers() {
        let input = file("{\"question\":\"a\",\"split\":\"edit\"}\n{\"question\":\"b\",\"split\":\"validation\"}");
        let data = load_with_mode(input.path(), "prompts", None, Some("question")).unwrap();
        assert_eq!(data.mode, "prompts");
        assert!(data.rows.iter().all(|row| row.target.is_empty()));
        let output = tempfile::NamedTempFile::new().unwrap();
        write(output.path(), &data.rows).unwrap();
        let loaded = load(output.path()).unwrap();
        assert_eq!(loaded.mode, "prompts");
        assert_eq!(loaded.counts, data.counts);
        let mixed = file("{\"mode\":\"corpus\",\"prompt\":\"\",\"target\":\"Reference\",\"split\":\"edit\"}\n{\"mode\":\"prompts\",\"prompt\":\"Input\",\"target\":\"\",\"split\":\"validation\"}");
        assert!(load(mixed.path()).is_err());
    }
}
