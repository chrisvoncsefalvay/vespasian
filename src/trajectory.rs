//! Optional, append-only records of observable editor exchanges and measured turns.
use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    process::Stdio,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const MAX_RECORD_BYTES: usize = 16 * 1024 * 1024;
const MAX_FILE_BYTES: u64 = 512 * 1024 * 1024;
const KINDS: &[&str] = &[
    "editor_request",
    "editor_response",
    "editor_error",
    "baseline",
    "turn",
    "health",
    "event",
];
const METADATA: &[&str] = &[
    "run_id",
    "version",
    "model",
    "model_requested",
    "revision",
    "model_revision",
    "goal",
    "controls",
    "editor",
    "editor_model",
    "reasoning_effort",
    "budget_usd",
    "seed",
    "metric",
    "dataset_sha256",
    "dataset_mode",
    "target_type",
    "health_checks",
    "health_manifest",
    "health_max_drop",
    "max_turns",
    "budget_mode",
    "budget_policy",
];

fn credential_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase().replace('-', "_");
    matches!(
        key.as_str(),
        "hf_token"
            | "token"
            | "api_key"
            | "authorization"
            | "password"
            | "secret"
            | "access_token"
            | "refresh_token"
            | "headers"
            | "environment"
            | "env"
            | "hf_token_path"
            | "huggingface_token"
            | "hugging_face_hub_token"
            | "huggingfacehub_api_token"
    ) || key.ends_with("_api_key")
        || key.ends_with("_access_token")
}

fn absolute_path(value: &Value) -> bool {
    value.as_str().is_some_and(|value| {
        value.starts_with('/')
            || value.starts_with("~/")
            || value.starts_with("\\\\")
            || (value.len() >= 3
                && value.as_bytes()[0].is_ascii_alphabetic()
                && value.as_bytes()[1] == b':'
                && matches!(value.as_bytes()[2], b'/' | b'\\'))
    })
}

fn private_path_key(key: &str) -> bool {
    matches!(
        key,
        "path"
            | "output"
            | "checkpoint"
            | "source_path"
            | "dataset_source"
            | "python"
            | "file"
            | "cwd"
            | "working_directory"
            | "model"
            | "model_requested"
    ) || key.ends_with("_path")
}

fn scrub(value: &Value, secrets: &[String]) -> Value {
    match value {
        Value::Object(values) => Value::Object(
            values
                .iter()
                .filter(|(key, value)| {
                    !(credential_key(key) || private_path_key(key) && absolute_path(value))
                })
                .map(|(key, value)| (key.clone(), scrub(value, secrets)))
                .collect(),
        ),
        Value::Array(values) => {
            Value::Array(values.iter().map(|value| scrub(value, secrets)).collect())
        }
        Value::String(value) => {
            let mut value = value.clone();
            for secret in secrets {
                value = value.replace(secret, "[redacted]");
            }
            Value::String(value)
        }
        _ => value.clone(),
    }
}

fn now_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("The system clock is before the Unix epoch")?
        .as_millis()
        .try_into()?)
}

/// Every line has schema, version, sequence, timestamp_unix_ms, type and payload.
/// Creation writes run_start. A run without run_end is an incomplete trajectory.
/// The recorder never includes raw provider process logs or environment dumps.
pub struct Recorder {
    file: File,
    sequence: u64,
    bytes: u64,
    last_timestamp: u64,
    secrets: Vec<String>,
    finished: bool,
}

impl Recorder {
    pub fn new(path: &Path, metadata: &Value, mut secrets: Vec<String>) -> Result<Self> {
        let source = metadata
            .as_object()
            .context("Trajectory metadata must be an object")?;
        let mut values: Map<String, Value> = source
            .iter()
            .filter(|(key, _)| METADATA.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let started = now_ms()?;
        if !values.get("run_id").is_some_and(|value| {
            value.as_str().is_some_and(|value| {
                !value.is_empty()
                    && value.len() <= 128
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
            })
        }) {
            let identity = format!("{}:{started}:{}", path.display(), std::process::id());
            values.insert(
                "run_id".into(),
                json!(format!("{:x}", Sha256::digest(identity.as_bytes()))[..32]),
            );
        }
        if let Some(parent) = path.parent().filter(|path| path.is_absolute()) {
            secrets.push(parent.to_string_lossy().into_owned());
        }
        for key in ["HOME", "USERPROFILE"] {
            if let Ok(home) = std::env::var(key) {
                if home.len() > 1 && Path::new(&home).is_absolute() {
                    secrets.push(home);
                }
            }
        }
        secrets.retain(|secret| !secret.is_empty());
        secrets.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
        secrets.dedup();
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options
            .open(path)
            .context("Could not create the editor trajectory file")?;
        let mut recorder = Self {
            file,
            sequence: 0,
            bytes: 0,
            last_timestamp: started,
            secrets,
            finished: false,
        };
        recorder.write("run_start", &Value::Object(values))?;
        Ok(recorder)
    }

    fn write(&mut self, kind: &str, payload: &Value) -> Result<()> {
        if self.finished {
            bail!("The editor trajectory has already ended")
        }
        if !payload.is_object() {
            bail!("A trajectory payload must be an object")
        }
        let timestamp = now_ms()?.max(self.last_timestamp);
        let record = json!({
            "schema":"vespasian.trajectory", "version":1, "sequence":self.sequence,
            "timestamp_unix_ms":timestamp, "type":kind, "payload":scrub(payload, &self.secrets)
        });
        let mut bytes = serde_json::to_vec(&record)?;
        if bytes.len() > MAX_RECORD_BYTES || self.bytes + bytes.len() as u64 + 1 > MAX_FILE_BYTES {
            bail!("The editor trajectory reached its file-size limit")
        }
        bytes.push(b'\n');
        self.file
            .write_all(&bytes)
            .context("Could not append to the editor trajectory")?;
        self.file.flush()?;
        self.sequence += 1;
        self.bytes += bytes.len() as u64;
        self.last_timestamp = timestamp;
        Ok(())
    }

    pub fn record(&mut self, kind: &str, payload: &Value) -> Result<()> {
        if !KINDS.contains(&kind) {
            bail!("Unknown editor trajectory record type")
        }
        self.write(kind, payload)
    }

    pub fn finish(&mut self, status: &str, payload: &Value) -> Result<()> {
        if !["complete", "failed", "cancelled"].contains(&status) {
            bail!("Unknown editor trajectory completion status")
        }
        self.write("run_end", &json!({"status":status,"result":payload}))?;
        self.file.sync_data()?;
        self.finished = true;
        Ok(())
    }
}

pub async fn upload(
    python: &Path,
    run: &Path,
    repo: &str,
    visibility: &str,
    create: bool,
    token: Option<&str>,
) -> Result<Value> {
    let mut command = tokio::process::Command::new(python);
    command
        .args(["-u", "-c", include_str!("../python/trajectory.py"), "--run"])
        .arg(run)
        .args(["--repo", repo, "--visibility", visibility])
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        .stdin(Stdio::null())
        .kill_on_drop(true);
    if create {
        command.arg("--create");
    }
    if let Some(token) = token.filter(|token| !token.trim().is_empty()) {
        command.env("HF_TOKEN", token);
    }
    let result = tokio::time::timeout(Duration::from_secs(1800), command.output())
        .await
        .context("Trajectory upload timed out. The local file is saved")?
        .context("Could not start the trajectory upload")?;
    if !result.status.success() {
        let diagnostics = String::from_utf8_lossy(&result.stderr);
        let message = diagnostics
            .lines()
            .rev()
            .find_map(|line| line.strip_prefix("Vespasian: "))
            .unwrap_or("Trajectory upload failed. The local file is saved");
        let message = match token.filter(|token| !token.is_empty()) {
            Some(token) => message.replace(token, "[redacted]"),
            None => message.into(),
        };
        bail!("{message}")
    }
    let receipt: Value = serde_json::from_slice(&result.stdout)
        .context("The trajectory upload returned an invalid receipt")?;
    let commit = receipt["commit"].as_str().unwrap_or_default();
    if receipt["version"] != 1
        || receipt["repo"] != repo
        || receipt["private"] != (visibility == "private")
        || ![40, 64].contains(&commit.len())
        || !commit.bytes().all(|byte| byte.is_ascii_hexdigit())
        || !receipt["files"].is_array()
        || receipt["verified"] != true
        || receipt["repo_url"] != format!("https://huggingface.co/datasets/{repo}")
    {
        bail!("The trajectory receipt does not match the requested repository")
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records(path: &Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn run_start_preserves_reasoning_and_optional_turn_cap() {
        for max_turns in [Value::Null, json!(12)] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("trajectory.jsonl");
            let _recorder = Recorder::new(
                &path,
                &json!({"reasoning_effort":"medium","max_turns":max_turns,"budget_usd":10.0}),
                vec![],
            )
            .unwrap();
            let rows = records(&path);
            assert_eq!(rows[0]["payload"]["reasoning_effort"], "medium");
            assert!(rows[0]["payload"]
                .as_object()
                .unwrap()
                .contains_key("max_turns"));
            assert_eq!(rows[0]["payload"]["max_turns"], max_turns);
        }
    }

    #[test]
    fn records_are_ordered_and_flushed_before_completion() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trajectory.jsonl");
        let mut recorder = Recorder::new(
            &path,
            &json!({"goal":"Answer concisely","version":"0.1.2"}),
            vec![],
        )
        .unwrap();
        recorder
            .record(
                "editor_request",
                &json!({"context":{"goal":"Answer concisely"},"system":"Return JSON"}),
            )
            .unwrap();
        recorder
            .record(
                "editor_response",
                &json!({"response":{"rationale":"Shorten the answer"},"spent_usd":0.01}),
            )
            .unwrap();
        assert_eq!(records(&path).len(), 3);
        recorder
            .finish("complete", &json!({"accepted_turns":1}))
            .unwrap();
        let rows = records(&path);
        assert_eq!(rows[0]["type"], "run_start");
        assert_eq!(rows[3]["payload"]["status"], "complete");
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(row["sequence"], index);
            assert_eq!(row["schema"], "vespasian.trajectory");
            assert_eq!(row["version"], 1);
        }
        assert!(recorder.record("turn", &json!({})).is_err());
        assert!(recorder.finish("complete", &json!({})).is_err());
    }

    #[test]
    fn credentials_and_private_paths_are_removed_but_rollouts_remain() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trajectory.jsonl");
        let mut recorder = Recorder::new(&path, &json!({"goal":"Keep the answer","model":"/private/model","hf_token":"secret-one","environment":{"HOME":"private"},"output":"/private/run"}), vec!["secret-one".into(),"secret-two".into()]).unwrap();
        recorder.record("editor_request", &json!({"context":{"prompt":"Copy the phrase","target":"the phrase","token_ids":[1,2]},"api_key":"secret-one","headers":{"Authorization":"secret-two"},"nested":{"error":"Token secret-two failed","path":"/private/run"}})).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        for private in [
            "secret-one",
            "secret-two",
            "/private",
            "api_key",
            "environment",
            "hf_token",
            "Authorization",
        ] {
            assert!(!text.contains(private), "{private} was retained");
        }
        assert!(text.contains("Copy the phrase"));
        assert!(text.contains("token_ids"));
        assert!(text.contains("[redacted]"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn recording_requires_explicit_new_file_and_valid_event_type() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trajectory.jsonl");
        let mut recorder = Recorder::new(&path, &json!({}), vec![]).unwrap();
        assert!(Recorder::new(&path, &json!({}), vec![]).is_err());
        assert!(recorder.record("run_end", &json!({})).is_err());
        assert!(recorder.record("turn", &Value::Null).is_err());
        assert!(recorder.finish("maybe", &json!({})).is_err());
        recorder
            .finish("cancelled", &json!({"reason":"User stopped the run"}))
            .unwrap();
        assert_eq!(records(&path)[1]["payload"]["status"], "cancelled");
    }

    #[test]
    fn dropped_recorder_does_not_invent_a_successful_end() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trajectory.jsonl");
        {
            let mut recorder = Recorder::new(&path, &json!({}), vec![]).unwrap();
            recorder
                .record(
                    "editor_error",
                    &json!({"error":"Request failed","reserved_usd":0.1}),
                )
                .unwrap();
        }
        let rows = records(&path);
        assert_eq!(rows.len(), 2);
        assert!(!rows.iter().any(|row| row["type"] == "run_end"));
    }

    #[tokio::test]
    async fn missing_trajectory_is_rejected_without_hub_access() {
        let dir = tempfile::tempdir().unwrap();
        let error = upload(
            &crate::platform::test_python(),
            dir.path(),
            "owner/trajectories",
            "private",
            true,
            None,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("trajectory.jsonl"));
    }
}
