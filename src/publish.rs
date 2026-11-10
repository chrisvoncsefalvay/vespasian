use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::{path::Path, process::Stdio, time::Duration};

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
        .args(["-u", "-c", include_str!("../python/publish.py"), "--run"])
        .arg(run)
        .args(["--repo", repo, "--visibility", visibility])
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        .stdin(Stdio::null())
        .kill_on_drop(true);
    if create {
        command.arg("--create");
    }
    if let Some(token) = token {
        command.env("HF_TOKEN", token);
    }
    let result = tokio::time::timeout(Duration::from_secs(1800), command.output())
        .await
        .context("Hugging Face upload timed out. The local checkpoint is saved")?
        .context("Could not start the Hugging Face upload")?;
    if !result.status.success() {
        let diagnostics = String::from_utf8_lossy(&result.stderr);
        let message = diagnostics
            .lines()
            .rev()
            .find_map(|line| line.strip_prefix("Vespasian: "))
            .unwrap_or("Hugging Face upload failed. The local checkpoint is saved");
        let message = match token.filter(|token| !token.is_empty()) {
            Some(token) => message.replace(token, "[redacted]"),
            None => message.into(),
        };
        bail!("{message}")
    }
    let receipt: Value =
        serde_json::from_slice(&result.stdout).context("The upload returned an invalid receipt")?;
    let commit = receipt["commit"].as_str().unwrap_or_default();
    if receipt["repo"] != repo
        || receipt["private"] != (visibility == "private")
        || ![40, 64].contains(&commit.len())
        || !commit.bytes().all(|byte| byte.is_ascii_hexdigit())
        || !receipt["files"].is_array()
        || receipt["repo_url"] != format!("https://huggingface.co/{repo}")
    {
        bail!("The upload receipt does not match the requested repository")
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn incomplete_run_is_rejected_before_hub_access() {
        let directory = tempfile::tempdir().unwrap();
        let error = upload(
            &crate::platform::test_python(),
            directory.path(),
            "owner/checkpoint",
            "private",
            true,
            Some("hf_test_session_secret"),
        )
        .await
        .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("report.json"));
        assert!(!message.contains("hf_test_session_secret"));
    }
}
