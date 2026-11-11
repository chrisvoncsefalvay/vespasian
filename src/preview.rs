use anyhow::{Context, Result};
use serde_json::Value;
use std::{
    io::Read,
    path::PathBuf,
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc::{self, Receiver, TryRecvError},
    time::{Duration, Instant},
};

fn dataset_script() -> Result<tempfile::TempDir> {
    // Keep source out of the command line: Windows limits a process command to
    // 32,767 characters. The directory lives as long as the preview job does.
    let directory = tempfile::tempdir().context("Cannot prepare the dataset preview runtime")?;
    std::fs::write(
        directory.path().join("dataset.py"),
        include_str!("../python/dataset.py"),
    )?;
    std::fs::write(
        directory.path().join("task_metadata.py"),
        include_str!("../python/task_metadata.py"),
    )?;
    Ok(directory)
}

const OUTPUT_LIMIT: u64 = 1024 * 1024;
const ERROR_LIMIT: u64 = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreviewRequest {
    pub repo: String,
    pub config: Option<String>,
    pub split: String,
    pub revision: Option<String>,
    pub prompt_field: Option<String>,
    pub target_field: Option<String>,
    pub dataset_mode: String,
    pub hf_task: Option<String>,
    pub model_id: Option<String>,
    pub text_field: Option<String>,
}

pub struct PreviewClient {
    python: PathBuf,
    token: Option<String>,
    job: Option<Job>,
    timeout: Duration,
}

struct Job {
    child: Child,
    script_directory: Option<tempfile::TempDir>,
    output: Receiver<Result<Vec<u8>, String>>,
    errors: Receiver<Result<Vec<u8>, String>>,
    bytes: Option<Vec<u8>>,
    diagnostics: Option<Vec<u8>>,
    status: Option<ExitStatus>,
    started: Instant,
    complete: bool,
}

fn reader(stream: impl Read + Send + 'static, maximum: u64) -> Receiver<Result<Vec<u8>, String>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = match stream.take(maximum + 1).read_to_end(&mut bytes) {
            Ok(_) if bytes.len() as u64 <= maximum => Ok(bytes),
            Ok(_) => Err("The dataset preview exceeded its display limit".into()),
            Err(_) => Err("Could not read the dataset preview".into()),
        };
        let _ = tx.send(result);
    });
    rx
}

impl Job {
    fn stop(&mut self) {
        if self.complete {
            return;
        }
        #[cfg(unix)]
        {
            unsafe extern "C" {
                fn kill(pid: i32, signal: i32) -> i32;
            }
            // Each preview owns its process group; cancellation includes children.
            unsafe {
                kill(-(self.child.id() as i32), 9);
            }
        }
        #[cfg(windows)]
        crate::platform::terminate_tree(self.child.id());
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.complete = true;
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        self.stop();
    }
}

impl PreviewClient {
    pub fn new(python: PathBuf) -> Self {
        Self {
            python,
            token: None,
            job: None,
            timeout: Duration::from_secs(60),
        }
    }

    pub fn set_token(&mut self, token: Option<String>) {
        let token = token
            .map(|token| token.trim().to_string())
            .filter(|token| !token.is_empty());
        if self.token != token {
            self.cancel();
            self.token = token;
        }
    }

    pub fn start(&mut self, request: PreviewRequest) -> Result<()> {
        self.cancel();
        let directory = dataset_script()?;
        let mut command = Command::new(&self.python);
        command
            .arg("-u")
            .arg(directory.path().join("dataset.py"))
            .arg("--preview")
            .arg("--repo")
            .arg(request.repo)
            .arg("--split")
            .arg(request.split)
            .arg("--dataset-mode")
            .arg(request.dataset_mode)
            .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            .env("HF_HUB_ETAG_TIMEOUT", "10")
            .env("HF_HUB_DOWNLOAD_TIMEOUT", "20");
        for (flag, value) in [
            ("--config", request.config),
            ("--revision", request.revision),
            ("--prompt-field", request.prompt_field),
            ("--target-field", request.target_field),
            ("--text-field", request.text_field),
            ("--task", request.hf_task),
            ("--model-id", request.model_id),
        ] {
            if let Some(value) = value.filter(|v| !v.trim().is_empty()) {
                command.arg(flag).arg(value);
            }
        }
        self.spawn(command)?;
        self.job.as_mut().expect("spawned preview").script_directory = Some(directory);
        Ok(())
    }

    fn spawn(&mut self, mut command: Command) -> Result<()> {
        if let Some(token) = &self.token {
            command.env("HF_TOKEN", token);
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command
            .spawn()
            .context("Cannot open the dataset runtime. Run vespasian setup or select --python")?;
        let stdout = child
            .stdout
            .take()
            .context("Dataset preview output is unavailable")?;
        let stderr = child
            .stderr
            .take()
            .context("Dataset preview diagnostics are unavailable")?;
        self.job = Some(Job {
            child,
            script_directory: None,
            output: reader(stdout, OUTPUT_LIMIT),
            errors: reader(stderr, ERROR_LIMIT),
            bytes: None,
            diagnostics: None,
            status: None,
            started: Instant::now(),
            complete: false,
        });
        Ok(())
    }

    pub fn poll(&mut self) -> Option<Result<Value, String>> {
        let token = self.token.as_deref();
        let job = self.job.as_mut()?;
        let outcome = (|| {
            if job.started.elapsed() >= self.timeout {
                return Some(Err(
                    "Dataset preview timed out. Check the connection and try again".into(),
                ));
            }
            if job.bytes.is_none() {
                match job.output.try_recv() {
                    Ok(Ok(bytes)) => job.bytes = Some(bytes),
                    Ok(Err(error)) => return Some(Err(error)),
                    Err(TryRecvError::Disconnected) => {
                        return Some(Err("Dataset preview output stopped unexpectedly".into()))
                    }
                    Err(TryRecvError::Empty) => (),
                }
            }
            if job.diagnostics.is_none() {
                match job.errors.try_recv() {
                    Ok(Ok(bytes)) => job.diagnostics = Some(bytes),
                    Ok(Err(error)) => return Some(Err(error)),
                    Err(TryRecvError::Disconnected) => {
                        return Some(Err(
                            "Dataset preview diagnostics stopped unexpectedly".into()
                        ))
                    }
                    Err(TryRecvError::Empty) => (),
                }
            }
            match job.child.try_wait() {
                Ok(status) => job.status = status,
                Err(_) => return Some(Err("Could not check the dataset preview process".into())),
            }
            let status = job.status?;
            let output = job.bytes.as_ref()?;
            let diagnostics = job.diagnostics.as_ref()?;
            if !status.success() {
                let stderr = String::from_utf8_lossy(diagnostics);
                let message=stderr.lines().find_map(|s|s.strip_prefix("Vespasian: "))
                    .map(|s|token.map_or_else(||s.to_string(), |token|s.replace(token,"[redacted]")))
                    .map(|s|s.chars().filter(|c|!c.is_control()).take(400).collect::<String>())
                    .unwrap_or_else(||"Could not preview this dataset. Check the runtime, dataset access and connection".into());
                return Some(Err(message));
            }
            let value = serde_json::from_slice::<Value>(output)
                .map_err(|_| "The dataset preview returned an invalid response".to_string())
                .and_then(validate);
            job.complete = true;
            Some(value)
        })();
        if outcome.is_some() {
            self.job.take();
        }
        outcome.map(|result| match (&self.token, result) {
            (Some(token), Ok(mut value)) => {
                redact(&mut value, token);
                Ok(value)
            }
            (Some(token), Err(error)) => Err(error.replace(token, "[redacted]")),
            (None, result) => result,
        })
    }

    pub fn cancel(&mut self) {
        self.job.take();
    }
}

fn redact(value: &mut Value, token: &str) {
    match value {
        Value::String(text) => *text = text.replace(token, "[redacted]"),
        Value::Array(values) => values.iter_mut().for_each(|value| redact(value, token)),
        Value::Object(object) => {
            *object = std::mem::take(object)
                .into_iter()
                .map(|(key, mut value)| {
                    redact(&mut value, token);
                    (key.replace(token, "[redacted]"), value)
                })
                .collect();
        }
        _ => (),
    }
}

fn validate(value: Value) -> Result<Value, String> {
    let pinned = value["revision"]
        .as_str()
        .is_some_and(|s| (40..=64).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_hexdigit()));
    let unresolved_error = value["revision"].is_null()
        && value["error"].is_string()
        && value["rows"].as_array().is_some_and(Vec::is_empty);
    if !value.is_object()
        || !value["rows"].is_array()
        || !value["columns"].is_array()
        || !value["configs"].is_array()
        || !value["splits"].is_array()
        || !value["repo"].is_string()
        || !value["split"].is_string()
        || (!pinned && !unresolved_error)
        || value["rows"]
            .as_array()
            .is_some_and(|rows| rows.len() > 5 || rows.iter().any(|r| !r.is_object()))
    {
        return Err("The dataset preview returned an incomplete response".into());
    }
    if let Some(mapped) = value.get("mapped_rows") {
        let row_count = value["rows"].as_array().map_or(0, Vec::len);
        if mapped.as_array().is_none_or(|pairs| {
            pairs.len() > 20
                || pairs.iter().any(|pair| {
                    !pair.is_object()
                        || pair["row_index"]
                            .as_u64()
                            .is_none_or(|index| index >= row_count as u64)
                        || pair["example_index"].as_u64().is_none()
                        || ["prompt", "target"].iter().any(|key| {
                            pair[key].as_str().is_none_or(|text| {
                                let optional = (*key == "prompt"
                                    && value["dataset_mode"] == "corpus")
                                    || (*key == "target" && value["dataset_mode"] == "prompts");
                                !optional && text.trim().is_empty()
                            })
                        })
                })
        }) {
            return Err("The dataset preview returned invalid prompt and target pairs".into());
        }
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn payload() -> Value {
        json!({"repo":"org/data","revision":"a".repeat(40),"split":"train","config":"main","configs":["main"],"splits":["train"],"columns":[{"name":"question","dtype":"string"}],"rows":[{"question":"hello"}],"mapping_error":"Choose a target column","error":null})
    }
    fn wait(client: &mut PreviewClient) -> Result<Value, String> {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(result) = client.poll() {
                return result;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn script(source: &str) -> Command {
        let mut command = Command::new(crate::platform::test_python());
        command.args(["-u", "-c", source]);
        command
    }
    #[test]
    fn metadata_errors_remain_available_to_ui() {
        assert!(validate(payload()).is_ok());
    }
    #[test]
    fn malformed_or_unpinned_results_are_rejected() {
        let mut p = payload();
        p["revision"] = json!("main");
        assert!(validate(p).is_err());
        assert!(validate(json!({"rows":[]})).is_err());
    }
    #[test]
    fn mapped_pairs_are_bounded_and_reference_real_preview_rows() {
        let mut value = payload();
        let pair = json!({"row_index":0,"example_index":2,"prompt":"Question","target":"0"});
        value["mapped_rows"] = json!([pair]);
        assert!(validate(value.clone()).is_ok());
        value["mapped_rows"][0]["row_index"] = json!(1);
        assert!(validate(value.clone()).is_err());
        value["mapped_rows"] = json!([pair]);
        value["mapped_rows"][0]["target"] = json!(null);
        assert!(validate(value.clone()).is_err());
        value["mapped_rows"] = json!(vec![pair; 21]);
        assert!(validate(value).is_err());
    }
    #[test]
    fn unpaired_preview_only_requires_the_relevant_field() {
        let mut value = payload();
        value["dataset_mode"] = json!("corpus");
        value["mapped_rows"] =
            json!([{"row_index":0,"example_index":0,"prompt":"","target":"Style example"}]);
        assert!(validate(value.clone()).is_ok());
        value["dataset_mode"] = json!("prompts");
        assert!(validate(value.clone()).is_err());
        value["mapped_rows"] = json!([{"row_index":0,"example_index":0,"prompt":"Write a short introduction","target":""}]);
        assert!(validate(value).is_ok());
    }
    #[test]
    fn preview_is_nonblocking_and_returns_real_payload() {
        let mut c = PreviewClient::new(crate::platform::test_python());
        c.spawn(script(&format!(
            "import time\ntime.sleep(.1)\nprint({:?})",
            payload().to_string()
        )))
        .unwrap();
        assert!(c.poll().is_none());
        assert_eq!(wait(&mut c).unwrap()["rows"][0]["question"], "hello");
        assert!(c.poll().is_none());
    }
    #[test]
    fn cancelled_job_cannot_deliver_stale_data() {
        let mut c = PreviewClient::new(crate::platform::test_python());
        c.spawn(script("import time\ntime.sleep(30)\nprint('{}')"))
            .unwrap();
        let now = Instant::now();
        c.cancel();
        assert!(now.elapsed() < Duration::from_secs(1));
        assert!(c.poll().is_none());
        c.spawn(script(&format!("print({:?})", payload().to_string())))
            .unwrap();
        assert_eq!(wait(&mut c).unwrap()["repo"], "org/data");
    }
    #[test]
    fn timeout_stops_the_child() {
        let mut c = PreviewClient::new(crate::platform::test_python());
        c.timeout = Duration::from_millis(30);
        c.spawn(script("import time\ntime.sleep(30)")).unwrap();
        assert!(wait(&mut c).unwrap_err().contains("timed out"));
        assert!(c.job.is_none());
    }
    #[test]
    fn unstructured_diagnostics_are_not_exposed() {
        let mut c = PreviewClient::new(crate::platform::test_python());
        c.spawn(script(
            "import sys\nprint('Authorization: Bearer secret',file=sys.stderr)\nsys.exit(1)",
        ))
        .unwrap();
        let error = wait(&mut c).unwrap_err();
        assert!(!error.contains("secret"));
    }
    #[test]
    fn output_size_is_bounded() {
        let mut c = PreviewClient::new(crate::platform::test_python());
        c.spawn(script("print('x'*1100000)")).unwrap();
        assert!(wait(&mut c).unwrap_err().contains("display limit"));
    }

    #[test]
    fn embedded_preview_bootstraps_shared_task_helper() {
        let directory = dataset_script().unwrap();
        let result = Command::new(crate::platform::test_python())
            .arg(directory.path().join("dataset.py"))
            .arg("--help")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let help = String::from_utf8_lossy(&result.stdout);
        assert!(help.contains("--task"));
        assert!(help.contains("--model-id"));
    }

    #[test]
    fn preview_script_directory_is_removed_after_job_cancellation() {
        let directory = dataset_script().unwrap();
        let path = directory.path().to_path_buf();
        assert!(path.join("dataset.py").is_file());
        assert!(path.join("task_metadata.py").is_file());
        let mut client = PreviewClient::new(crate::platform::test_python());
        client.spawn(script("import time; time.sleep(30)")).unwrap();
        client.job.as_mut().unwrap().script_directory = Some(directory);
        client.cancel();
        assert!(!path.exists());
    }

    #[test]
    #[cfg(unix)]
    fn token_reaches_only_the_child_environment() {
        use sha2::{Digest, Sha256};
        use std::os::unix::fs::PermissionsExt;

        let token = "hf_preview_session_test_secret";
        let inherited = std::env::var_os("HF_TOKEN");
        let directory = tempfile::tempdir().unwrap();
        let python = directory.path().join("python");
        let expected = format!("{:x}", Sha256::digest(token.as_bytes()));
        let source = format!(
            "#!/usr/bin/env python3\nimport hashlib,json,os,sys\ntoken=os.environ['HF_TOKEN']\npayload=json.loads({:?})\npayload['received']=hashlib.sha256(token.encode()).hexdigest()=={expected:?}\npayload['argument_leak']=any(token in arg for arg in sys.argv)\npayload['error']=token\nprint(json.dumps(payload))\n",
            payload().to_string()
        );
        std::fs::write(&python, source).unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o700)).unwrap();
        let mut client = PreviewClient::new(python);
        client.set_token(Some(token.into()));
        client
            .start(PreviewRequest {
                repo: "org/private-data".into(),
                config: None,
                split: "train".into(),
                revision: None,
                prompt_field: Some("question".into()),
                target_field: Some("answer".into()),
                dataset_mode: "paired".into(),
                hf_task: Some("text-generation".into()),
                model_id: None,
                text_field: None,
            })
            .unwrap();
        let value = wait(&mut client).unwrap();
        assert_eq!(value["received"], true);
        assert_eq!(value["argument_leak"], false);
        assert_eq!(value["error"], "[redacted]");
        assert!(!value.to_string().contains(token));
        assert_eq!(std::env::var_os("HF_TOKEN"), inherited);
    }

    #[test]
    fn changed_token_cancels_stale_preview() {
        let mut client = PreviewClient::new(crate::platform::test_python());
        client.set_token(Some("hf_first".into()));
        client.spawn(script("import time\ntime.sleep(30)")).unwrap();
        let original_pid = client.job.as_ref().unwrap().child.id();
        client.set_token(Some("hf_first".into()));
        assert_eq!(client.job.as_ref().unwrap().child.id(), original_pid);
        client.set_token(Some("hf_second".into()));
        assert!(client.job.is_none());
        assert!(client.poll().is_none());
        client.spawn(script("import time\ntime.sleep(30)")).unwrap();
        client.set_token(None);
        assert!(client.job.is_none());
    }

    #[test]
    fn structured_diagnostics_redact_the_token_before_truncating() {
        let token = "hf_preview_diagnostic_test_secret";
        let mut client = PreviewClient::new(crate::platform::test_python());
        client.set_token(Some(token.into()));
        client.spawn(script("import os,sys\nprint('Vespasian: '+'x'*390+os.environ['HF_TOKEN'],file=sys.stderr)\nsys.exit(1)")).unwrap();
        let error = wait(&mut client).unwrap_err();
        assert!(!error.contains("hf_preview"));
        assert!(error.contains("[redacted]"));
    }
}
