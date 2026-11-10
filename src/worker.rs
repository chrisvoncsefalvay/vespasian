use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::{fs::File, path::Path, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
};

#[derive(Debug)]
pub struct Rejection(pub String);
impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Rejection {}

pub struct Worker {
    child: Child,
    input: ChildStdin,
    output: tokio::io::Lines<BufReader<ChildStdout>>,
    id: u64,
    token: Option<String>,
    diagnostics: Option<tokio::task::JoinHandle<()>>,
}
impl Worker {
    pub fn spawn(python: &Path, script: &Path, log: &Path, token: Option<&str>) -> Result<Self> {
        let token = token.map(str::trim).filter(|token| !token.is_empty());
        let mut command = Command::new(python);
        if let Some(token) = token {
            command.env("HF_TOKEN", token);
        }
        let mut log = tokio::fs::File::from_std(File::create(log)?);
        let mut child = command
            .arg("-u")
            .arg(script)
            .env("TOKENIZERS_PARALLELISM", "false")
            .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| {
                format!(
                    "Cannot start {}. Run vespasian setup or set --python",
                    python.display()
                )
            })?;
        let stderr = child
            .stderr
            .take()
            .context("Worker diagnostics unavailable")?;
        let log_token = token.map(str::to_string);
        let diagnostics = tokio::spawn(async move {
            let mut stderr = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = stderr.next_line().await {
                let line = log_token
                    .as_ref()
                    .map_or_else(|| line.clone(), |token| line.replace(token, "[redacted]"));
                if log.write_all(line.as_bytes()).await.is_err()
                    || log.write_all(b"\n").await.is_err()
                {
                    break;
                }
            }
            let _ = log.flush().await;
        });
        let input = child.stdin.take().context("Worker input unavailable")?;
        let output =
            BufReader::new(child.stdout.take().context("Worker output unavailable")?).lines();
        Ok(Self {
            child,
            input,
            output,
            id: 0,
            token: token.map(str::to_string),
            diagnostics: Some(diagnostics),
        })
    }
    pub async fn request(&mut self, mut v: Value, events: &crate::Events) -> Result<Value> {
        self.id += 1;
        v["id"] = json!(self.id);
        let mut bytes = serde_json::to_vec(&v)?;
        bytes.push(b'\n');
        self.input.write_all(&bytes).await?;
        self.input.flush().await?;
        loop {
            let line = self
                .output
                .next_line()
                .await?
                .context("Model worker stopped. See worker.log in the run directory")?;
            let mut r: Value =
                serde_json::from_str(&line).context("Model worker returned an invalid message")?;
            if let Some(token) = &self.token {
                redact(&mut r, token);
            }
            if r["type"] == "result" {
                if r["id"].as_u64() != Some(self.id) {
                    bail!("Model worker response ID mismatch");
                }
                if r["ok"] == true {
                    return Ok(r["data"].clone());
                }
                return Err(Rejection(
                    r["error"]
                        .as_str()
                        .unwrap_or("Model worker failed")
                        .to_string(),
                )
                .into());
            }
            contextualise_event(&mut r, &v, self.id);
            events.emit(r)?;
        }
    }
    pub async fn stop(&mut self) {
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
        if let Some(mut diagnostics) = self.diagnostics.take() {
            if tokio::time::timeout(Duration::from_secs(1), &mut diagnostics)
                .await
                .is_err()
            {
                diagnostics.abort();
            }
        }
    }
}

fn contextualise_event(event: &mut Value, request: &Value, request_id: u64) {
    event["request_id"] = json!(request_id);
    for field in ["turn", "label"] {
        if event.get(field).is_none() {
            if let Some(value) = request.get(field) {
                event[field] = value.clone();
            }
        }
    }
    if event["type"] == "answer" {
        if let Some(receipt) = request.get("receipt") {
            let mut receipt = receipt.clone();
            if let Some(index) = event["current"].as_u64().and_then(|n| n.checked_sub(1)) {
                receipt["index"] = json!(index);
            }
            event["receipt"] = receipt;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    #[test]
    fn answer_events_link_to_their_checkpoint_receipt() {
        let request = json!({"turn":3,"label":"turn-003","receipt":{"file":"turn-003.json","pointer":"/evaluation/samples"}});
        let mut answer = json!({"type":"answer","current":2,"example":{"output":"Salve!"}});
        contextualise_event(&mut answer, &request, 8);
        assert_eq!(answer["turn"], 3);
        assert_eq!(answer["label"], "turn-003");
        assert_eq!(answer["request_id"], 8);
        assert_eq!(answer["receipt"]["index"], 1);
        assert_eq!(answer["receipt"]["pointer"], "/evaluation/samples");
        let mut progress = json!({"type":"progress","turn":0,"label":"baseline"});
        contextualise_event(&mut progress, &request, 9);
        assert_eq!(progress["turn"], 0);
        assert_eq!(progress["label"], "baseline");
        assert!(progress.get("receipt").is_none());
    }
    async fn failure(script: &str) -> anyhow::Error {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("worker.py");
        std::fs::write(&path, script).unwrap();
        let events = crate::Events {
            tx: None,
            trajectory: None,
            file: Mutex::new(File::create(dir.path().join("events.jsonl")).unwrap()),
            json: false,
            quiet: true,
        };
        let mut worker = Worker::spawn(
            &crate::platform::test_python(),
            &path,
            &dir.path().join("worker.log"),
            None,
        )
        .unwrap();
        let error = worker
            .request(json!({"command":"edit"}), &events)
            .await
            .unwrap_err();
        worker.stop().await;
        error
    }
    #[tokio::test]
    async fn explicit_rejections_are_recoverable() {
        let error=failure("import sys,json\nr=json.loads(sys.stdin.readline())\nprint(json.dumps({'type':'result','id':r['id'],'ok':False,'error':'empty direction'}),flush=True)\n").await;
        assert!(error.downcast_ref::<Rejection>().is_some());
    }
    #[tokio::test]
    async fn transport_failures_are_not_candidate_rejections() {
        for script in ["import sys\nsys.stdin.readline()\n", "import sys\nsys.stdin.readline()\nprint('invalid json',flush=True)\n", "import sys,json\nsys.stdin.readline()\nprint(json.dumps({'type':'result','id':999,'ok':False,'error':'wrong request'}),flush=True)\n"]{
            assert!(failure(script).await.downcast_ref::<Rejection>().is_none());
        }
    }

    #[tokio::test]
    async fn token_stays_out_of_arguments_requests_results_events_and_logs() {
        use sha2::{Digest, Sha256};

        let token = "hf_worker_session_test_secret";
        let inherited = std::env::var_os("HF_TOKEN");
        let directory = tempfile::tempdir().unwrap();
        let script = directory.path().join("worker.py");
        let log = directory.path().join("worker.log");
        let event_path = directory.path().join("events.jsonl");
        let expected = format!("{:x}", Sha256::digest(token.as_bytes()));
        std::fs::write(&script, format!(
            "import hashlib,json,os,sys\nrequest=json.loads(sys.stdin.readline())\ntoken=os.environ['HF_TOKEN']\nprint('token '+token,file=sys.stderr,flush=True)\nprint(json.dumps({{'type':'progress','message':token}}),flush=True)\ndata={{'received':hashlib.sha256(token.encode()).hexdigest()=={expected:?},'argument_leak':any(token in arg for arg in sys.argv),'request_leak':token in json.dumps(request),'echo':token}}\nprint(json.dumps({{'type':'result','id':request['id'],'ok':True,'data':data}}),flush=True)\n"
        )).unwrap();
        let events = crate::Events {
            tx: None,
            trajectory: None,
            file: Mutex::new(File::create(&event_path).unwrap()),
            json: false,
            quiet: true,
        };
        let mut worker =
            Worker::spawn(&crate::platform::test_python(), &script, &log, Some(token)).unwrap();
        let result = worker
            .request(json!({"command":"evaluate"}), &events)
            .await
            .unwrap();
        worker.stop().await;
        assert_eq!(result["received"], true);
        assert_eq!(result["argument_leak"], false);
        assert_eq!(result["request_leak"], false);
        assert_eq!(result["echo"], "[redacted]");
        assert!(!result.to_string().contains(token));
        for path in [&event_path, &log] {
            let contents = std::fs::read_to_string(path).unwrap();
            assert!(!contents.contains(token));
            assert!(contents.contains("[redacted]"));
        }
        assert_eq!(std::env::var_os("HF_TOKEN"), inherited);
    }

    #[tokio::test]
    async fn rejection_messages_redact_the_token() {
        let token = "hf_worker_rejection_test_secret";
        let directory = tempfile::tempdir().unwrap();
        let script = directory.path().join("worker.py");
        std::fs::write(&script,"import json,os,sys\nrequest=json.loads(sys.stdin.readline())\nprint(json.dumps({'type':'result','id':request['id'],'ok':False,'error':'Cannot load '+os.environ['HF_TOKEN']}),flush=True)\n").unwrap();
        let events = crate::Events {
            tx: None,
            trajectory: None,
            file: Mutex::new(File::create(directory.path().join("events.jsonl")).unwrap()),
            json: false,
            quiet: true,
        };
        let mut worker = Worker::spawn(
            &crate::platform::test_python(),
            &script,
            &directory.path().join("worker.log"),
            Some(token),
        )
        .unwrap();
        let error = worker
            .request(json!({"command":"load"}), &events)
            .await
            .unwrap_err();
        worker.stop().await;
        assert!(error.downcast_ref::<Rejection>().is_some());
        assert!(error.to_string().contains("[redacted]"));
        assert!(!error.to_string().contains(token));
    }
}
