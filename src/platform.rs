use std::path::{Path, PathBuf};

pub fn home_dir() -> PathBuf {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_default()
}

fn application_dir(os: &str, home: &Path, local: Option<PathBuf>, xdg: Option<PathBuf>) -> PathBuf {
    match os {
        "windows" => local.unwrap_or_else(|| home.join("AppData").join("Local")),
        "macos" => home.join("Library").join("Application Support"),
        _ => xdg.unwrap_or_else(|| home.join(".local").join("share")),
    }
    .join("vespasian")
}

pub fn data_dir() -> PathBuf {
    if let Some(directory) = std::env::var_os("VESPASIAN_DATA_HOME") {
        return directory.into();
    }
    application_dir(
        std::env::consts::OS,
        &home_dir(),
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from),
        std::env::var_os("XDG_DATA_HOME").map(PathBuf::from),
    )
}

pub fn runtime_python(directory: &Path) -> PathBuf {
    venv_python(directory, cfg!(windows))
}

fn venv_python(directory: &Path, windows: bool) -> PathBuf {
    if windows {
        directory.join("Scripts").join("python.exe")
    } else {
        directory.join("bin").join("python")
    }
}

pub fn default_python() -> &'static str {
    if cfg!(windows) {
        "python"
    } else {
        "python3"
    }
}

pub fn runtime_python_request() -> &'static str {
    if cfg!(all(windows, target_arch = "aarch64")) {
        "cpython-3.12-windows-x86_64-none"
    } else {
        "3.12"
    }
}

/// npm-installed editor CLIs use command shims on Windows.
pub fn executable(name: &str) -> PathBuf {
    #[cfg(windows)]
    if let Some(paths) = std::env::var_os("PATH") {
        for directory in std::env::split_paths(&paths) {
            for extension in ["exe", "cmd", "bat"] {
                let candidate = directory.join(format!("{name}.{extension}"));
                if candidate.is_file() {
                    return candidate;
                }
            }
        }
    }
    PathBuf::from(name)
}

#[cfg(windows)]
pub fn terminate_tree(pid: u32) {
    use std::process::{Command, Stdio};
    let _ = Command::new("taskkill.exe")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(test)]
pub fn test_python() -> PathBuf {
    std::env::var_os("VESPASIAN_TEST_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_python().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_paths_follow_each_platforms_virtual_environment_layout() {
        let root = Path::new("runtime");
        assert_eq!(
            venv_python(root, true),
            root.join("Scripts").join("python.exe")
        );
        assert_eq!(venv_python(root, false), root.join("bin").join("python"));
    }

    #[test]
    fn application_paths_follow_platform_conventions() {
        let home = Path::new("profile");
        assert_eq!(
            application_dir("windows", home, Some("local".into()), None),
            Path::new("local").join("vespasian")
        );
        assert_eq!(
            application_dir("windows", home, None, None),
            home.join("AppData/Local/vespasian")
        );
        assert_eq!(
            application_dir("macos", home, None, None),
            home.join("Library/Application Support/vespasian")
        );
        assert_eq!(
            application_dir("linux", home, None, Some("xdg".into())),
            Path::new("xdg").join("vespasian")
        );
        assert_eq!(
            application_dir("linux", home, None, None),
            home.join(".local/share/vespasian")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_command_shims_receive_literal_arguments() {
        let directory = tempfile::tempdir().unwrap();
        let script = directory.path().join("echo-arguments.cmd");
        std::fs::write(&script, "@echo off\r\necho %~1\r\n").unwrap();
        let result = std::process::Command::new(&script)
            .arg("model name")
            .output()
            .unwrap();
        assert!(result.status.success());
        assert_eq!(
            String::from_utf8(result.stdout).unwrap().trim(),
            "model name"
        );
    }
}
