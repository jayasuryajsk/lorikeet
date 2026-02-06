use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::config::AppConfig;

#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    pub enabled: bool,
    pub root: PathBuf,
    pub allow_paths: Vec<PathBuf>,
    pub deny_paths: Vec<PathBuf>,
    pub allow_commands: HashSet<String>,
    pub allow_tools: HashSet<String>,
}

#[derive(Debug)]
pub enum SandboxError {
    ToolNotAllowed(String),
    PathNotAllowed(PathBuf),
    CommandNotAllowed(String),
}

#[derive(Debug, Clone, Default)]
pub struct SandboxDecision {
    pub allowed: bool,
    pub reason: Option<String>,
}

impl SandboxDecision {
    pub fn allow() -> Self {
        Self {
            allowed: true,
            reason: None,
        }
    }

    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            reason: Some(reason.into()),
        }
    }
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::ToolNotAllowed(name) => write!(f, "Sandbox: tool not allowed: {}", name),
            SandboxError::PathNotAllowed(path) => {
                write!(f, "Sandbox: path not allowed: {}", path.display())
            }
            SandboxError::CommandNotAllowed(cmd) => {
                write!(f, "Sandbox: command not allowed: {}", cmd)
            }
        }
    }
}

impl std::error::Error for SandboxError {}

impl SandboxPolicy {
    pub fn from_config(config: AppConfig, workspace_root: PathBuf, tool_names: &[&str]) -> Self {
        let sandbox = config.sandbox.unwrap_or_default();

        let enabled = sandbox.enabled.unwrap_or(true);
        let root = sandbox.root.unwrap_or_else(|| workspace_root.clone());

        let allow_paths = sandbox.allow_paths.unwrap_or_else(|| vec![root.clone()]);

        let deny_paths = sandbox.deny_paths.unwrap_or_default();

        let allow_commands = sandbox
            .allow_commands
            .unwrap_or_else(default_allow_commands)
            .into_iter()
            .collect::<HashSet<_>>();

        let allow_tools = sandbox
            .allow_tools
            .unwrap_or_else(|| tool_names.iter().map(|t| t.to_string()).collect())
            .into_iter()
            .collect::<HashSet<_>>();

        Self {
            enabled,
            root,
            allow_paths,
            deny_paths,
            allow_commands,
            allow_tools,
        }
    }

    pub fn check_tool_allowed(&self, tool_name: &str) -> Result<(), SandboxError> {
        if !self.enabled {
            return Ok(());
        }
        if self.allow_tools.contains(tool_name) {
            Ok(())
        } else {
            Err(SandboxError::ToolNotAllowed(tool_name.to_string()))
        }
    }

    pub fn check_path_allowed(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        if !self.enabled {
            return Ok(path.to_path_buf());
        }

        let normalized = normalize_path(path, &self.root);

        if self
            .deny_paths
            .iter()
            .any(|deny| is_within(&normalized, deny))
        {
            return Err(SandboxError::PathNotAllowed(normalized));
        }

        if self
            .allow_paths
            .iter()
            .any(|allow| is_within(&normalized, allow))
        {
            Ok(normalized)
        } else {
            Err(SandboxError::PathNotAllowed(normalized))
        }
    }

    pub fn check_command_allowed(&self, cmd: &str) -> Result<(), SandboxError> {
        if !self.enabled {
            return Ok(());
        }
        let executable = extract_executable(cmd);
        if executable.is_empty() {
            return Err(SandboxError::CommandNotAllowed(cmd.to_string()));
        }
        if self.allow_commands.contains(executable) {
            Ok(())
        } else {
            Err(SandboxError::CommandNotAllowed(executable.to_string()))
        }
    }

    // Best-effort path allow/deny checks for shell commands. This is not a full shell parser;
    // it is intended to catch obvious path arguments like `./foo`, `../bar`, `/etc/passwd`.
    pub fn check_bash_paths(&self, command: &str) -> Result<(), SandboxError> {
        if !self.enabled {
            return Ok(());
        }

        let mut parts = command.split_whitespace();
        let _exec = parts.next();

        for part in parts {
            if part.starts_with('-') && !part.contains('/') {
                continue;
            }
            if looks_like_path(part) {
                let path = Path::new(part.trim_matches('"').trim_matches('\''));
                let _ = self.check_path_allowed(path)?;
            }
        }
        Ok(())
    }
}

fn default_allow_commands() -> Vec<String> {
    vec![
        "rg", "ls", "cat", "pwd", "sed", "awk", "find", "wc", "head", "tail", "git",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect()
}

fn normalize_path(path: &Path, root: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };

    match std::fs::canonicalize(&joined) {
        Ok(canon) => canon,
        Err(_) => joined,
    }
}

fn is_within(candidate: &Path, base: &Path) -> bool {
    if base.as_os_str().is_empty() {
        return false;
    }
    if let Ok(base_canon) = std::fs::canonicalize(base) {
        candidate.starts_with(base_canon)
    } else {
        candidate.starts_with(base)
    }
}

fn looks_like_path(token: &str) -> bool {
    token.starts_with('/')
        || token.starts_with("./")
        || token.starts_with("../")
        || token.contains('/')
}

fn extract_executable(command: &str) -> &str {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return "";
    }
    // Split on whitespace; accept simple commands and pipelines.
    let mut parts = trimmed.split_whitespace();
    let first = parts.next().unwrap_or("");
    // If command starts with env VAR= or `command` with leading `env`, try next token.
    if first.contains('=') && parts.clone().next().is_some() {
        return parts.next().unwrap_or("");
    }
    if first == "env" {
        return parts.next().unwrap_or("");
    }
    first
}
