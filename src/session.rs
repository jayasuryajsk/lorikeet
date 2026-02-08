use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::app::{Message, Role, ToolOutput, ToolStatus};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    Message {
        ts: i64,
        role: String,
        content: String,
        #[serde(default)]
        reasoning: Option<String>,
        #[serde(default)]
        tool_group_id: Option<u64>,
        #[serde(default)]
        local: bool,
    },
    Checkpoint {
        ts: i64,
        id: String,
        #[serde(default)]
        name: Option<String>,
    },
    Tool {
        ts: i64,
        tool: String,
        target: String,
        output: String,
        status: String,
        elapsed_ms: u128,

        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        args_raw: Option<String>,
        #[serde(default)]
        cwd: Option<String>,
        #[serde(default)]
        sandbox_allowed: Option<bool>,
        #[serde(default)]
        sandbox_reason: Option<String>,
        #[serde(default)]
        group_id: Option<u64>,
    },
    Meta {
        ts: i64,
        session_id: String,
        project_id: String,
        version: String,
    },
}

#[derive(Debug, Clone)]
pub struct SessionStore {
    pub session_id: String,
    pub project_id: String,
    pub events_path: PathBuf,
    pub latest_path: PathBuf,
}

impl SessionStore {
    pub fn new(project_root: &Path, session_id: String) -> std::io::Result<Self> {
        let base = sessions_dir(project_root)?;
        std::fs::create_dir_all(&base)?;

        let events_path = base.join(format!("{}.jsonl", session_id));
        let latest_path = base.join("latest");

        Ok(Self {
            project_id: project_id(project_root),
            session_id,
            events_path,
            latest_path,
        })
    }

    pub fn open_latest(project_root: &Path) -> std::io::Result<Option<Self>> {
        let base = sessions_dir(project_root)?;
        let latest_path = base.join("latest");
        let Ok(name) = std::fs::read_to_string(&latest_path) else {
            return Ok(None);
        };
        let session_id = name.trim().trim_end_matches(".jsonl").to_string();
        if session_id.is_empty() {
            return Ok(None);
        }
        let store = Self::new(project_root, session_id)?;
        if store.events_path.exists() {
            Ok(Some(store))
        } else {
            Ok(None)
        }
    }

    pub fn set_latest(&self) {
        let _ = std::fs::write(&self.latest_path, format!("{}.jsonl", self.session_id));
    }

    pub fn append(&self, event: &SessionEvent) {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.events_path)
        {
            if let Ok(line) = serde_json::to_string(event) {
                let _ = writeln_line(&mut f, &line);
            }
        }
    }

    pub fn init_file(&self) {
        self.set_latest();
        self.append(&SessionEvent::Meta {
            ts: unix_ts(),
            session_id: self.session_id.clone(),
            project_id: self.project_id.clone(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        });
    }

    pub fn record_message(&self, msg: &Message) {
        // Persist tool/system prompts too; display filtering is UI-level.
        self.append(&SessionEvent::Message {
            ts: unix_ts(),
            role: role_to_string(msg.role),
            content: msg.content.clone(),
            reasoning: msg.reasoning.clone(),
            tool_group_id: msg.tool_group_id,
            local: msg.local,
        });
    }

    pub fn record_checkpoint(&self, id: &str, name: Option<&str>) {
        self.append(&SessionEvent::Checkpoint {
            ts: unix_ts(),
            id: id.to_string(),
            name: name.map(|s| s.to_string()),
        });
    }

    pub fn record_tool(&self, tool: &ToolOutput) {
        self.append(&SessionEvent::Tool {
            ts: unix_ts(),
            tool: tool.tool.clone(),
            target: tool.target.clone(),
            output: tool.output.clone(),
            status: tool_status_to_string(tool.status),
            elapsed_ms: tool.elapsed().as_millis(),

            call_id: Some(tool.call_id.clone()),
            args_raw: Some(tool.args_raw.clone()),
            cwd: Some(tool.cwd.display().to_string()),
            sandbox_allowed: Some(tool.sandbox.allowed),
            sandbox_reason: tool.sandbox.reason.clone(),
            group_id: Some(tool.group_id),
        });
    }

    pub fn load_events(&self) -> std::io::Result<Vec<SessionEvent>> {
        let data = std::fs::read_to_string(&self.events_path)?;
        let mut out = Vec::new();
        for line in data.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(ev) = serde_json::from_str::<SessionEvent>(line) {
                out.push(ev);
            }
        }
        Ok(out)
    }

    pub fn count_events_lines(&self) -> std::io::Result<usize> {
        let f = std::fs::File::open(&self.events_path)?;
        let r = std::io::BufReader::new(f);
        Ok(r.lines().count())
    }

    pub fn truncate_to_lines(&self, keep_lines: usize) -> std::io::Result<()> {
        let src = std::fs::File::open(&self.events_path)?;
        let mut r = std::io::BufReader::new(src);

        let tmp = self
            .events_path
            .with_extension(format!("{}.tmp", self.session_id));
        let mut w = std::fs::File::create(&tmp)?;

        let mut line = String::new();
        for _ in 0..keep_lines {
            line.clear();
            let n = r.read_line(&mut line)?;
            if n == 0 {
                break;
            }
            w.write_all(line.as_bytes())?;
        }
        w.flush()?;
        drop(w);

        std::fs::rename(&tmp, &self.events_path)?;
        self.set_latest();
        Ok(())
    }
}

pub fn replay_into(
    events: &[SessionEvent],
    messages: &mut Vec<Message>,
    tools: &mut Vec<ToolOutput>,
) {
    let mut turn_id: u64 = 0;
    for ev in events {
        match ev {
            SessionEvent::Message {
                role,
                content,
                reasoning,
                tool_group_id,
                local,
                ..
            } => {
                if role.eq_ignore_ascii_case("user") {
                    turn_id = turn_id.saturating_add(1);
                }
                messages.push(Message {
                    role: string_to_role(role),
                    content: content.clone(),
                    reasoning: reasoning.clone(),
                    tool_calls: None,
                    tool_group_id: *tool_group_id,
                    local: *local,
                });
            }
            SessionEvent::Checkpoint { .. } => {
                // Session replay uses message/tool events only.
            }
            SessionEvent::Tool {
                tool,
                target,
                output,
                status,
                call_id,
                args_raw,
                cwd,
                sandbox_allowed,
                sandbox_reason,
                group_id,
                ..
            } => {
                let call_id = call_id.clone().unwrap_or_else(|| "<legacy>".to_string());
                let args_raw = args_raw.clone().unwrap_or_default();
                let cwd_path = cwd
                    .as_ref()
                    .map(|s| std::path::PathBuf::from(s))
                    .unwrap_or_else(|| std::path::PathBuf::from("."));
                let sandbox = crate::sandbox::SandboxDecision {
                    allowed: sandbox_allowed.unwrap_or(true),
                    reason: sandbox_reason.clone(),
                };

                let mut t = ToolOutput::new(
                    call_id,
                    tool.clone(),
                    args_raw,
                    target.clone(),
                    cwd_path,
                    sandbox,
                    turn_id,
                    group_id.unwrap_or(0),
                );
                t.set_output(output.clone());
                let success = status.eq_ignore_ascii_case("success");
                t.complete(success);
                // Preserve explicit error status if present.
                if status.eq_ignore_ascii_case("error") {
                    t.status = ToolStatus::Error;
                }
                tools.push(t);
            }
            SessionEvent::Meta { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_replay_assigns_turn_ids() {
        let events = vec![
            SessionEvent::Message {
                ts: 0,
                role: "user".into(),
                content: "hi".into(),
                reasoning: None,
                tool_group_id: None,
                local: false,
            },
            SessionEvent::Message {
                ts: 0,
                role: "assistant".into(),
                content: "ok".into(),
                reasoning: None,
                tool_group_id: None,
                local: false,
            },
            SessionEvent::Tool {
                ts: 0,
                tool: "bash".into(),
                target: "echo 1".into(),
                output: "1".into(),
                status: "success".into(),
                elapsed_ms: 1,
                call_id: None,
                args_raw: None,
                cwd: None,
                sandbox_allowed: None,
                sandbox_reason: None,
                group_id: None,
            },
            SessionEvent::Message {
                ts: 0,
                role: "user".into(),
                content: "next".into(),
                reasoning: None,
                tool_group_id: None,
                local: false,
            },
            SessionEvent::Tool {
                ts: 0,
                tool: "bash".into(),
                target: "echo 2".into(),
                output: "2".into(),
                status: "success".into(),
                elapsed_ms: 1,
                call_id: None,
                args_raw: None,
                cwd: None,
                sandbox_allowed: None,
                sandbox_reason: None,
                group_id: None,
            },
        ];

        let mut messages = Vec::new();
        let mut tools = Vec::new();
        replay_into(&events, &mut messages, &mut tools);

        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].turn_id, 1);
        assert_eq!(tools[1].turn_id, 2);
    }

    #[test]
    fn session_replay_restores_tool_invocation_metadata() {
        let events = vec![
            SessionEvent::Message {
                ts: 0,
                role: "user".into(),
                content: "hi".into(),
                reasoning: None,
                tool_group_id: None,
                local: false,
            },
            SessionEvent::Tool {
                ts: 0,
                tool: "bash".into(),
                target: "echo 1".into(),
                output: "1".into(),
                status: "success".into(),
                elapsed_ms: 1,
                call_id: Some("call-a".into()),
                args_raw: Some(r#"{"command":"echo 1"}"#.into()),
                cwd: Some("/tmp".into()),
                sandbox_allowed: Some(true),
                sandbox_reason: None,
                group_id: None,
            },
        ];

        let mut messages = Vec::new();
        let mut tools = Vec::new();
        replay_into(&events, &mut messages, &mut tools);

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].call_id, "call-a");
        assert!(tools[0].args_raw.contains("echo 1"));
        assert_eq!(tools[0].cwd.to_string_lossy(), "/tmp");
        assert!(tools[0].sandbox.allowed);
    }
}

fn sessions_dir(project_root: &Path) -> std::io::Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::Other, "Could not determine home dir")
    })?;

    Ok(home
        .join(".lorikeet")
        .join("sessions")
        .join(project_id(project_root)))
}

fn project_id(root: &Path) -> String {
    let canon = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let s = canon.to_string_lossy().to_string();

    // Stable, short-ish directory name.
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn role_to_string(r: Role) -> String {
    match r {
        Role::User => "user",
        Role::Agent => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
    .to_string()
}

fn string_to_role(s: &str) -> Role {
    match s {
        "user" => Role::User,
        "assistant" => Role::Agent,
        "system" => Role::System,
        "tool" => Role::Tool,
        _ => Role::System,
    }
}

fn tool_status_to_string(s: ToolStatus) -> String {
    match s {
        ToolStatus::Running => "running",
        ToolStatus::Success => "success",
        ToolStatus::Error => "error",
    }
    .to_string()
}

fn unix_ts() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn writeln_line(f: &mut std::fs::File, line: &str) -> std::io::Result<()> {
    use std::io::Write;
    f.write_all(line.as_bytes())?;
    f.write_all(b"\n")?;
    Ok(())
}
