use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
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
    },
    Tool {
        ts: i64,
        tool: String,
        target: String,
        output: String,
        status: String,
        elapsed_ms: u128,
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
                });
            }
            SessionEvent::Tool {
                tool,
                target,
                output,
                status,
                ..
            } => {
                let mut t = ToolOutput::new(tool.clone(), target.clone(), turn_id);
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
            },
            SessionEvent::Message {
                ts: 0,
                role: "assistant".into(),
                content: "ok".into(),
                reasoning: None,
            },
            SessionEvent::Tool {
                ts: 0,
                tool: "bash".into(),
                target: "echo 1".into(),
                output: "1".into(),
                status: "success".into(),
                elapsed_ms: 1,
            },
            SessionEvent::Message {
                ts: 0,
                role: "user".into(),
                content: "next".into(),
                reasoning: None,
            },
            SessionEvent::Tool {
                ts: 0,
                tool: "bash".into(),
                target: "echo 2".into(),
                output: "2".into(),
                status: "success".into(),
                elapsed_ms: 1,
            },
        ];

        let mut messages = Vec::new();
        let mut tools = Vec::new();
        replay_into(&events, &mut messages, &mut tools);

        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].turn_id, 1);
        assert_eq!(tools[1].turn_id, 2);
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
