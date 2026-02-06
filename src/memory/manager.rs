use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::Mutex;
use uuid::Uuid;

use crate::memory::redaction::Redactor;
use crate::memory::store::MemoryStore;
use crate::memory::types::{
    default_importance, Memory, MemoryScope, MemorySource, MemoryType, ScoredMemory,
};

pub struct MemoryManager {
    store: Arc<MemoryStore>,
    redactor: Arc<Redactor>,
    project_root: PathBuf,
    // Rate limiting / dedupe
    last_failure: Mutex<Option<(String, String)>>, // (tool, signature)
    last_user_learned: Mutex<Option<String>>,
}

impl MemoryManager {
    pub async fn init(project_root: &Path) -> anyhow::Result<Self> {
        let store = Arc::new(MemoryStore::init(project_root).await?);
        Ok(Self {
            store,
            redactor: Arc::new(Redactor::new()),
            project_root: project_root.to_path_buf(),
            last_failure: Mutex::new(None),
            last_user_learned: Mutex::new(None),
        })
    }

    #[allow(dead_code)]
    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    #[allow(dead_code)]
    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    pub async fn save_explicit(
        &self,
        memory_type: MemoryType,
        content: &str,
        why: Option<&str>,
        context: Option<&str>,
        tags: Vec<String>,
        scope: MemoryScope,
        source_file: Option<&Path>,
        source: MemorySource,
        confidence: Option<f32>,
        importance: Option<f32>,
    ) -> anyhow::Result<Memory> {
        let id = Uuid::new_v4().to_string();
        let mut memory = Memory::new(
            id,
            Some(self.store.project_id().to_string()),
            scope,
            memory_type,
            self.redactor.redact(content),
        );
        memory.why = why.map(|s| self.redactor.redact(s));
        memory.context = context.map(|s| self.redactor.redact(s));
        memory.tags = tags;
        memory.source = source;
        if let Some(c) = confidence {
            memory.confidence = c.clamp(0.0, 1.0);
        }
        if let Some(i) = importance {
            memory.importance = i.clamp(0.0, 1.0);
        }
        if let Some(p) = source_file {
            memory.source_file = Some(p.to_path_buf());
        }
        self.store.insert(&memory).await?;
        Ok(memory)
    }

    pub async fn recall(
        &self,
        query: &str,
        limit: usize,
        type_filter: Option<Vec<MemoryType>>,
    ) -> anyhow::Result<Vec<ScoredMemory>> {
        let results = self.store.search(query, limit, type_filter).await?;
        let ids: Vec<String> = results.iter().map(|sm| sm.memory.id.clone()).collect();
        // Best-effort reinforcement: bump recency/frequency for surfaced memories.
        let _ = self.store.mark_used(&ids).await;
        Ok(results)
    }

    pub async fn list(
        &self,
        limit: usize,
        type_filter: Option<MemoryType>,
    ) -> anyhow::Result<Vec<Memory>> {
        self.store.list(limit, type_filter).await
    }

    pub async fn forget(&self, id: &str) -> anyhow::Result<bool> {
        self.store.delete(id).await
    }

    /// Called after a tool call completes.
    ///
    /// This is a high-signal trigger to store "mistake" memories on failures.
    pub async fn on_tool_complete(
        &self,
        tool_name: &str,
        target: &str,
        output: &str,
        success: bool,
    ) {
        if success {
            return;
        }

        // Avoid storing obvious secrets.
        if self.redactor.looks_sensitive(output) {
            return;
        }

        let signature = format!("{}:{}", tool_name, normalize_sig(target));
        {
            let mut last = self.last_failure.lock().await;
            if last
                .as_ref()
                .is_some_and(|(t, sig)| t == tool_name && sig == &signature)
            {
                return;
            }
            *last = Some((tool_name.to_string(), signature.clone()));
        }

        // Minimal, actionable failure memory.
        let content = format!(
            "Tool failure: {}\nTarget: {}\nError: {}\nNext time: avoid repeating the same command; inspect error and adjust approach.",
            tool_name,
            target,
            output.lines().next().unwrap_or(output)
        );

        let _ = self
            .save_explicit(
                MemoryType::Mistake,
                &content,
                Some("Tool call failed; store the first-line error and the tool/target signature."),
                None,
                vec!["tool_failure".to_string(), tool_name.to_string()],
                MemoryScope::Project,
                None,
                MemorySource::Tool,
                Some(0.65),
                Some(default_importance(MemoryType::Mistake)),
            )
            .await;
    }

    /// Called when the user sends a message.
    ///
    /// This is a lightweight, rules-based extractor for high-signal preferences/corrections.
    pub async fn on_user_message(&self, user_message: &str, previous_agent_message: Option<&str>) {
        let msg = user_message.trim();
        if msg.is_empty() {
            return;
        }

        // Avoid storing obvious secrets.
        if self.redactor.looks_sensitive(msg) {
            return;
        }

        let lower = msg.to_lowercase();

        let starts_no =
            lower.starts_with("no") || lower.starts_with("nah") || lower.starts_with("nope");
        let has_avoid =
            lower.contains("don't") || lower.contains("do not") || lower.contains("never");
        let has_pref =
            lower.contains("i prefer") || lower.contains("prefer ") || lower.contains("i want");

        let mut memory_type: Option<MemoryType> = None;
        let mut why: Option<&'static str> = None;
        let mut importance: Option<f32> = None;
        let mut confidence: Option<f32> = None;
        let mut tags: Vec<String> = Vec::new();
        let mut context: Option<String> = None;

        if has_avoid {
            memory_type = Some(MemoryType::Avoid);
            why = Some("User asked to avoid this in the future.");
            importance = Some(default_importance(MemoryType::Avoid));
            confidence = Some(0.85);
            tags.extend(["user", "avoid"].into_iter().map(str::to_string));
        } else if has_pref {
            memory_type = Some(MemoryType::Preference);
            why = Some("User preference that should guide future responses/changes.");
            importance = Some(default_importance(MemoryType::Preference));
            confidence = Some(0.80);
            tags.extend(["user", "preference"].into_iter().map(str::to_string));
        } else if starts_no && msg.len() >= 18 {
            // Likely a correction to the previous assistant response.
            memory_type = Some(MemoryType::Avoid);
            why = Some("User corrected the previous suggestion; avoid repeating it.");
            importance = Some(default_importance(MemoryType::Avoid));
            confidence = Some(0.70);
            tags.extend(["user", "correction"].into_iter().map(str::to_string));

            if let Some(prev) = previous_agent_message {
                let prev_line = prev.lines().next().unwrap_or(prev).trim();
                if !prev_line.is_empty() {
                    context = Some(format!("Previous assistant message: {}", prev_line));
                }
            }
        }

        let Some(memory_type) = memory_type else {
            return;
        };

        // Dedupe within session (avoid storing the same preference repeatedly).
        let sig = format!("{}:{}", memory_type.as_str(), lower);
        {
            let mut last = self.last_user_learned.lock().await;
            if last.as_ref().is_some_and(|s| s == &sig) {
                return;
            }
            *last = Some(sig);
        }

        let _ = self
            .save_explicit(
                memory_type,
                msg,
                why,
                context.as_deref(),
                tags,
                MemoryScope::Project,
                None,
                MemorySource::User,
                confidence,
                importance,
            )
            .await;
    }

    /// Use an LLM to extract durable memories from a completed turn summary.
    ///
    /// This is optional (call from the app behind a config flag). It should be used to
    /// capture decisions and preferences that rule-based extraction misses.
    pub async fn llm_extract_and_save(
        &self,
        api_key: String,
        model: String,
        turn_summary: String,
    ) -> usize {
        let candidates = match crate::memory::llm_extractor::extract_memories_from_turn(
            api_key,
            model,
            turn_summary,
        )
        .await
        {
            Ok(c) => c,
            Err(_) => return 0,
        };

        let mut saved = 0usize;
        for c in candidates {
            if self.redactor.looks_sensitive(&c.content) {
                continue;
            }

            if c.confidence.unwrap_or(0.75) < 0.60 {
                continue;
            }
            if c.importance.unwrap_or(default_importance(c.memory_type)) < 0.30 {
                continue;
            }

            let mut tags = c.tags;
            tags.push("llm_extract".to_string());

            if self
                .save_explicit(
                    c.memory_type,
                    &c.content,
                    c.why.as_deref(),
                    c.context.as_deref(),
                    tags,
                    MemoryScope::Project,
                    None,
                    MemorySource::Llm,
                    c.confidence,
                    c.importance,
                )
                .await
                .is_ok()
            {
                saved += 1;
            }
        }

        saved
    }

    /// Build a compact context block to inject into the system prompt.
    pub async fn build_injection_context(
        &self,
        user_message: &str,
        active_paths: &[PathBuf],
    ) -> String {
        let mut q = user_message.to_string();
        for p in active_paths.iter().take(5) {
            q.push_str(" ");
            q.push_str(&p.to_string_lossy());
        }

        let results = self
            .recall(
                &q,
                8,
                Some(vec![
                    MemoryType::Avoid,
                    MemoryType::Mistake,
                    MemoryType::Preference,
                    MemoryType::Decision,
                ]),
            )
            .await
            .unwrap_or_default();

        if results.is_empty() {
            return String::new();
        }

        let mut out = String::new();
        out.push_str("\n[Memory]\n");
        for sm in results {
            let m = sm.memory;
            out.push_str(&format!(
                "- ({}, {}): {}\n",
                m.memory_type.as_str(),
                m.scope.as_str(),
                single_line(&m.content)
            ));
        }
        out
    }
}

fn single_line(s: &str) -> String {
    let mut out = s.lines().next().unwrap_or("").trim().to_string();
    if out.len() > 200 {
        out.truncate(200);
        out.push('…');
    }
    out
}

fn normalize_sig(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.len() > 120 {
        let mut t = trimmed.to_string();
        t.truncate(120);
        t
    } else {
        trimmed.to_string()
    }
}
