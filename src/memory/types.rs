use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Mistake,
    Preference,
    Decision,
    Fact,
    Avoid,
}

impl MemoryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryType::Mistake => "mistake",
            MemoryType::Preference => "preference",
            MemoryType::Decision => "decision",
            MemoryType::Fact => "fact",
            MemoryType::Avoid => "avoid",
        }
    }
}

impl std::str::FromStr for MemoryType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "mistake" => Ok(MemoryType::Mistake),
            "preference" | "pref" => Ok(MemoryType::Preference),
            "decision" => Ok(MemoryType::Decision),
            "fact" => Ok(MemoryType::Fact),
            "avoid" => Ok(MemoryType::Avoid),
            other => Err(format!("Unknown memory type: {}", other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    Project,
    Global,
}

impl MemoryScope {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryScope::Project => "project",
            MemoryScope::Global => "global",
        }
    }
}

impl std::str::FromStr for MemoryScope {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "project" | "repo" => Ok(MemoryScope::Project),
            "global" | "user" => Ok(MemoryScope::Global),
            other => Err(format!("Unknown memory scope: {}", other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemorySource {
    Tool,
    User,
    Llm,
}

impl MemorySource {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemorySource::Tool => "tool",
            MemorySource::User => "user",
            MemorySource::Llm => "llm",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub project_id: Option<String>,
    pub scope: MemoryScope,
    pub memory_type: MemoryType,
    pub content: String,
    pub why: Option<String>,
    pub context: Option<String>,
    pub tags: Vec<String>,
    pub source: MemorySource,
    pub confidence: f32,
    pub importance: f32,
    pub use_count: u64,
    pub created_at: i64,
    pub last_used: i64,
    pub source_file: Option<PathBuf>,
}

impl Memory {
    pub fn new(
        id: String,
        project_id: Option<String>,
        scope: MemoryScope,
        memory_type: MemoryType,
        content: String,
    ) -> Self {
        let now = unix_ts();
        Self {
            id,
            project_id,
            scope,
            memory_type,
            content,
            why: None,
            context: None,
            tags: Vec::new(),
            source: MemorySource::User,
            confidence: 0.75,
            importance: default_importance(memory_type),
            use_count: 0,
            created_at: now,
            last_used: now,
            source_file: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScoredMemory {
    pub memory: Memory,
    pub score: f32,
    #[allow(dead_code)]
    pub match_kind: MatchKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchKind {
    Semantic,
    Keyword,
}

pub fn unix_ts() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

pub fn default_importance(memory_type: MemoryType) -> f32 {
    match memory_type {
        MemoryType::Avoid => 0.95,
        MemoryType::Mistake => 0.85,
        MemoryType::Preference => 0.8,
        MemoryType::Decision => 0.75,
        MemoryType::Fact => 0.6,
    }
}
