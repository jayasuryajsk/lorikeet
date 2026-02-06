use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// A chunk of code extracted from a source file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    /// Unique identifier for this chunk (index in the vector store)
    pub id: u64,
    /// The actual code content
    pub content: String,
    /// Metadata about where this chunk came from
    pub metadata: ChunkMetadata,
}

/// Metadata about a code chunk's origin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Path to the source file (relative to project root)
    pub file_path: PathBuf,
    /// Starting line number (1-indexed)
    pub start_line: usize,
    /// Ending line number (1-indexed)
    pub end_line: usize,
    /// Programming language of the file
    pub language: Language,
    /// Optional: name of the function/class/method this chunk belongs to
    pub symbol_name: Option<String>,
    /// Optional: type of symbol (function, class, method, etc.)
    pub symbol_type: Option<SymbolType>,
}

/// Supported programming languages for AST-aware chunking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    C,
    Cpp,
    Java,
    Ruby,
    Unknown,
}

impl Language {
    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Language::Rust,
            "py" => Language::Python,
            "js" | "mjs" | "cjs" => Language::JavaScript,
            "ts" | "tsx" => Language::TypeScript,
            "go" => Language::Go,
            "c" | "h" => Language::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Language::Cpp,
            "java" => Language::Java,
            "rb" => Language::Ruby,
            _ => Language::Unknown,
        }
    }

    /// Get common file extensions for this language
    #[allow(dead_code)]
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Language::Rust => &["rs"],
            Language::Python => &["py"],
            Language::JavaScript => &["js", "mjs", "cjs"],
            Language::TypeScript => &["ts", "tsx"],
            Language::Go => &["go"],
            Language::C => &["c", "h"],
            Language::Cpp => &["cpp", "cc", "cxx", "hpp", "hh", "hxx"],
            Language::Java => &["java"],
            Language::Ruby => &["rb"],
            Language::Unknown => &[],
        }
    }
}

/// Type of code symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolType {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Interface,
    Trait,
    Module,
    Impl,
    Other,
}

/// A search result from the semantic search
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matched code chunk
    pub chunk: CodeChunk,
    /// Similarity score (higher is better, typically 0.0-1.0)
    pub score: f32,
}

/// Configuration for the semantic search engine
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Directory to store the index
    pub index_dir: PathBuf,
    /// Number of results to return
    pub top_k: usize,
    /// Minimum similarity score threshold
    pub min_score: f32,
    /// Maximum chunk size in characters
    pub max_chunk_size: usize,
    /// File patterns to exclude (glob patterns)
    pub exclude_patterns: Vec<String>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            index_dir: default_index_base_dir(),
            top_k: 10,
            min_score: 0.3,
            max_chunk_size: 2000,
            exclude_patterns: vec![
                "**/target/**".into(),
                "**/node_modules/**".into(),
                "**/.git/**".into(),
                "**/dist/**".into(),
                "**/build/**".into(),
                "**/__pycache__/**".into(),
                "**/vendor/**".into(),
            ],
        }
    }
}

impl SearchConfig {
    pub fn for_workspace(workspace_root: &Path) -> Self {
        let mut cfg = Self::default();
        cfg.index_dir = index_dir_for_workspace(workspace_root);
        cfg
    }
}

fn default_index_base_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".lorikeet")
        .join("index")
}

pub fn index_dir_for_workspace(workspace_root: &Path) -> PathBuf {
    let base = default_index_base_dir();
    base.join(project_id(workspace_root))
}

fn project_id(root: &Path) -> String {
    let canon = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let s = canon.to_string_lossy().to_string();

    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Statistics about the index
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    /// Total number of chunks indexed
    pub total_chunks: usize,
    /// Total number of files indexed
    pub total_files: usize,
    /// Size of the index on disk (bytes)
    pub index_size_bytes: u64,
    /// Languages indexed and their chunk counts
    pub languages: std::collections::HashMap<Language, usize>,
}
