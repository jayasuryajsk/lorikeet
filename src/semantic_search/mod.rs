pub mod chunker;
pub mod embedder;
pub mod index;
pub mod types;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use walkdir::WalkDir;

use chunker::Chunker;
use embedder::{Embedder, EmbedderError};
use index::{IndexError, VectorIndex};
use types::{CodeChunk, IndexStats, Language, SearchResult};

// Re-export key types for external use
pub use types::SearchConfig;

/// Semantic search engine for code
pub struct SemanticSearch {
    embedder: Embedder,
    index: Arc<VectorIndex>,
    chunker: Chunker,
    config: SearchConfig,
    project_root: RwLock<Option<PathBuf>>,
}

impl SemanticSearch {
    /// Create a new semantic search engine
    pub fn new(config: SearchConfig) -> Result<Self, SemanticSearchError> {
        let embedder = Embedder::new().map_err(SemanticSearchError::Embedder)?;
        let index = VectorIndex::new(&config.index_dir, embedder.dimension())
            .map_err(SemanticSearchError::Index)?;
        let chunker = Chunker::new(config.max_chunk_size);

        Ok(Self {
            embedder,
            index: Arc::new(index),
            chunker,
            config,
            project_root: RwLock::new(None),
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self, SemanticSearchError> {
        Self::new(SearchConfig::default())
    }

    /// Set the project root directory
    pub fn set_project_root(&self, path: PathBuf) {
        let mut root = self.project_root.write();
        *root = Some(path);
    }

    /// Get the current project root
    pub fn project_root(&self) -> Option<PathBuf> {
        self.project_root.read().clone()
    }

    /// Search for code similar to the query
    pub fn search(&self, query: &str) -> Result<Vec<SearchResult>, SemanticSearchError> {
        self.search_with_options(query, self.config.top_k, self.config.min_score)
    }

    /// Search with custom options
    pub fn search_with_options(
        &self,
        query: &str,
        top_k: usize,
        min_score: f32,
    ) -> Result<Vec<SearchResult>, SemanticSearchError> {
        // Check if index is empty and we have a project root - auto-index
        if self.index.is_empty() {
            if let Some(root) = self.project_root() {
                self.index_directory(&root)?;
            }
        }

        // If still empty, return empty results
        if self.index.is_empty() {
            return Ok(vec![]);
        }

        // Generate query embedding
        let query_embedding = self
            .embedder
            .embed_query(query)
            .map_err(SemanticSearchError::Embedder)?;

        // Search the index
        let results = self
            .index
            .search(&query_embedding, top_k)
            .map_err(SemanticSearchError::Index)?;

        // Convert to SearchResults with full chunk data
        let mut search_results = Vec::new();
        for (id, score) in results {
            if score < min_score {
                continue;
            }

            if let Some(metadata) = self.index.get_metadata(id) {
                // Read the actual content from the file
                let content = self.read_chunk_content(&metadata)?;

                search_results.push(SearchResult {
                    chunk: CodeChunk {
                        id,
                        content,
                        metadata,
                    },
                    score,
                });
            }
        }

        Ok(search_results)
    }

    /// Read the content of a chunk from its source file
    fn read_chunk_content(
        &self,
        metadata: &types::ChunkMetadata,
    ) -> Result<String, SemanticSearchError> {
        let file_path = if metadata.file_path.is_absolute() {
            metadata.file_path.clone()
        } else if let Some(root) = self.project_root() {
            root.join(&metadata.file_path)
        } else {
            metadata.file_path.clone()
        };

        let bytes = std::fs::read(&file_path)
            .map_err(|e| SemanticSearchError::Io(e.to_string()))?;
        let content = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return Ok(String::new()), // Skip non-UTF8 (likely binary)
        };

        // Extract the specific lines
        let lines: Vec<&str> = content.lines().collect();
        let start = metadata.start_line.saturating_sub(1);
        let end = metadata.end_line.min(lines.len());

        if start >= lines.len() {
            return Ok(String::new());
        }

        Ok(lines[start..end].join("\n"))
    }

    /// Index a directory
    pub fn index_directory(&self, dir: &Path) -> Result<IndexStats, SemanticSearchError> {
        // Clear existing index
        self.index.clear().map_err(SemanticSearchError::Index)?;

        // Set project root
        self.set_project_root(dir.to_path_buf());

        // Collect all files to index
        let files = self.collect_files(dir)?;

        // Process files and collect chunks
        let mut all_chunks = Vec::new();
        for file_path in &files {
            let chunks = self.process_file(file_path)?;
            all_chunks.extend(chunks);
        }

        // Generate embeddings in batches
        let batch_size = 32;
        let mut indexed_count = 0;

        for chunk_batch in all_chunks.chunks(batch_size) {
            let texts: Vec<String> = chunk_batch
                .iter()
                .map(|c| {
                    // Include file path context for better embeddings
                    format!(
                        "File: {}\n\n{}",
                        c.metadata.file_path.display(),
                        c.content
                    )
                })
                .collect();

            let embeddings = match self.embedder.embed_batch(&texts) {
                Ok(e) => e,
                Err(_) => continue, // Skip this batch on embedding error
            };

            // Add to index - handle individual chunk errors gracefully
            for (chunk, embedding) in chunk_batch.iter().zip(embeddings.iter()) {
                if self.index.add(chunk, embedding).is_ok() {
                    indexed_count += 1;
                }
            }
        }

        // Only fail if we couldn't index anything at all
        if indexed_count == 0 && !all_chunks.is_empty() {
            return Err(SemanticSearchError::Io(
                "Failed to index any chunks".to_string(),
            ));
        }

        // Save the index
        self.index.save().map_err(SemanticSearchError::Index)?;

        Ok(self.index.stats())
    }

    /// Collect files to index from a directory
    fn collect_files(&self, dir: &Path) -> Result<Vec<PathBuf>, SemanticSearchError> {
        let mut files = Vec::new();

        for entry in WalkDir::new(dir)
            .follow_links(true)
            .into_iter()
            .filter_entry(|e| !self.should_exclude(e.path()))
        {
            let entry = entry.map_err(|e| SemanticSearchError::Io(e.to_string()))?;

            if entry.file_type().is_file() && self.should_include(entry.path()) {
                files.push(entry.path().to_path_buf());
            }
        }

        Ok(files)
    }

    /// Check if a path should be excluded
    fn should_exclude(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        for pattern in &self.config.exclude_patterns {
            if let Ok(glob_pattern) = glob::Pattern::new(pattern) {
                if glob_pattern.matches(&path_str) {
                    return true;
                }
            }

            // Also check just the path components
            for component in path.components() {
                if let std::path::Component::Normal(name) = component {
                    let name_str = name.to_string_lossy();
                    // Check for common excluded directories
                    if matches!(
                        name_str.as_ref(),
                        "target" | "node_modules" | ".git" | "dist" | "build" | "__pycache__" | "vendor"
                    ) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Check if a file should be included
    fn should_include(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        for pattern in &self.config.exclude_patterns {
            if let Ok(glob_pattern) = glob::Pattern::new(pattern) {
                if glob_pattern.matches(&path_str) {
                    return false;
                }
            }
        }

        true
    }

    /// Process a single file into chunks
    fn process_file(&self, file_path: &Path) -> Result<Vec<CodeChunk>, SemanticSearchError> {
        let bytes = std::fs::read(file_path).map_err(|e| SemanticSearchError::Io(e.to_string()))?;
        let content = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return Ok(Vec::new()), // Skip non-UTF8 (likely binary) files
        };

        let ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let language = Language::from_extension(ext);

        // Make path relative to project root
        let relative_path = if let Some(root) = self.project_root() {
            file_path
                .strip_prefix(&root)
                .unwrap_or(file_path)
                .to_path_buf()
        } else {
            file_path.to_path_buf()
        };

        Ok(self.chunker.chunk_file(&content, &relative_path, language))
    }

    /// Get index statistics
    #[allow(dead_code)]
    pub fn stats(&self) -> IndexStats {
        self.index.stats()
    }

    /// Check if the index is empty
    #[allow(dead_code)]
    pub fn is_indexed(&self) -> bool {
        !self.index.is_empty()
    }

    /// Force re-index
    #[allow(dead_code)]
    pub fn reindex(&self) -> Result<IndexStats, SemanticSearchError> {
        if let Some(root) = self.project_root() {
            self.index_directory(&root)
        } else {
            Err(SemanticSearchError::NoProjectRoot)
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum SemanticSearchError {
    Embedder(EmbedderError),
    Index(IndexError),
    Io(String),
    NoProjectRoot,
}

impl std::fmt::Display for SemanticSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SemanticSearchError::Embedder(e) => write!(f, "Embedder error: {}", e),
            SemanticSearchError::Index(e) => write!(f, "Index error: {}", e),
            SemanticSearchError::Io(msg) => write!(f, "IO error: {}", msg),
            SemanticSearchError::NoProjectRoot => write!(f, "No project root set"),
        }
    }
}

impl std::error::Error for SemanticSearchError {}

/// Format search results for display
pub fn format_search_results(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No results found.".to_string();
    }

    let mut output = String::new();
    for (i, result) in results.iter().enumerate() {
        let meta = &result.chunk.metadata;
        let symbol_info = match (&meta.symbol_name, &meta.symbol_type) {
            (Some(name), Some(stype)) => format!(" ({:?}: {})", stype, name),
            (Some(name), None) => format!(" ({})", name),
            _ => String::new(),
        };

        output.push_str(&format!(
            "{}. {}:{}-{}{} [score: {:.2}]\n",
            i + 1,
            meta.file_path.display(),
            meta.start_line,
            meta.end_line,
            symbol_info,
            result.score
        ));

        // Show a preview of the content
        let preview: String = result
            .chunk
            .content
            .lines()
            .take(5)
            .collect::<Vec<_>>()
            .join("\n");
        output.push_str(&format!("   {}\n", preview.replace('\n', "\n   ")));

        if result.chunk.content.lines().count() > 5 {
            output.push_str("   ...\n");
        }
        output.push('\n');
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download
    fn test_semantic_search_creation() {
        let config = SearchConfig {
            index_dir: PathBuf::from("/tmp/test_index"),
            ..Default::default()
        };
        let _search = SemanticSearch::new(config).unwrap();
    }
}
