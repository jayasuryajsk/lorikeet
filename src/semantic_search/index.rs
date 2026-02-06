use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::semantic_search::types::{ChunkMetadata, CodeChunk, IndexStats};

const INDEX_FILE: &str = "index.bin";

/// Serializable index data
#[derive(Serialize, Deserialize)]
struct IndexData {
    vectors: Vec<(u64, Vec<f32>)>,
    metadata: HashMap<u64, ChunkMetadata>,
}

/// Vector index using usearch (HNSW algorithm)
pub struct VectorIndex {
    index: RwLock<Index>,
    metadata: RwLock<HashMap<u64, ChunkMetadata>>,
    vectors: RwLock<HashMap<u64, Vec<f32>>>, // Store vectors for persistence
    index_dir: PathBuf,
    dimension: usize,
    next_id: RwLock<u64>,
}

impl VectorIndex {
    /// Create a new vector index or load an existing one
    pub fn new(index_dir: &Path, dimension: usize) -> Result<Self, IndexError> {
        fs::create_dir_all(index_dir).map_err(|e| IndexError::Io(e.to_string()))?;

        let index_path = index_dir.join(INDEX_FILE);

        let options = IndexOptions {
            dimensions: dimension,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 16,     // HNSW M parameter
            expansion_add: 128,   // ef_construction
            expansion_search: 64, // ef_search
            ..Default::default()
        };

        let index = Index::new(&options).map_err(|e| IndexError::Index(e.to_string()))?;

        // Pre-reserve capacity to avoid reallocation issues
        index
            .reserve(10000)
            .map_err(|e| IndexError::Index(e.to_string()))?;

        let vi = Self {
            index: RwLock::new(index),
            metadata: RwLock::new(HashMap::new()),
            vectors: RwLock::new(HashMap::new()),
            index_dir: index_dir.to_path_buf(),
            dimension,
            next_id: RwLock::new(0),
        };

        // Load existing index if present
        if index_path.exists() {
            vi.load()?;
        }

        Ok(vi)
    }

    /// Add a chunk with its embedding to the index
    pub fn add(&self, chunk: &CodeChunk, embedding: &[f32]) -> Result<u64, IndexError> {
        if embedding.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                got: embedding.len(),
            });
        }

        let id = {
            let mut next_id = self.next_id.write();
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Try to add to the usearch index, expand capacity if needed
        {
            let index = self.index.write();
            let current_size = index.size();
            let current_capacity = index.capacity();

            // Ensure we have capacity
            if current_size >= current_capacity {
                // Try to reserve more space
                if let Err(e) = index.reserve(current_capacity + 5000) {
                    return Err(IndexError::Index(format!("reserve failed: {}", e)));
                }
            }

            // Add the vector - this is the operation that can fail with usearch
            if let Err(e) = index.add(id, embedding) {
                return Err(IndexError::Index(format!("add failed: {}", e)));
            }
        }

        // Only update metadata and vectors if usearch add succeeded
        {
            let mut metadata = self.metadata.write();
            metadata.insert(id, chunk.metadata.clone());
        }

        {
            let mut vectors = self.vectors.write();
            vectors.insert(id, embedding.to_vec());
        }

        Ok(id)
    }

    /// Get all stored vectors (for persistence)
    fn get_all_vectors(&self) -> Vec<(u64, Vec<f32>)> {
        let vectors = self.vectors.read();
        vectors.iter().map(|(k, v)| (*k, v.clone())).collect()
    }

    /// Add multiple chunks with their embeddings in batch
    #[allow(dead_code)]
    pub fn add_batch(
        &self,
        chunks: &[CodeChunk],
        embeddings: &[Vec<f32>],
    ) -> Result<Vec<u64>, IndexError> {
        if chunks.len() != embeddings.len() {
            return Err(IndexError::BatchSizeMismatch {
                chunks: chunks.len(),
                embeddings: embeddings.len(),
            });
        }

        let mut ids = Vec::with_capacity(chunks.len());

        // Reserve capacity
        {
            let index = self.index.write();
            let needed = index.size() + chunks.len();
            if needed > index.capacity() {
                index
                    .reserve(needed + 1000)
                    .map_err(|e| IndexError::Index(e.to_string()))?;
            }
        }

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            let id = self.add(chunk, embedding)?;
            ids.push(id);
        }

        Ok(ids)
    }

    /// Search for similar vectors
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, IndexError> {
        if query_embedding.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                got: query_embedding.len(),
            });
        }

        let index = self.index.read();
        if index.size() == 0 {
            return Ok(vec![]);
        }

        let results = index
            .search(query_embedding, top_k)
            .map_err(|e| IndexError::Index(e.to_string()))?;

        Ok(results
            .keys
            .into_iter()
            .zip(results.distances.into_iter())
            .map(|(key, distance)| {
                // Convert distance to similarity score (cosine distance to cosine similarity)
                let similarity = 1.0 - distance;
                (key, similarity)
            })
            .collect())
    }

    /// Get metadata for a chunk by ID
    pub fn get_metadata(&self, id: u64) -> Option<ChunkMetadata> {
        let metadata = self.metadata.read();
        metadata.get(&id).cloned()
    }

    /// Save the index to disk
    pub fn save(&self) -> Result<(), IndexError> {
        let index_path = self.index_dir.join(INDEX_FILE);

        let metadata = self.metadata.read();
        let data = IndexData {
            vectors: self.get_all_vectors(),
            metadata: metadata.clone(),
        };

        let serialized =
            bincode::serialize(&data).map_err(|e| IndexError::Serialization(e.to_string()))?;
        fs::write(&index_path, serialized).map_err(|e| IndexError::Io(e.to_string()))?;

        Ok(())
    }

    /// Load the index from disk
    pub fn load(&self) -> Result<(), IndexError> {
        let index_path = self.index_dir.join(INDEX_FILE);

        if index_path.exists() {
            let data = fs::read(&index_path).map_err(|e| IndexError::Io(e.to_string()))?;
            let index_data: IndexData = bincode::deserialize(&data)
                .map_err(|e| IndexError::Serialization(e.to_string()))?;

            // Rebuild the usearch index from stored vectors
            let mut loaded_count = 0;
            {
                let index = self.index.write();
                if !index_data.vectors.is_empty() {
                    // Reserve with some extra capacity
                    let _ = index.reserve(index_data.vectors.len() + 1000);
                }
                for (id, vector) in &index_data.vectors {
                    // Skip vectors that fail to load
                    if index.add(*id, vector).is_ok() {
                        loaded_count += 1;
                    }
                }
            }

            // Only restore metadata for successfully loaded vectors
            if loaded_count > 0 {
                // Restore metadata
                {
                    let mut metadata = self.metadata.write();
                    *metadata = index_data.metadata;
                }

                // Restore vectors
                {
                    let mut vectors = self.vectors.write();
                    *vectors = index_data.vectors.iter().cloned().collect();
                }

                // Update next_id
                {
                    let mut next_id = self.next_id.write();
                    *next_id = index_data
                        .vectors
                        .iter()
                        .map(|(id, _)| *id)
                        .max()
                        .map(|id| id + 1)
                        .unwrap_or(0);
                }
            }
        }

        Ok(())
    }

    /// Clear the index
    pub fn clear(&self) -> Result<(), IndexError> {
        {
            let index = self.index.write();
            let _ = index.reset();
        }

        {
            let mut metadata = self.metadata.write();
            metadata.clear();
        }

        {
            let mut vectors = self.vectors.write();
            vectors.clear();
        }

        {
            let mut next_id = self.next_id.write();
            *next_id = 0;
        }

        // Remove index file
        let index_path = self.index_dir.join(INDEX_FILE);
        if index_path.exists() {
            fs::remove_file(&index_path).map_err(|e| IndexError::Io(e.to_string()))?;
        }

        Ok(())
    }

    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        let index = self.index.read();
        let metadata = self.metadata.read();

        let mut languages = std::collections::HashMap::new();
        let mut files = std::collections::HashSet::new();

        for meta in metadata.values() {
            *languages.entry(meta.language).or_insert(0) += 1;
            files.insert(&meta.file_path);
        }

        let index_size = {
            let index_path = self.index_dir.join(INDEX_FILE);
            fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0)
        };

        IndexStats {
            total_chunks: index.size(),
            total_files: files.len(),
            index_size_bytes: index_size,
            languages,
        }
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        let index = self.index.read();
        index.size() == 0
    }

    /// Get the number of vectors in the index
    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        let index = self.index.read();
        index.size()
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum IndexError {
    Io(String),
    Index(String),
    Serialization(String),
    DimensionMismatch { expected: usize, got: usize },
    BatchSizeMismatch { chunks: usize, embeddings: usize },
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::Io(msg) => write!(f, "IO error: {}", msg),
            IndexError::Index(msg) => write!(f, "Index error: {}", msg),
            IndexError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            IndexError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            IndexError::BatchSizeMismatch { chunks, embeddings } => {
                write!(
                    f,
                    "Batch size mismatch: {} chunks, {} embeddings",
                    chunks, embeddings
                )
            }
        }
    }
}

impl std::error::Error for IndexError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_search::types::Language;
    use tempfile::TempDir;

    #[test]
    fn test_index_creation() {
        let temp_dir = TempDir::new().unwrap();
        let index = VectorIndex::new(temp_dir.path(), 384).unwrap();
        assert!(index.is_empty());
    }

    #[test]
    fn test_add_and_search() {
        let temp_dir = TempDir::new().unwrap();
        let index = VectorIndex::new(temp_dir.path(), 4).unwrap();

        let chunk = CodeChunk {
            id: 0,
            content: "test content".to_string(),
            metadata: ChunkMetadata {
                file_path: PathBuf::from("test.rs"),
                start_line: 1,
                end_line: 5,
                language: Language::Rust,
                symbol_name: Some("test_fn".to_string()),
                symbol_type: None,
            },
        };

        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let id = index.add(&chunk, &embedding).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);

        let metadata = index.get_metadata(id).unwrap();
        assert_eq!(metadata.file_path, PathBuf::from("test.rs"));
    }

    #[test]
    fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();

        // Create and populate index
        {
            let index = VectorIndex::new(temp_dir.path(), 4).unwrap();
            let chunk = CodeChunk {
                id: 0,
                content: "test".to_string(),
                metadata: ChunkMetadata {
                    file_path: PathBuf::from("test.rs"),
                    start_line: 1,
                    end_line: 1,
                    language: Language::Rust,
                    symbol_name: None,
                    symbol_type: None,
                },
            };
            index.add(&chunk, &[1.0, 0.0, 0.0, 0.0]).unwrap();
            index.save().unwrap();
        }

        // Load and verify
        {
            let index = VectorIndex::new(temp_dir.path(), 4).unwrap();
            assert_eq!(index.size(), 1);
        }
    }
}
