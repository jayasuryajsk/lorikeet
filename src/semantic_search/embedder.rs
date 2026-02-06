use std::sync::Arc;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use parking_lot::RwLock;

/// Wrapper around fastembed for generating text embeddings
pub struct Embedder {
    model: Arc<RwLock<TextEmbedding>>,
}

impl Embedder {
    /// Create a new embedder with the AllMiniLML6V2 model
    /// This will download the model on first use (~22MB)
    pub fn new() -> Result<Self, EmbedderError> {
        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::AllMiniLML6V2))
            .map_err(|e| EmbedderError::ModelInit(e.to_string()))?;

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
        })
    }

    /// Get the embedding dimension for this model
    pub fn dimension(&self) -> usize {
        384 // AllMiniLML6V2 produces 384-dimensional embeddings
    }

    /// Generate embeddings for a batch of texts
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let model = self.model.read();
        model
            .embed(texts.to_vec(), None)
            .map_err(|e| EmbedderError::Embedding(e.to_string()))
    }

    /// Generate embedding for a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        let embeddings = self.embed_batch(&[text.to_string()])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::Embedding("No embedding returned".into()))
    }

    /// Generate embeddings for code chunks, adding context
    #[allow(dead_code)]
    pub fn embed_code(&self, code: &str, file_path: Option<&str>) -> Result<Vec<f32>, EmbedderError> {
        // Add file path context to improve search relevance
        let text = match file_path {
            Some(path) => format!("File: {}\n\n{}", path, code),
            None => code.to_string(),
        };
        self.embed(&text)
    }

    /// Generate embedding for a search query
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedderError> {
        // For search queries, we might want to add query-specific formatting
        // but AllMiniLML6V2 works well with raw queries
        self.embed(query)
    }
}

impl Clone for Embedder {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
        }
    }
}

#[derive(Debug)]
pub enum EmbedderError {
    ModelInit(String),
    Embedding(String),
}

impl std::fmt::Display for EmbedderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedderError::ModelInit(msg) => write!(f, "Failed to initialize embedding model: {}", msg),
            EmbedderError::Embedding(msg) => write!(f, "Embedding error: {}", msg),
        }
    }
}

impl std::error::Error for EmbedderError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download
    fn test_embedder_creation() {
        let embedder = Embedder::new().unwrap();
        assert_eq!(embedder.dimension(), 384);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_embedding() {
        let embedder = Embedder::new().unwrap();
        let embedding = embedder.embed("Hello, world!").unwrap();
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_batch_embedding() {
        let embedder = Embedder::new().unwrap();
        let texts = vec!["Hello".to_string(), "World".to_string()];
        let embeddings = embedder.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
    }
}
