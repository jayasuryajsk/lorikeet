use std::path::{Path, PathBuf};
use std::sync::Arc;

use rusqlite::{params, Connection, OptionalExtension};
use tokio::sync::Mutex;

use crate::memory::types::{MatchKind, Memory, MemoryScope, MemorySource, MemoryType, ScoredMemory};
use crate::semantic_search::embedder::Embedder;

const DB_FILENAME: &str = "memories.db";

pub struct MemoryStore {
    conn: Arc<Mutex<Connection>>,
    #[allow(dead_code)]
    db_path: PathBuf,
    project_id: String,
    embedder: Option<Arc<Embedder>>,
}

impl MemoryStore {
    pub async fn init(project_root: &Path) -> anyhow::Result<Self> {
        let dir = project_root.join(".lorikeet").join("memory");
        tokio::fs::create_dir_all(&dir).await?;
        let db_path = dir.join(DB_FILENAME);

        let conn = Connection::open(&db_path)?;
        init_schema(&conn)?;

        // Best-effort embedding init. If model can't load, we fall back to keyword search.
        let embedder = match Embedder::new() {
            Ok(e) => Some(Arc::new(e)),
            Err(_) => None,
        };

        let project_id = project_id(project_root);

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path,
            project_id,
            embedder,
        })
    }

    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    pub async fn insert(&self, memory: &Memory) -> anyhow::Result<()> {
        let embedding: Option<Vec<u8>> = match (&self.embedder, &memory.content) {
            (Some(emb), content) => {
                let vec = emb.embed(content).ok();
                vec.map(|v| embedding_to_bytes(&v))
            }
            _ => None,
        };

        let conn = self.conn.lock().await;
        conn.execute(
            r#"
            INSERT INTO memories (
                id, project_id, scope, type, content, why, context, tags,
                source, confidence, importance, use_count,
                created_at, last_used, source_file, embedding
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
            "#,
            params![
                memory.id,
                memory.project_id,
                memory.scope.as_str(),
                memory.memory_type.as_str(),
                memory.content,
                memory.why,
                memory.context,
                memory.tags.join(","),
                memory.source.as_str(),
                memory.confidence,
                memory.importance,
                memory.use_count as i64,
                memory.created_at,
                memory.last_used,
                memory
                    .source_file
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                embedding,
            ],
        )?;
        Ok(())
    }

    pub async fn delete(&self, id: &str) -> anyhow::Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        Ok(n > 0)
    }

    pub async fn list(&self, limit: usize, type_filter: Option<MemoryType>) -> anyhow::Result<Vec<Memory>> {
        let conn = self.conn.lock().await;
        if let Some(t) = type_filter {
            let mut stmt = conn.prepare(
                "SELECT * FROM memories WHERE project_id = ?1 AND type = ?2 ORDER BY importance DESC, last_used DESC LIMIT ?3",
            )?;
            let rows = stmt.query_map(params![self.project_id, t.as_str(), limit as i64], |row| {
                row_to_memory(row)
            })?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        } else {
            let mut stmt = conn.prepare(
                "SELECT * FROM memories WHERE project_id = ?1 ORDER BY importance DESC, last_used DESC LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![self.project_id, limit as i64], |row| row_to_memory(row))?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        }
    }

    pub async fn mark_used(&self, ids: &[String]) -> anyhow::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let now = crate::memory::types::unix_ts();
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        for id in ids {
            tx.execute(
                "UPDATE memories SET use_count = use_count + 1, last_used = ?2 WHERE id = ?1",
                params![id, now],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        type_filter: Option<Vec<MemoryType>>,
    ) -> anyhow::Result<Vec<ScoredMemory>> {
        // Strategy:
        // - If embeddings are available: semantic topK + keyword fallback.
        // - Otherwise: keyword only.
        let semantic = self.search_semantic(query, limit * 2, type_filter.clone()).await.unwrap_or_default();
        let keyword = self.search_keyword(query, limit * 2, type_filter.clone()).await?;

        // Merge by id, keep best score.
        use std::collections::HashMap;
        let mut merged: HashMap<String, ScoredMemory> = HashMap::new();
        for sm in keyword.into_iter().chain(semantic.into_iter()) {
            merged
                .entry(sm.memory.id.clone())
                .and_modify(|existing| {
                    if sm.score > existing.score {
                        *existing = sm.clone();
                    }
                })
                .or_insert(sm);
        }

        let mut out: Vec<ScoredMemory> = merged.into_values().collect();
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(limit);
        Ok(out)
    }

    async fn search_keyword(
        &self,
        query: &str,
        limit: usize,
        type_filter: Option<Vec<MemoryType>>,
    ) -> anyhow::Result<Vec<ScoredMemory>> {
        let q = format!("%{}%", query);
        let conn = self.conn.lock().await;

        let (sql, params_vec): (String, Vec<rusqlite::types::Value>) = if let Some(types) = type_filter {
            let mut sql = String::from(
                "SELECT * FROM memories WHERE project_id = ?1 AND (content LIKE ?2 OR why LIKE ?2 OR context LIKE ?2)",
            );
            sql.push_str(" AND type IN (");
            for i in 0..types.len() {
                if i > 0 {
                    sql.push(',');
                }
                sql.push_str(&format!("?{}", 3 + i));
            }
            sql.push_str(") ORDER BY importance DESC, last_used DESC LIMIT ?X");
            // We'll replace ?X with correct index below.
            let limit_idx = 3 + types.len();
            sql = sql.replace("?X", &format!("?{}", limit_idx));

            let mut p: Vec<rusqlite::types::Value> = Vec::new();
            p.push(self.project_id.clone().into());
            p.push(q.clone().into());
            for t in types {
                p.push(t.as_str().to_string().into());
            }
            p.push((limit as i64).into());
            (sql, p)
        } else {
            (
                "SELECT * FROM memories WHERE project_id = ?1 AND (content LIKE ?2 OR why LIKE ?2 OR context LIKE ?2) ORDER BY importance DESC, last_used DESC LIMIT ?3".to_string(),
                vec![self.project_id.clone().into(), q.clone().into(), (limit as i64).into()],
            )
        };

        let mut stmt = conn.prepare(&sql)?;
        let mapped = stmt
            .query_map(rusqlite::params_from_iter(params_vec), |row| {
                let m = row_to_memory(row)?;
                Ok(ScoredMemory {
                    memory: m,
                    score: 0.5,
                    match_kind: MatchKind::Keyword,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(mapped)
    }

    async fn search_semantic(
        &self,
        query: &str,
        limit: usize,
        type_filter: Option<Vec<MemoryType>>,
    ) -> anyhow::Result<Vec<ScoredMemory>> {
        let Some(embedder) = &self.embedder else {
            return Ok(Vec::new());
        };

        let query_vec = match embedder.embed_query(query) {
            Ok(v) => v,
            Err(_) => return Ok(Vec::new()),
        };

        let conn = self.conn.lock().await;

        let (sql, params_vec): (String, Vec<rusqlite::types::Value>) = if let Some(types) = type_filter {
            let mut sql = String::from("SELECT id, embedding FROM memories WHERE project_id = ?1 AND embedding IS NOT NULL");
            sql.push_str(" AND type IN (");
            for i in 0..types.len() {
                if i > 0 {
                    sql.push(',');
                }
                sql.push_str(&format!("?{}", 2 + i));
            }
            sql.push(')');
            let mut p: Vec<rusqlite::types::Value> = Vec::new();
            p.push(self.project_id.clone().into());
            for t in types {
                p.push(t.as_str().to_string().into());
            }
            (sql, p)
        } else {
            (
                "SELECT id, embedding FROM memories WHERE project_id = ?1 AND embedding IS NOT NULL".to_string(),
                vec![self.project_id.clone().into()],
            )
        };

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params_vec), |row| {
                let id: String = row.get(0)?;
                let bytes: Vec<u8> = row.get(1)?;
                Ok((id, bytes))
            })?
            .filter_map(|r| r.ok())
            .collect::<Vec<_>>();

        // Compute cosine similarity in-process. This is fine for small-medium stores.
        let mut scored: Vec<(String, f32)> = Vec::new();
        for (id, bytes) in rows {
            if let Some(vec) = bytes_to_embedding(&bytes) {
                let score = cosine_similarity(&query_vec, &vec);
                scored.push((id, score));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        let mut out = Vec::new();
        for (id, score) in scored {
            if let Some(m) = get_by_id_locked(&conn, &id)? {
                out.push(ScoredMemory {
                    memory: m,
                    score,
                    match_kind: MatchKind::Semantic,
                });
            }
        }
        Ok(out)
    }
}

fn init_schema(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            scope TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            why TEXT,
            context TEXT,
            tags TEXT,
            source TEXT NOT NULL,
            confidence REAL NOT NULL,
            importance REAL NOT NULL,
            use_count INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            last_used INTEGER NOT NULL,
            source_file TEXT,
            embedding BLOB
        );

        CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id);
        CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
        CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
        CREATE INDEX IF NOT EXISTS idx_memories_last_used ON memories(last_used);
        "#,
    )?;
    Ok(())
}

fn row_to_memory(row: &rusqlite::Row<'_>) -> rusqlite::Result<Memory> {
    let id: String = row.get("id")?;
    let project_id: Option<String> = row.get("project_id")?;
    let scope: String = row.get("scope")?;
    let mem_type: String = row.get("type")?;
    let content: String = row.get("content")?;
    let why: Option<String> = row.get("why")?;
    let context: Option<String> = row.get("context")?;
    let tags_str: Option<String> = row.get("tags")?;
    let source: String = row.get("source")?;
    let confidence: f32 = row.get("confidence")?;
    let importance: f32 = row.get("importance")?;
    let use_count: i64 = row.get("use_count")?;
    let created_at: i64 = row.get("created_at")?;
    let last_used: i64 = row.get("last_used")?;
    let source_file: Option<String> = row.get("source_file")?;

    let memory_type: MemoryType = mem_type.parse().map_err(|_| rusqlite::Error::InvalidQuery)?;
    let scope: MemoryScope = scope.parse().map_err(|_| rusqlite::Error::InvalidQuery)?;
    let source: MemorySource = match source.as_str() {
        "tool" => MemorySource::Tool,
        "llm" => MemorySource::Llm,
        _ => MemorySource::User,
    };

    let mut memory = Memory::new(id, project_id, scope, memory_type, content);
    memory.why = why;
    memory.context = context;
    memory.tags = tags_str
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    memory.source = source;
    memory.confidence = confidence;
    memory.importance = importance;
    memory.use_count = use_count.max(0) as u64;
    memory.created_at = created_at;
    memory.last_used = last_used;
    memory.source_file = source_file.map(PathBuf::from);
    Ok(memory)
}

fn get_by_id_locked(conn: &Connection, id: &str) -> rusqlite::Result<Option<Memory>> {
    conn.query_row(
        "SELECT * FROM memories WHERE id = ?1",
        params![id],
        |row| row_to_memory(row),
    )
    .optional()
}

fn embedding_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn bytes_to_embedding(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().ok()?;
        out.push(f32::from_le_bytes(arr));
    }
    Some(out)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

fn project_id(root: &Path) -> String {
    // Stable-ish per project. Using the canonical path is enough for local usage.
    std::fs::canonicalize(root)
        .unwrap_or_else(|_| root.to_path_buf())
        .to_string_lossy()
        .to_string()
}
