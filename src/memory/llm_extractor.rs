use serde::Deserialize;

use crate::llm::{call_llm_nonstream, ChatMessage, LlmProvider};
use crate::memory::types::MemoryType;

#[derive(Debug, Clone)]
pub struct ExtractedMemoryCandidate {
    pub memory_type: MemoryType,
    pub content: String,
    pub why: Option<String>,
    pub context: Option<String>,
    pub tags: Vec<String>,
    pub confidence: Option<f32>,
    pub importance: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExtractionResponse {
    #[serde(default)]
    memories: Vec<ExtractedMemoryRaw>,
}

#[derive(Debug, Clone, Deserialize)]
struct ExtractedMemoryRaw {
    #[serde(rename = "type")]
    memory_type: String,
    content: String,
    #[serde(default)]
    why: Option<String>,
    #[serde(default)]
    context: Option<String>,
    #[serde(default)]
    tags: Option<Vec<String>>,
    #[serde(default)]
    confidence: Option<f32>,
    #[serde(default)]
    importance: Option<f32>,
}

const EXTRACTION_SYSTEM_PROMPT: &str = r#"You are a memory extraction assistant for a coding agent.

Goal: extract a small number of high-signal, long-lived memories from a single conversation turn.

Memory types (choose one per memory):
- mistake: a specific mistake and how to avoid it next time
- avoid: a hard constraint / thing to not do again
- preference: a stable user preference (style, structure, UX expectations)
- decision: an architectural or implementation decision and why
- fact: stable project knowledge (stack, conventions, constraints)

Rules:
- Output ONLY valid JSON.
- Return an object: {"memories": [ ... ]}.
- Each memory must be concise (1-2 sentences) and actionable.
- Do not include secrets, API keys, tokens, or personal data.
- Skip trivial / one-off task instructions ("do it", "ship it") unless it is a stable preference.
- Max 5 memories.

Schema:
{"memories": [{"type": "mistake|avoid|preference|decision|fact", "content": "...", "why": "...", "context": "...", "tags": ["..."], "confidence": 0.0-1.0, "importance": 0.0-1.0}]}
"#;

pub async fn extract_memories_from_turn(
    provider: LlmProvider,
    api_key: String,
    model: String,
    turn_summary: String,
) -> anyhow::Result<Vec<ExtractedMemoryCandidate>> {
    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: Some(EXTRACTION_SYSTEM_PROMPT.to_string()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        ChatMessage {
            role: "user".into(),
            content: Some(format!(
                "Extract memories from this turn summary:\n\n{}",
                turn_summary
            )),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ];

    let raw = call_llm_nonstream(provider, api_key, model, messages)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    let json_str = extract_json_object(&raw)
        .ok_or_else(|| anyhow::anyhow!("Memory extraction did not return valid JSON"))?;

    let parsed: ExtractionResponse = serde_json::from_str(&json_str)?;

    let mut out: Vec<ExtractedMemoryCandidate> = Vec::new();
    for m in parsed.memories {
        let memory_type: MemoryType = match m.memory_type.parse() {
            Ok(t) => t,
            Err(_) => continue,
        };

        let content = m.content.trim().to_string();
        if content.len() < 8 {
            continue;
        }

        out.push(ExtractedMemoryCandidate {
            memory_type,
            content,
            why: m
                .why
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            context: m
                .context
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
            tags: m
                .tags
                .unwrap_or_default()
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            confidence: m.confidence,
            importance: m.importance,
        });

        if out.len() >= 5 {
            break;
        }
    }

    Ok(out)
}

fn extract_json_object(s: &str) -> Option<String> {
    let mut t = s.trim().to_string();

    // Strip ```json fences if present.
    if t.starts_with("```") {
        if let Some(nl) = t.find('\n') {
            t = t[nl + 1..].to_string();
        }
        if let Some(end) = t.rfind("```") {
            t = t[..end].to_string();
        }
        t = t.trim().to_string();
    }

    let start = t.find('{')?;
    let end = t.rfind('}')?;
    if end <= start {
        return None;
    }

    Some(t[start..=end].to_string())
}
