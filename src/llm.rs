use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::events::AppEvent;
use crate::types::{ToolCallFunction, ToolCallMessage};

pub const MODEL: &str = "z-ai/glm-4.7-flash";
const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const OPENAI_URL: &str = "https://api.openai.com/v1/chat/completions";
const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmProvider {
    OpenRouter,
    OpenAI,
    Codex,
}

fn normalize_codex_model(model: &str) -> String {
    // The Codex ChatGPT backend expects model slugs like `gpt-5.2` / `gpt-5.2-codex`,
    // not OpenRouter-style `openai/gpt-5.2`.
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return model.to_string();
    }
    trimmed
        .rsplit('/')
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| model.to_string())
}

fn fallback_codex_model() -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Cache {
        models: Vec<Model>,
    }
    #[derive(serde::Deserialize)]
    struct Model {
        slug: Option<String>,
    }

    let path = dirs::home_dir()?.join(".codex").join("models_cache.json");
    let raw = std::fs::read_to_string(path).ok()?;
    let cache: Cache = serde_json::from_str(&raw).ok()?;
    let mut slugs: Vec<String> = cache
        .models
        .into_iter()
        .filter_map(|m| m.slug)
        .filter(|s| !s.trim().is_empty())
        .collect();

    // Prefer a codex-capable slug if present.
    if let Some(p) = slugs.iter().find(|s| s.contains("gpt-5.2-codex")) {
        return Some(p.clone());
    }
    if let Some(p) = slugs.iter().find(|s| s.contains("codex")) {
        return Some(p.clone());
    }
    slugs.sort();
    slugs.first().cloned()
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    content: Option<String>,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
struct DeltaToolCall {
    index: usize,
    id: Option<String>,
    function: Option<DeltaFunction>,
}

#[derive(Debug, Clone, Deserialize)]
struct DeltaFunction {
    name: Option<String>,
    arguments: Option<String>,
}

// Tool call being assembled from stream
#[derive(Debug, Clone, Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

fn get_tools() -> Vec<Tool> {
    vec![
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "bash".into(),
                description: "Run a shell command. Use for: reading files (cat), listing dirs (ls, tree), git, builds, tests, installs, etc.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute"
                        }
                    },
                    "required": ["command"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "rg".into(),
                description: "Fast text search using ripgrep. Use for exact symbol or string search across the codebase.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The text or regex to search for"
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory or file to search (defaults to current directory)"
                        },
                        "context": {
                            "type": "integer",
                            "description": "Optional number of context lines to include before/after matches"
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "smart_search".into(),
                description: "Combined search: runs ripgrep (exact) + semantic search (meaning) and returns a merged, ranked list. Prefer this when you don't know exact symbol names.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for (natural language or exact text)"
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional scope within the workspace (directory or file). Defaults to current directory."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max number of results to return (default 20, max 50)"
                        },
                        "rg": {
                            "type": "boolean",
                            "description": "Whether to run rg (default true)"
                        },
                        "semantic": {
                            "type": "boolean",
                            "description": "Whether to run semantic search (default true)"
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "lsp".into(),
                description: "Language Server Protocol bridge for code-aware operations. Actions: definition, references, rename, diagnostics. Provide path + 1-based line/column for symbol operations.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "language": {"type": "string", "description": "auto|rust|typescript (default auto)"},
                        "action": {"type": "string", "description": "definition|references|rename|diagnostics"},
                        "path": {"type": "string", "description": "File path (workspace-relative or absolute)"},
                        "line": {"type": "integer", "description": "1-based line number (required for definition/references/rename)"},
                        "column": {"type": "integer", "description": "1-based column number (required for definition/references/rename)"},
                        "new_name": {"type": "string", "description": "New symbol name (required for rename)"},
                        "include_declaration": {"type": "boolean", "description": "For references: include the declaration (default false)"},
                        "limit": {"type": "integer", "description": "Max results (default 20)"}
                    },
                    "required": ["action","path"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "read_file".into(),
                description: "Read the contents of a file at the given path.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file path to read"
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "write_file".into(),
                description: "Write content to a file at the given path. Creates the file if it doesn't exist, overwrites if it does.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file path to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "list_files".into(),
                description: "List files and directories at the given path.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory path to list (defaults to current directory)"
                        }
                    },
                    "required": []
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "edit_file".into(),
                description: "Make a surgical edit to a file by replacing old_string with new_string. The old_string must match exactly and be unique in the file.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file path to edit"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact text to find and replace (must be unique in file)"
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The text to replace it with"
                        }
                    },
                    "required": ["path", "old_string", "new_string"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "apply_patch".into(),
                description: "Apply a patch to one or more files. Prefer this for non-trivial edits/refactors. Patch format uses *** Begin Patch / *** Update File / *** Add File / *** Delete File / *** End Patch blocks with diff-style lines starting with ' ', '+', '-'.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "patch": {
                            "type": "string",
                            "description": "Patch text to apply"
                        }
                    },
                    "required": ["patch"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "open_at".into(),
                description: "Read a file around a specific 1-based line number with context and line numbers. Use after smart_search/rg when you have path:line results.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "line": {"type": "integer", "description": "1-based line number to center on"},
                        "context": {"type": "integer", "description": "Lines of context before/after (default 40)"}
                    },
                    "required": ["path", "line"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "semantic_search".into(),
                description: "Search code semantically using natural language. Returns ranked results with file paths and line numbers. Use for finding code related to concepts, features, or functionality. Auto-indexes on first use.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what you're looking for (e.g., 'authentication handling', 'database connection', 'error logging')"
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "verify".into(),
                description: "Run the workspace verify command (tests/build). If command is omitted, auto-detects a suggestion. Respects sandbox allow_commands; if blocked, adjust sandbox allowlist.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Optional verify command (e.g. 'cargo test', 'pnpm test')"}
                    },
                    "required": []
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "memory_recall".into(),
                description: "Recall relevant long-term memory for the current project. Use before making decisions or repeating actions. Returns ranked memories.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max memories to return (default 8)"},
                        "types": {"type": "array", "items": {"type": "string"}, "description": "Optional list of memory types"}
                    },
                    "required": ["query"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "memory_save".into(),
                description: "Save a long-term memory. Use for user preferences, decisions, and mistakes. Never store secrets.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "description": "mistake|preference|decision|fact|avoid"},
                        "content": {"type": "string", "description": "The memory content"},
                        "why": {"type": "string", "description": "Why this memory matters / how it should change future behavior"},
                        "context": {"type": "string", "description": "Optional context"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
                        "scope": {"type": "string", "description": "project|global (default project)"},
                        "confidence": {"type": "number", "description": "0..1"},
                        "importance": {"type": "number", "description": "0..1"}
                    },
                    "required": ["type", "content"]
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "memory_list".into(),
                description: "List stored memories for this project.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Max to return (default 30)"},
                        "type": {"type": "string", "description": "Optional memory type filter"}
                    }
                }),
            },
        },
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "memory_forget".into(),
                description: "Delete a memory by id.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Memory id"}
                    },
                    "required": ["id"]
                }),
            },
        },
    ]
}

pub async fn call_llm(
    tx: mpsc::UnboundedSender<AppEvent>,
    provider: LlmProvider,
    api_key: String,
    codex_account_id: Option<String>,
    model: String,
    messages: Vec<ChatMessage>,
    tools_enabled: bool,
) {
    if matches!(provider, LlmProvider::Codex) {
        let model = normalize_codex_model(&model);
        // Refresh on every call (mirrors OpenCode behavior) so long-running sessions don't
        // die on token expiry.
        let auth = match crate::codex_oauth::codex_chatgpt_auth().await {
            Ok(a) => a,
            Err(e) => {
                let _ = tx.send(AppEvent::AgentError(e));
                let _ = tx.send(AppEvent::AgentDone);
                return;
            }
        };
        let account_id = codex_account_id.or(auth.account_id);
        call_llm_codex_responses(tx, auth.access_token, account_id, model, messages, tools_enabled)
            .await;
        return;
    }

    let client = reqwest::Client::new();

    let request = ChatRequest {
        model,
        messages,
        stream: true,
        tools: if tools_enabled {
            Some(get_tools())
        } else {
            None
        },
    };

    let url = match provider {
        LlmProvider::OpenRouter => OPENROUTER_URL,
        LlmProvider::OpenAI => OPENAI_URL,
        LlmProvider::Codex => OPENAI_URL, // unreachable (handled above)
    };

    let mut req = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request);

    // OpenRouter recommends these headers; OpenAI ignores unknown headers.
    if provider == LlmProvider::OpenRouter {
        req = req
            .header("HTTP-Referer", "https://github.com/jayasuryajsk/lorikeet")
            .header("X-Title", "Lorikeet");
    }

    let response = req.send().await;

    let response = match response {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.send(AppEvent::AgentError(e.to_string()));
            return;
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let _ = tx.send(AppEvent::AgentError(format!("HTTP {}: {}", status, body)));
        return;
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut pending_tool_calls: Vec<PendingToolCall> = Vec::new();
    let mut finish_reason: Option<String> = None;

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.send(AppEvent::AgentError(e.to_string()));
                return;
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim().to_string();
            buffer = buffer[pos + 1..].to_string();

            if line.is_empty() || line == "data: [DONE]" {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                    for choice in chunk.choices {
                        // Track finish reason
                        if let Some(reason) = choice.finish_reason {
                            finish_reason = Some(reason);
                        }

                        // Handle reasoning tokens (DeepSeek, o1, etc.)
                        if let Some(reasoning) =
                            choice.delta.reasoning_content.or(choice.delta.reasoning)
                        {
                            let _ = tx.send(AppEvent::AgentReasoning(reasoning));
                        }

                        // Handle content tokens
                        if let Some(content) = choice.delta.content {
                            let _ = tx.send(AppEvent::AgentChunk(content));
                        }

                        // Handle tool calls (assembled from deltas)
                        if let Some(tool_calls) = choice.delta.tool_calls {
                            for tc in tool_calls {
                                // Ensure we have enough slots
                                while pending_tool_calls.len() <= tc.index {
                                    pending_tool_calls.push(PendingToolCall::default());
                                }

                                let pending = &mut pending_tool_calls[tc.index];

                                if let Some(id) = tc.id {
                                    pending.id = id;
                                }

                                if let Some(func) = tc.function {
                                    if let Some(name) = func.name {
                                        pending.name = name;
                                    }
                                    if let Some(args) = func.arguments {
                                        pending.arguments.push_str(&args);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Check if we have tool calls to execute
    if finish_reason.as_deref() == Some("tool_calls") && !pending_tool_calls.is_empty() {
        if !tools_enabled {
            let _ = tx.send(AppEvent::AgentError(
                "Plan mode: tool calls requested but tools are disabled".to_string(),
            ));
            let _ = tx.send(AppEvent::AgentDone);
            return;
        }
        let tool_calls: Vec<ToolCallMessage> = pending_tool_calls
            .into_iter()
            .filter(|tc| !tc.id.is_empty())
            .map(|tc| ToolCallMessage {
                id: tc.id,
                call_type: "function".into(),
                function: ToolCallFunction {
                    name: tc.name,
                    arguments: tc.arguments,
                },
            })
            .collect();

        if !tool_calls.is_empty() {
            let _ = tx.send(AppEvent::AgentToolCalls(tool_calls));
            return;
        }
    }

    let _ = tx.send(AppEvent::AgentDone);
}

fn tools_for_responses(tools_enabled: bool) -> Vec<serde_json::Value> {
    if !tools_enabled {
        return Vec::new();
    }
    get_tools()
        .into_iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "name": t.function.name,
                "description": t.function.description,
                "parameters": t.function.parameters,
            })
        })
        .collect()
}

fn build_codex_responses_request(
    model: &str,
    messages: &[ChatMessage],
    tools_enabled: bool,
) -> serde_json::Value {
    // Codex Responses API expects:
    // - instructions: string (system prompt)
    // - input: array of items (messages + function_call + function_call_output)
    let mut instructions_parts: Vec<String> = Vec::new();
    let mut input: Vec<serde_json::Value> = Vec::new();

    for m in messages {
        let role = m.role.as_str();
        let content = m.content.as_deref().unwrap_or("").to_string();

        if role == "system" {
            if !content.trim().is_empty() {
                instructions_parts.push(content);
            }
            continue;
        }

        if role == "tool" {
            if let Some(call_id) = m.tool_call_id.as_deref().filter(|s| !s.trim().is_empty()) {
                input.push(serde_json::json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content,
                }));
            }
            continue;
        }

        if !content.trim().is_empty() {
            let item_type = if role == "assistant" {
                "output_text"
            } else {
                "input_text"
            };
            input.push(serde_json::json!({
                "type": "message",
                "role": role,
                "content": [
                    {"type": item_type, "text": content}
                ]
            }));
        }

        // Assistant tool calls are represented as separate output items in the Responses API.
        if role == "assistant" {
            if let Some(calls) = &m.tool_calls {
                for tc in calls {
                    input.push(serde_json::json!({
                        "type": "function_call",
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                        "call_id": tc.id,
                    }));
                }
            }
        }
    }

    let instructions = instructions_parts.join("\n\n");
    let tools = tools_for_responses(tools_enabled);

    serde_json::json!({
        "model": model,
        "instructions": instructions,
        "input": input,
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "store": false,
        "stream": true,
        "include": [],
    })
}

async fn call_llm_codex_responses(
    tx: mpsc::UnboundedSender<AppEvent>,
    access_token: String,
    account_id: Option<String>,
    model: String,
    messages: Vec<ChatMessage>,
    tools_enabled: bool,
) {
    let client = reqwest::Client::new();
    let mut model = model;
    let url = format!("{}/responses", CODEX_BASE_URL.trim_end_matches('/'));

    // One retry to recover from common "wrong model id" mistakes when using Codex OAuth.
    for attempt in 0..2 {
        let body = build_codex_responses_request(&model, &messages, tools_enabled);
        let mut req = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", access_token))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body);
        if let Some(id) = account_id.as_ref().filter(|s| !s.trim().is_empty()) {
            // Matches OpenCode's behavior; required for some ChatGPT subscription org setups.
            req = req.header("ChatGPT-Account-Id", id);
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(AppEvent::AgentError(e.to_string()));
                let _ = tx.send(AppEvent::AgentDone);
                return;
            }
        };

        if resp.status().is_success() {
            // Continue below to stream SSE.
            let mut stream = resp.bytes_stream();
            let mut buffer = String::new();
            let mut tool_calls: Vec<ToolCallMessage> = Vec::new();

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(AppEvent::AgentError(e.to_string()));
                        let _ = tx.send(AppEvent::AgentDone);
                        return;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // SSE events are separated by a blank line.
                while let Some(pos) = buffer.find("\n\n") {
                    let event_block = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    let mut data_lines = Vec::new();
                    for line in event_block.lines() {
                        let line = line.trim_end();
                        if let Some(d) = line.strip_prefix("data:") {
                            data_lines.push(d.trim_start().to_string());
                        }
                    }
                    if data_lines.is_empty() {
                        continue;
                    }
                    let data = data_lines.join("\n").trim().to_string();
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }

                    let v: serde_json::Value = match serde_json::from_str(&data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let kind = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
                    match kind {
                        "response.output_text.delta" => {
                            if let Some(delta) = v.get("delta").and_then(|x| x.as_str()) {
                                let _ = tx.send(AppEvent::AgentChunk(delta.to_string()));
                            }
                        }
                        "response.reasoning_text.delta" => {
                            if let Some(delta) = v.get("delta").and_then(|x| x.as_str()) {
                                let _ = tx.send(AppEvent::AgentReasoning(delta.to_string()));
                            }
                        }
                        "response.output_item.done" => {
                            if let Some(item) = v.get("item").and_then(|x| x.as_object()) {
                                let item_type =
                                    item.get("type").and_then(|x| x.as_str()).unwrap_or("");
                                if item_type == "function_call" {
                                    let call_id = item
                                        .get("call_id")
                                        .and_then(|x| x.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let name = item
                                        .get("name")
                                        .and_then(|x| x.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let arguments = item
                                        .get("arguments")
                                        .and_then(|x| x.as_str())
                                        .unwrap_or("{}")
                                        .to_string();
                                    if !call_id.is_empty() && !name.is_empty() {
                                        tool_calls.push(ToolCallMessage {
                                            id: call_id,
                                            call_type: "function".into(),
                                            function: ToolCallFunction { name, arguments },
                                        });
                                    }
                                }
                            }
                        }
                        "response.failed" => {
                            let msg = v
                                .get("response")
                                .and_then(|r| r.get("error"))
                                .and_then(|e| e.get("message"))
                                .and_then(|m| m.as_str())
                                .unwrap_or("response.failed");
                            let _ = tx.send(AppEvent::AgentError(msg.to_string()));
                        }
                        _ => {}
                    }
                }
            }

            if !tool_calls.is_empty() {
                if !tools_enabled {
                    let _ = tx.send(AppEvent::AgentError(
                        "Plan mode: tool calls requested but tools are disabled".to_string(),
                    ));
                    let _ = tx.send(AppEvent::AgentDone);
                    return;
                }
                let _ = tx.send(AppEvent::AgentToolCalls(tool_calls));
                return;
            }

            let _ = tx.send(AppEvent::AgentDone);
            return;
        }

        let status = resp.status();
        let body_txt = resp.text().await.unwrap_or_default();

        // Retry with a Codex model slug if the backend rejects the current model.
        if attempt == 0
            && status.as_u16() == 400
            && (body_txt.contains("model is not supported")
                || body_txt.contains("not supported when using Codex"))
        {
            if let Some(fallback) = fallback_codex_model() {
                if fallback != model {
                    model = fallback;
                    continue;
                }
            }
        }

        let _ = tx.send(AppEvent::AgentError(format!("HTTP {}: {}", status, body_txt)));
        let _ = tx.send(AppEvent::AgentDone);
        return;
    }

    // Unreachable: loop returns on success or error.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_request_tools_disabled_in_plan_mode() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            stream: true,
            tools: None,
        };
        let v = serde_json::to_value(&req).unwrap();
        assert!(
            v.get("tools").is_none(),
            "expected tools to be omitted when None"
        );
    }
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

/// Non-streaming helper for one-shot calls (e.g., memory extraction).
pub async fn call_llm_nonstream(
    provider: LlmProvider,
    api_key: String,
    model: String,
    messages: Vec<ChatMessage>,
) -> Result<String, String> {
    let client = reqwest::Client::new();

    let request = ChatRequest {
        model,
        messages,
        stream: false,
        tools: None,
    };

    let url = match provider {
        LlmProvider::OpenRouter => OPENROUTER_URL,
        LlmProvider::OpenAI => OPENAI_URL,
        LlmProvider::Codex => {
            return Err("Codex provider: non-streaming calls are not supported yet".to_string())
        }
    };

    let mut req = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request);

    if provider == LlmProvider::OpenRouter {
        req = req
            .header("HTTP-Referer", "https://github.com/jayasuryajsk/lorikeet")
            .header("X-Title", "Lorikeet");
    }

    let response = req.send().await.map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("HTTP {}: {}", status, body));
    }

    let parsed: ChatResponse = response
        .json()
        .await
        .map_err(|e| format!("Error parsing response: {}", e))?;

    let content = parsed
        .choices
        .get(0)
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();

    Ok(content)
}
