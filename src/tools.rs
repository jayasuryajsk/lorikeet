use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;

use parking_lot::Mutex;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;

use crate::events::AppEvent;
use crate::sandbox::SandboxPolicy;
use crate::semantic_search::{format_search_results, SearchConfig, SemanticSearch};

pub const TOOL_NAMES: &[&str] = &[
    "bash",
    "rg",
    "read_file",
    "write_file",
    "list_files",
    "edit_file",
    "semantic_search",
    "smart_search",
    "memory_recall",
    "memory_save",
    "memory_list",
    "memory_forget",
];

// Global semantic search engine (initialized lazily on first use)
static SEMANTIC_SEARCH: OnceLock<Mutex<Option<SemanticSearch>>> = OnceLock::new();

fn get_semantic_search() -> &'static Mutex<Option<SemanticSearch>> {
    SEMANTIC_SEARCH.get_or_init(|| Mutex::new(None))
}

pub async fn execute_tool(
    name: &str,
    args: &str,
    call_id: &str,
    tx: &mpsc::UnboundedSender<AppEvent>,
    policy: &SandboxPolicy,
) -> String {
    if let Err(err) = policy.check_tool_allowed(name) {
        let msg = err.to_string();
        let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
            call_id: call_id.to_string(),
            chunk: msg.clone(),
        }));
        let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
            call_id: call_id.to_string(),
            success: false,
        }));
        return msg;
    }

    let args: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(e) => return format!("Error parsing arguments: {}", e),
    };

    match name {
        "memory_recall" | "memory_save" | "memory_list" | "memory_forget" => {
            // Memory tools are handled in App context (need access to MemoryManager).
            let msg = "Error: memory tools must be handled by the app".to_string();
            let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                call_id: call_id.to_string(),
                chunk: msg.clone(),
            }));
            let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                call_id: call_id.to_string(),
                success: false,
            }));
            msg
        }
        "bash" => {
            let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");

            if let Err(err) = policy.check_command_allowed(command) {
                let msg = err.to_string();
                let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                    call_id: call_id.to_string(),
                    chunk: msg.clone(),
                }));
                let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                    call_id: call_id.to_string(),
                    success: false,
                }));
                return msg;
            }

            if let Err(err) = policy.check_bash_paths(command) {
                let msg = err.to_string();
                let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                    call_id: call_id.to_string(),
                    chunk: msg.clone(),
                }));
                let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                    call_id: call_id.to_string(),
                    success: false,
                }));
                return msg;
            }

            let (result, success) = execute_bash_streaming(command, call_id, tx.clone()).await;
            let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                call_id: call_id.to_string(),
                success,
            }));
            result
        }
        "rg" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            let context = args
                .get("context")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);

            let checked_path = match policy.check_path_allowed(Path::new(path)) {
                Ok(p) => p,
                Err(err) => {
                    let msg = err.to_string();
                    let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                        call_id: call_id.to_string(),
                        chunk: msg.clone(),
                    }));
                    let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                        call_id: call_id.to_string(),
                        success: false,
                    }));
                    return msg;
                }
            };

            let result = execute_rg(query, checked_path.to_string_lossy().as_ref(), context).await;
            let success = !result.starts_with("Error:");
            let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                call_id: call_id.to_string(),
                chunk: result.clone(),
            }));
            let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                call_id: call_id.to_string(),
                success,
            }));
            result
        }
        "read_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");

            let checked_path = match policy.check_path_allowed(Path::new(path)) {
                Ok(p) => p,
                Err(err) => {
                    let msg = err.to_string();
                    let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                        call_id: call_id.to_string(),
                        chunk: msg.clone(),
                    }));
                    let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                        call_id: call_id.to_string(),
                        success: false,
                    }));
                    return msg;
                }
            };

            let result = match tokio::fs::read_to_string(&checked_path).await {
                Ok(content) => content,
                Err(e) => format!("Error reading file: {}", e),
            };
            let success = !result.starts_with("Error");
            let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                call_id: call_id.to_string(),
                chunk: result.clone(),
            }));
            let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                call_id: call_id.to_string(),
                success,
            }));
            result
        }
        "write_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");

            let checked_path = match policy.check_path_allowed(Path::new(path)) {
                Ok(p) => p,
                Err(err) => {
                    let msg = err.to_string();
                    let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                        call_id: call_id.to_string(),
                        chunk: msg.clone(),
                    }));
                    let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                        call_id: call_id.to_string(),
                        success: false,
                    }));
                    return msg;
                }
            };

            let result = match tokio::fs::write(&checked_path, content).await {
                Ok(_) => format!("Successfully wrote {} bytes to {}", content.len(), path),
                Err(e) => format!("Error writing file: {}", e),
            };
            let success = !result.starts_with("Error");
            let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                call_id: call_id.to_string(),
                chunk: result.clone(),
            }));
            let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                call_id: call_id.to_string(),
                success,
            }));
            result
        }
        "list_files" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

            let checked_path = match policy.check_path_allowed(Path::new(path)) {
                Ok(p) => p,
                Err(err) => {
                    let msg = err.to_string();
                    let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                        call_id: call_id.to_string(),
                        chunk: msg.clone(),
                    }));
                    let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                        call_id: call_id.to_string(),
                        success: false,
                    }));
                    return msg;
                }
            };

            let result = match tokio::fs::read_dir(&checked_path).await {
                Ok(mut entries) => {
                    let mut files = Vec::new();
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        let name = entry.file_name().to_string_lossy().to_string();
                        let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
                        files.push(if is_dir { format!("{}/", name) } else { name });
                    }
                    files.sort();
                    files.join("\n")
                }
                Err(e) => format!("Error listing directory: {}", e),
            };
            let success = !result.starts_with("Error");
            let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                call_id: call_id.to_string(),
                chunk: result.clone(),
            }));
            let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                call_id: call_id.to_string(),
                success,
            }));
            result
        }
        "edit_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let old_str = args
                .get("old_string")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let new_str = args
                .get("new_string")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let checked_path = match policy.check_path_allowed(Path::new(path)) {
                Ok(p) => p,
                Err(err) => {
                    let msg = err.to_string();
                    let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                        call_id: call_id.to_string(),
                        chunk: msg.clone(),
                    }));
                    let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                        call_id: call_id.to_string(),
                        success: false,
                    }));
                    return msg;
                }
            };

            let result = edit_file(&checked_path, old_str, new_str).await;
            let success = !result.starts_with("Error");
            let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                call_id: call_id.to_string(),
                chunk: result.clone(),
            }));
            let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                call_id: call_id.to_string(),
                success,
            }));
            result
        }
        "semantic_search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");

            let result = execute_semantic_search(query, policy).await;
            let success = !result.starts_with("Error");
            let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                call_id: call_id.to_string(),
                chunk: result.clone(),
            }));
            let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                call_id: call_id.to_string(),
                success,
            }));
            result
        }
        "smart_search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            let limit = args
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            let use_rg = args.get("rg").and_then(|v| v.as_bool()).unwrap_or(true);
            let use_semantic = args
                .get("semantic")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            let result =
                execute_smart_search(query, path, limit, use_rg, use_semantic, policy).await;
            let success = !result.starts_with("Error");
            let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                call_id: call_id.to_string(),
                chunk: result.clone(),
            }));
            let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                call_id: call_id.to_string(),
                success,
            }));
            result
        }
        _ => format!("Unknown tool: {}", name),
    }
}

async fn execute_bash_streaming(
    command: &str,
    call_id: &str,
    tx: mpsc::UnboundedSender<AppEvent>,
) -> (String, bool) {
    // Tool results are fed back into the model, so we aggregate output.
    // Keep it bounded and prefer tail output.
    const MAX_TOOL_RESULT_CHARS: usize = 20_000;
    const TIMEOUT_SECS: u64 = 60;

    #[derive(Default)]
    struct OutputAcc {
        lines: VecDeque<String>,
        chars: usize,
        truncated: bool,
    }

    impl OutputAcc {
        fn push_line(&mut self, line: String) {
            self.chars = self.chars.saturating_add(line.len());
            self.lines.push_back(line);
            while self.chars > MAX_TOOL_RESULT_CHARS {
                self.truncated = true;
                if let Some(front) = self.lines.pop_front() {
                    self.chars = self.chars.saturating_sub(front.len());
                } else {
                    break;
                }
            }
        }

        fn into_string(self) -> String {
            let mut out = String::new();
            if self.truncated {
                out.push_str("... (output truncated; showing tail)\n");
            }
            for l in self.lines {
                out.push_str(&l);
            }
            out
        }
    }

    let acc = std::sync::Arc::new(tokio::sync::Mutex::new(OutputAcc::default()));

    let call_id_owned = call_id.to_string();

    let mut child = match Command::new("bash")
        .arg("-c")
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return (format!("Error spawning process: {}", e), false),
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    async fn stream_lossy<R: tokio::io::AsyncRead + Unpin>(
        r: R,
        call_id: String,
        tx: mpsc::UnboundedSender<AppEvent>,
        prefix: &'static str,
        acc: std::sync::Arc<tokio::sync::Mutex<OutputAcc>>,
    ) {
        let mut reader = BufReader::new(r);
        let mut buf: Vec<u8> = Vec::with_capacity(1024);

        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf).await {
                Ok(0) => break,
                Ok(_) => {
                    let s = String::from_utf8_lossy(&buf);
                    let chunk = if prefix.is_empty() {
                        s.to_string()
                    } else {
                        format!("{}{}", prefix, s)
                    };

                    let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                        call_id: call_id.to_string(),
                        chunk: chunk.clone(),
                    }));

                    let mut guard = acc.lock().await;
                    guard.push_line(chunk);
                }
                Err(_) => break,
            }
        }
    }

    let stdout_task = if let Some(out) = stdout {
        let tx2 = tx.clone();
        let acc2 = acc.clone();
        Some(tokio::spawn(stream_lossy(
            out,
            call_id_owned.clone(),
            tx2,
            "",
            acc2,
        )))
    } else {
        None
    };

    let stderr_task = if let Some(err) = stderr {
        let tx2 = tx.clone();
        let acc2 = acc.clone();
        Some(tokio::spawn(stream_lossy(
            err,
            call_id_owned.clone(),
            tx2,
            "[stderr] ",
            acc2,
        )))
    } else {
        None
    };

    let timeout = tokio::time::Duration::from_secs(TIMEOUT_SECS);
    let (success, code) = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => (status.success(), status.code().unwrap_or(-1)),
        Ok(Err(e)) => {
            let msg = format!("Error executing command: {}\n", e);
            let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                call_id: call_id.to_string(),
                chunk: msg.clone(),
            }));
            let mut guard = acc.lock().await;
            guard.push_line(msg);
            (false, -1)
        }
        Err(_) => {
            let msg = format!("Error: Command timed out after {} seconds\n", TIMEOUT_SECS);
            let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                call_id: call_id.to_string(),
                chunk: msg.clone(),
            }));
            let mut guard = acc.lock().await;
            guard.push_line(msg);
            let _ = child.start_kill();
            // Best-effort: allow the child to exit so pipe readers finish.
            let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), child.wait()).await;
            (false, -1)
        }
    };

    if let Some(t) = stdout_task {
        let _ = t.await;
    }
    if let Some(t) = stderr_task {
        let _ = t.await;
    }

    // Add an exit marker to both the UI stream and the model-visible tool result.
    let exit_line = format!("[exit] {}\n", code);
    let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
        call_id: call_id.to_string(),
        chunk: exit_line.clone(),
    }));
    {
        let mut guard = acc.lock().await;
        guard.push_line(exit_line);
    }

    let out = {
        let mut guard = acc.lock().await;
        std::mem::take(&mut *guard).into_string()
    };

    (out, success)
}

async fn execute_rg(query: &str, path: &str, context: Option<usize>) -> String {
    if query.trim().is_empty() {
        return "Error: Query cannot be empty".to_string();
    }

    let mut cmd = Command::new("rg");
    cmd.arg("--line-number")
        .arg("--column")
        .arg("--no-heading")
        .arg("--hidden")
        .arg("--glob")
        .arg("!.git/*");

    if let Some(lines) = context {
        cmd.arg("-C").arg(lines.to_string());
    }

    cmd.arg(query);
    if !path.is_empty() {
        cmd.arg(path);
    }

    let child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return format!("Error spawning rg: {}", e),
    };

    let timeout = tokio::time::Duration::from_secs(30);
    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if !output.status.success() {
                if output.status.code() == Some(1) {
                    return "No matches.".to_string();
                }
                if !stderr.trim().is_empty() {
                    return format!("Error: {}", stderr.trim());
                }
                return "Error: rg failed".to_string();
            }

            let mut result = stdout.to_string();
            if result.trim().is_empty() {
                result = "No matches.".to_string();
            }
            if result.len() > 10000 {
                result.truncate(10000);
                result.push_str("\n... (output truncated)");
            }
            result
        }
        Ok(Err(e)) => format!("Error executing rg: {}", e),
        Err(_) => "Error: rg timed out after 30 seconds".into(),
    }
}

async fn execute_semantic_search(query: &str, policy: &SandboxPolicy) -> String {
    if query.trim().is_empty() {
        return "Error: Query cannot be empty".to_string();
    }

    let cwd = match std::env::current_dir() {
        Ok(dir) => dir,
        Err(e) => return format!("Error: {}", e),
    };

    let checked_root = match policy.check_path_allowed(&cwd) {
        Ok(p) => p,
        Err(err) => return err.to_string(),
    };

    // Initialize or get the semantic search engine
    let search_mutex = get_semantic_search();
    let mut search_guard = search_mutex.lock();

    // Initialize if not already done.
    if search_guard.is_none() {
        let cfg = SearchConfig::for_workspace(&checked_root);
        match SemanticSearch::new(cfg) {
            Ok(search) => {
                // Set project root to current directory
                search.set_project_root(checked_root);
                *search_guard = Some(search);
            }
            Err(e) => {
                return format!("Error initializing semantic search: {}", e);
            }
        }
    }

    let search = search_guard.as_ref().unwrap();

    // Perform the search
    match search.search(query) {
        Ok(results) => {
            if results.is_empty() {
                "No results found. The index may be empty - try indexing the project first."
                    .to_string()
            } else {
                format_search_results(&results)
            }
        }
        Err(e) => format!("Error searching: {}", e),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmartSource {
    Rg,
    Semantic,
    Both,
}

#[derive(Debug, Clone)]
struct SmartHit {
    source: SmartSource,
    score: f32, // higher is better
    path: String,
    line: usize,
    col: Option<usize>,
    snippet: String,
}

fn trunc_line(s: &str, max: usize) -> String {
    let t = s.trim();
    if t.len() <= max {
        return t.to_string();
    }
    let mut out = t.to_string();
    out.truncate(max.saturating_sub(3));
    out.push_str("...");
    out
}

fn first_snippet(text: &str) -> String {
    for l in text.lines() {
        let t = l.trim();
        if !t.is_empty() {
            return trunc_line(t, 140);
        }
    }
    String::new()
}

fn parse_rg_line(line: &str) -> Option<(String, usize, usize, String)> {
    // Expected: path:line:col:text
    let mut parts = line.splitn(4, ':');
    let path = parts.next()?.to_string();
    let line_no = parts.next()?.parse::<usize>().ok()?;
    let col = parts.next()?.parse::<usize>().ok()?;
    let text = parts.next().unwrap_or("").to_string();
    Some((path, line_no, col, text))
}

async fn execute_smart_search(
    query: &str,
    path: &str,
    limit: Option<usize>,
    use_rg: bool,
    use_semantic: bool,
    policy: &SandboxPolicy,
) -> String {
    if query.trim().is_empty() {
        return "Error: query cannot be empty".to_string();
    }

    let workspace_root = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => return format!("Error: cwd: {}", e),
    };

    let checked_scope = match policy.check_path_allowed(Path::new(path)) {
        Ok(p) => p,
        Err(e) => return e.to_string(),
    };

    // Scope prefix for filtering semantic hits (index uses workspace-root-relative paths).
    let scope_rel: PathBuf = if checked_scope == workspace_root {
        PathBuf::from(".")
    } else if let Ok(rel) = checked_scope.strip_prefix(&workspace_root) {
        rel.to_path_buf()
    } else {
        // Fallback: treat as '.' to avoid filtering out everything.
        PathBuf::from(".")
    };

    let limit = limit.unwrap_or(20).clamp(1, 50);

    let mut hits_by_key: std::collections::HashMap<(String, usize), SmartHit> =
        std::collections::HashMap::new();

    let mut rg_count = 0usize;
    let mut sem_count = 0usize;

    if use_rg {
        let out = execute_rg(query, checked_scope.to_string_lossy().as_ref(), None).await;
        if !out.starts_with("Error:") && !out.starts_with("No matches") {
            for l in out.lines() {
                if let Some((p, line_no, col, text)) = parse_rg_line(l) {
                    rg_count += 1;
                    let key = (p.clone(), line_no);
                    let snippet = first_snippet(&text);
                    let candidate = SmartHit {
                        source: SmartSource::Rg,
                        score: 0.80, // base; BOTH will be boosted.
                        path: p,
                        line: line_no,
                        col: Some(col),
                        snippet,
                    };
                    hits_by_key
                        .entry(key)
                        .and_modify(|h| {
                            // Keep best version.
                            if h.source == SmartSource::Semantic {
                                h.source = SmartSource::Both;
                                h.score = h.score.max(0.95);
                                if h.snippet.is_empty() {
                                    h.snippet = candidate.snippet.clone();
                                }
                                if h.col.is_none() {
                                    h.col = candidate.col;
                                }
                            } else if candidate.score > h.score {
                                *h = candidate.clone();
                            }
                        })
                        .or_insert(candidate);
                }
            }
        }
    }

    if use_semantic {
        // Initialize or get the semantic search engine (workspace-scoped index dir).
        let checked_root = match policy.check_path_allowed(&workspace_root) {
            Ok(p) => p,
            Err(e) => return e.to_string(),
        };

        let search_mutex = get_semantic_search();
        let mut search_guard = search_mutex.lock();
        if search_guard.is_none() {
            let cfg = SearchConfig::for_workspace(&checked_root);
            match SemanticSearch::new(cfg) {
                Ok(search) => {
                    search.set_project_root(checked_root.clone());
                    *search_guard = Some(search);
                }
                Err(e) => return format!("Error initializing semantic search: {}", e),
            }
        }
        let search = search_guard.as_ref().unwrap();

        match search.search(query) {
            Ok(results) => {
                for r in results {
                    let rel_path = r.chunk.metadata.file_path.clone();
                    if scope_rel != PathBuf::from(".") && !rel_path.starts_with(&scope_rel) {
                        continue;
                    }
                    sem_count += 1;

                    let p = rel_path.to_string_lossy().to_string();
                    let line_no = r.chunk.metadata.start_line.max(1);
                    let key = (p.clone(), line_no);
                    let snippet = first_snippet(&r.chunk.content);
                    let candidate = SmartHit {
                        source: SmartSource::Semantic,
                        score: r.score, // 0..1
                        path: p,
                        line: line_no,
                        col: None,
                        snippet,
                    };

                    hits_by_key
                        .entry(key)
                        .and_modify(|h| {
                            if h.source == SmartSource::Rg {
                                h.source = SmartSource::Both;
                                h.score = h.score.max(0.95);
                                if h.snippet.is_empty() {
                                    h.snippet = candidate.snippet.clone();
                                }
                            } else if candidate.score > h.score {
                                *h = candidate.clone();
                            }
                        })
                        .or_insert(candidate);
                }
            }
            Err(e) => return format!("Error searching: {}", e),
        }
    }

    let deduped_total = hits_by_key.len();
    if deduped_total == 0 {
        return "No results.".to_string();
    }

    let mut hits: Vec<SmartHit> = hits_by_key.into_values().collect();
    hits.sort_by(|a, b| {
        let pri = |s: SmartSource| match s {
            SmartSource::Both => 0,
            SmartSource::Rg => 1,
            SmartSource::Semantic => 2,
        };
        pri(a.source)
            .cmp(&pri(b.source))
            .then_with(|| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.line.cmp(&b.line))
    });
    hits.truncate(limit);
    let shown = hits.len();

    let mut out = String::new();
    for h in hits {
        let tag = match h.source {
            SmartSource::Both => "BOTH",
            SmartSource::Rg => "RG",
            SmartSource::Semantic => "SEM",
        };
        let loc = if let Some(c) = h.col {
            format!("{}:{}:{}", h.path, h.line, c)
        } else {
            format!("{}:{}", h.path, h.line)
        };
        let snippet = if h.snippet.is_empty() {
            "".to_string()
        } else {
            format!("  {}", h.snippet)
        };
        out.push_str(&format!(
            "{:<4} score={:.2}  {}{}\n",
            tag, h.score, loc, snippet
        ));
    }

    out.push_str(&format!(
        "\n(counts: rg={} sem={} deduped={} shown={})",
        rg_count, sem_count, deduped_total, shown
    ));
    out
}
/// Set the project root for semantic search (called when starting the app)
#[allow(dead_code)]
pub fn set_semantic_search_project_root(path: PathBuf) {
    let search_mutex = get_semantic_search();
    let search_guard = search_mutex.lock();

    if let Some(ref search) = *search_guard {
        search.set_project_root(path);
    }
}

/// Index a directory for semantic search
#[allow(dead_code)]
pub async fn index_directory_for_search(
    path: &std::path::Path,
    policy: &SandboxPolicy,
) -> Result<String, String> {
    let checked_path = policy.check_path_allowed(path).map_err(|e| e.to_string())?;

    let search_mutex = get_semantic_search();
    let mut search_guard = search_mutex.lock();

    // Initialize if not already done
    if search_guard.is_none() {
        let cfg = SearchConfig::for_workspace(&checked_path);
        match SemanticSearch::new(cfg) {
            Ok(search) => {
                *search_guard = Some(search);
            }
            Err(e) => {
                return Err(format!("Error initializing semantic search: {}", e));
            }
        }
    }

    let search = search_guard.as_ref().unwrap();

    match search.index_directory(&checked_path) {
        Ok(stats) => Ok(format!(
            "Indexed {} chunks from {} files\nIndex size: {} bytes\nLanguages: {:?}",
            stats.total_chunks, stats.total_files, stats.index_size_bytes, stats.languages
        )),
        Err(e) => Err(format!("Error indexing: {}", e)),
    }
}

async fn edit_file(path: &Path, old_string: &str, new_string: &str) -> String {
    // Read the file
    let content = match tokio::fs::read_to_string(path).await {
        Ok(c) => c,
        Err(e) => return format!("Error reading file: {}", e),
    };

    // Check if old_string exists
    if !content.contains(old_string) {
        return format!(
            "Error: Could not find the specified text in {}",
            path.display()
        );
    }

    // Check for uniqueness - count occurrences
    let count = content.matches(old_string).count();
    if count > 1 {
        return format!(
            "Error: Found {} occurrences of the text. Please provide a more unique string.",
            count
        );
    }

    // Replace
    let new_content = content.replacen(old_string, new_string, 1);

    // Write back
    match tokio::fs::write(path, &new_content).await {
        Ok(_) => {
            let old_lines = old_string.lines().count();
            let new_lines = new_string.lines().count();
            format!(
                "Edited {} ({} lines → {} lines)",
                path.display(),
                old_lines,
                new_lines
            )
        }
        Err(e) => format!("Error writing file: {}", e),
    }
}
