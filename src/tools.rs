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
use crate::semantic_search::{format_search_results, SemanticSearch};

pub const TOOL_NAMES: &[&str] = &[
    "bash",
    "rg",
    "read_file",
    "write_file",
    "list_files",
    "edit_file",
    "semantic_search",
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

            let display = if query.len() > 50 {
                format!("{}... in {}", &query[..50], path)
            } else {
                format!("{} in {}", query, path)
            };

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

            let display = if old_str.len() > 40 {
                format!("{} ({}→{})", path, &old_str[..40], new_str.len())
            } else {
                format!("{}", path)
            };

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
            let display = if query.len() > 50 {
                format!("{}...", &query[..50])
            } else {
                query.to_string()
            };

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

    // Initialize if not already done
    if search_guard.is_none() {
        match SemanticSearch::with_defaults() {
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
        match SemanticSearch::with_defaults() {
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
