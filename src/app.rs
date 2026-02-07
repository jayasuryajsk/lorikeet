use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::prelude::Rect;
use tokio::sync::mpsc;

use crate::config::AppConfig;
use crate::events::AppEvent;
use crate::llm::{call_llm, ChatMessage};
use crate::memory::MemoryManager;
use crate::sandbox::SandboxPolicy;
use crate::semantic_search::{index_dir_for_workspace, SearchConfig, SemanticSearch};
use crate::session::{replay_into, SessionStore};
use crate::tools::execute_tool;
use crate::types::ToolCallMessage;
use crate::verify::detect_suggestions;

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Option<Vec<ToolCallMessage>>,
    pub tool_group_id: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Agent,
    System,
    Tool,
}

impl Message {
    fn to_chat_message(&self) -> ChatMessage {
        match self.role {
            Role::Tool => ChatMessage {
                role: "tool".into(),
                content: Some(self.content.clone()),
                tool_calls: None,
                tool_call_id: self.reasoning.clone(), // We store tool_call_id in reasoning field for Tool messages
                name: None,
            },
            _ => ChatMessage {
                role: match self.role {
                    Role::User => "user".into(),
                    Role::Agent => "assistant".into(),
                    Role::System => "system".into(),
                    Role::Tool => "tool".into(),
                },
                content: if self.content.is_empty() {
                    None
                } else {
                    Some(self.content.clone())
                },
                tool_calls: self.tool_calls.clone(),
                tool_call_id: None,
                name: None,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub call_id: String,
    pub tool: String,
    pub args_raw: String,
    pub args_summary: String,
    pub args_pretty_lines: Vec<String>,
    pub cwd: PathBuf,
    pub sandbox: crate::sandbox::SandboxDecision,

    // Retained for compatibility with existing summary/memory code paths.
    pub target: String,

    // Full aggregated output (model-visible). The UI should prefer output_lines/tail_lines.
    pub output: String,

    // Append-only line buffer for stable UI tails.
    pub output_lines: VecDeque<String>,
    output_partial: String,
    output_total_lines: usize,
    output_truncated: bool,

    pub status: ToolStatus,
    pub turn_id: u64,
    pub group_id: u64,

    start_time: Instant,
    end_time: Option<Instant>,
}

impl ToolOutput {
    pub fn new(
        call_id: String,
        tool: String,
        args_raw: String,
        args_summary: String,
        cwd: PathBuf,
        sandbox: crate::sandbox::SandboxDecision,
        turn_id: u64,
        group_id: u64,
    ) -> Self {
        let target = args_summary.clone();
        Self {
            call_id,
            tool,
            args_raw: args_raw.clone(),
            args_summary: args_summary.clone(),
            args_pretty_lines: pretty_args_lines(&args_raw),
            cwd,
            sandbox,
            target,
            output: String::new(),
            output_lines: VecDeque::new(),
            output_partial: String::new(),
            output_total_lines: 0,
            output_truncated: false,
            status: ToolStatus::Running,
            turn_id,
            group_id,
            start_time: Instant::now(),
            end_time: None,
        }
    }

    pub fn elapsed(&self) -> Duration {
        match self.end_time {
            Some(end) => end.duration_since(self.start_time),
            None => self.start_time.elapsed(),
        }
    }

    pub fn complete(&mut self, success: bool) {
        self.status = if success {
            ToolStatus::Success
        } else {
            ToolStatus::Error
        };
        self.end_time = Some(Instant::now());

        // Flush any pending partial line so it can be shown in tails.
        if !self.output_partial.is_empty() {
            let line = std::mem::take(&mut self.output_partial);
            self.push_line(line);
        }
    }

    pub fn append_chunk(&mut self, chunk: String) {
        self.output.push_str(&chunk);

        for seg in chunk.split_inclusive('\n') {
            if let Some(stripped) = seg.strip_suffix('\n') {
                self.output_partial.push_str(stripped);
                let line = std::mem::take(&mut self.output_partial);
                self.push_line(line);
            } else {
                self.output_partial.push_str(seg);
            }
        }
    }

    pub fn set_output(&mut self, content: String) {
        self.output.clear();
        self.output_lines.clear();
        self.output_partial.clear();
        self.output_total_lines = 0;
        self.output_truncated = false;

        self.append_chunk(content);
    }

    fn push_line(&mut self, mut line: String) {
        const MAX_STORED_LINES: usize = 5_000;
        const MAX_STORED_CHARS: usize = 80_000;

        // Avoid CRLF artifacts in terminals.
        if line.ends_with('\r') {
            line.pop();
        }

        self.output_total_lines = self.output_total_lines.saturating_add(1);
        self.output_lines.push_back(line);

        while self.output_lines.len() > MAX_STORED_LINES {
            self.output_truncated = true;
            self.output_lines.pop_front();
        }

        // Best-effort char bound: drop from the front if needed.
        let mut chars: usize = self.output_lines.iter().map(|l| l.len()).sum();
        while chars > MAX_STORED_CHARS {
            self.output_truncated = true;
            if let Some(front) = self.output_lines.pop_front() {
                chars = chars.saturating_sub(front.len());
            } else {
                break;
            }
        }
    }

    pub fn total_output_lines(&self) -> usize {
        let partial = if self.output_partial.is_empty() { 0 } else { 1 };
        self.output_total_lines.saturating_add(partial)
    }

    pub fn tail_lines(&self, max_lines: usize) -> (Vec<String>, usize) {
        if max_lines == 0 {
            return (Vec::new(), self.total_output_lines());
        }

        let mut all: Vec<String> = self.output_lines.iter().cloned().collect();
        if !self.output_partial.is_empty() {
            all.push(self.output_partial.clone());
        }

        let total = self.total_output_lines();
        let start = all.len().saturating_sub(max_lines);
        let tail = all[start..].to_vec();
        let remaining = total.saturating_sub(tail.len());
        (tail, remaining)
    }

    pub fn output_is_truncated(&self) -> bool {
        self.output_truncated
    }

    /// Get the icon for this tool type
    pub fn icon(&self) -> &'static str {
        match self.tool.as_str() {
            "bash" => "$",
            "rg" => "⌕",
            "smart_search" => "≈",
            "read_file" => "▶",
            "write_file" => "◀",
            "list_files" => "◇",
            "edit_file" => "±",
            "apply_patch" => "▦",
            "open_at" => "↗",
            "semantic_search" => "?",
            "verify" => "✓",
            _ => "○",
        }
    }

    /// Get the action verb (present tense while running, past tense when done)
    pub fn action_verb(&self) -> &'static str {
        match (&self.tool.as_str(), &self.status) {
            (&"bash", ToolStatus::Running) => "Running",
            (&"bash", _) => "Ran",
            (&"rg", ToolStatus::Running) => "Searching",
            (&"rg", _) => "Searched",
            (&"smart_search", ToolStatus::Running) => "Searching",
            (&"smart_search", _) => "Searched",
            (&"read_file", ToolStatus::Running) => "Reading",
            (&"read_file", _) => "Read",
            (&"write_file", ToolStatus::Running) => "Writing",
            (&"write_file", _) => "Wrote",
            (&"list_files", ToolStatus::Running) => "Listing",
            (&"list_files", _) => "Listed",
            (&"edit_file", ToolStatus::Running) => "Editing",
            (&"edit_file", _) => "Edited",
            (&"apply_patch", ToolStatus::Running) => "Applying",
            (&"apply_patch", _) => "Applied",
            (&"open_at", ToolStatus::Running) => "Opening",
            (&"open_at", _) => "Opened",
            (&"semantic_search", ToolStatus::Running) => "Searching",
            (&"semantic_search", _) => "Searched",
            (&"verify", ToolStatus::Running) => "Verifying",
            (&"verify", _) => "Verified",
            (_, ToolStatus::Running) => "Processing",
            (_, _) => "Done",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToolStatus {
    Running,
    Success,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pane {
    Chat,
    Context,
}

const SYSTEM_PROMPT: &str = r#"You are Lorikeet, an autonomous coding agent.

Tools:
- bash: Run any shell command. Use for reading files (cat), listing dirs (ls), git, builds, tests, etc.
- rg: Fast exact text search across files. Use for symbols, strings, or precise matches.
- smart_search: Combined search (rg + semantic). Use when you don't know exact identifiers; returns ranked hits.
- read_file: Read file contents directly.
- write_file: Write content to a file directly.
- list_files: List directory contents directly.
- edit_file: Make surgical edits to files. Args: path, old_string, new_string. The old_string must be unique in the file.
- apply_patch: Apply a patch (*** Begin Patch / Update File / Add File / Delete File). Use for refactors and non-trivial edits.
- open_at: Read a file around a specific line with context + line numbers. Use after search results (path:line).
- semantic_search: Search code semantically using natural language. Returns ranked results with file:line. Use for finding code related to concepts, features, or functionality. Auto-indexes on first use.
- verify: Run a verify command (tests/build). If omitted, auto-detects a suggestion. Respects sandbox.
- memory_recall: Retrieve relevant long-term memory. Use before repeating actions or making risky changes.
- memory_save: Save long-term memory about mistakes, preferences, and decisions. Never store secrets.
- memory_list: List memories.
- memory_forget: Delete a memory by id.

Workflow: (1) Write a short plan. (2) Execute using tools. (3) Verify changes by running relevant tests/build commands when possible. If sandbox blocks a verification command, explain what to allowlist.

Be concise. Verify your work. If something fails, try a different approach."#;

/// Status of the background indexing process
#[derive(Debug, Clone)]
pub enum IndexingStatus {
    NotStarted,
    Indexing {
        files_done: usize,
        total_files: usize,
    },
    Complete {
        chunks: usize,
        files: usize,
    },
    Error(String),
}

pub struct App {
    pub input: String,
    pub cursor_pos: usize,
    messages: Vec<Message>,
    pub messages_scroll: usize,
    pub messages_auto_scroll: bool,
    pub tool_outputs: Vec<ToolOutput>,
    pub active_pane: Pane,
    pub should_quit: bool,
    pub is_processing: bool,
    pub processing_start: Option<Instant>,
    pub current_response: String,
    pub current_reasoning: String,
    pub spinner_frame: usize,
    pub tool_spinner_frame: usize,
    event_tx: mpsc::UnboundedSender<AppEvent>,
    api_key: String,
    pub model: String,
    sandbox_policy: Arc<SandboxPolicy>,
    pub memory: Arc<MemoryManager>,
    pub config: AppConfig,
    workspace_root: PathBuf,
    // Layout areas for mouse handling
    pub chat_area: Rect,
    pub context_area: Rect,
    pub root_area: Rect,
    pub splitter_area: Rect,
    pub split_ratio: u16, // left pane percentage
    dragging_splitter: bool,
    // Settings UI
    pub settings_open: bool,
    pub settings_selected: usize,
    pub settings_input: String,
    pub settings_cursor: usize,
    settings_draft: AppConfig,
    // Turn tracking (for memory extraction)
    turn_user_message: Option<String>,
    turn_tool_start_idx: usize,

    // Turn tracking (for memory extraction and grouping user messages)
    pub current_turn_id: u64,

    // Tool trace grouping (one group per assistant tool-call phase)
    next_tool_group_id: u64,
    pub last_tool_group_id: Option<u64>,
    tool_group_by_call_id: HashMap<String, u64>,

    // Inline tool trace UI state (keyed by tool_group_id)
    pub tool_trace_expanded: HashMap<u64, bool>,
    pub tool_trace_show_details: HashMap<u64, bool>,

    // Tool run index
    tool_index_by_call_id: HashMap<String, usize>,

    // Tool loop guard (prevents infinite retries on repeated failures)
    tool_failure_counts: HashMap<(u64, String), usize>,
    tool_loop_abort: Option<(u64, String)>, // (turn_id, message)

    // Context sidebar
    pub recent_files: VecDeque<String>,
    pub last_searches: VecDeque<String>,

    // Indexing status
    pub indexing_status: IndexingStatus,
    pub indexing_spinner_frame: usize,

    // Verify suggestions
    pub verify_suggestions: Vec<crate::verify::VerifySuggestion>,

    // Session persistence
    pub session: Option<SessionStore>,
}

impl App {
    pub fn new(
        event_tx: mpsc::UnboundedSender<AppEvent>,
        api_key: String,
        sandbox_policy: Arc<SandboxPolicy>,
        config: AppConfig,
        workspace_root: PathBuf,
        memory: Arc<MemoryManager>,
    ) -> Self {
        let split_ratio = config
            .general
            .as_ref()
            .and_then(|g| g.split_ratio)
            .unwrap_or(60);
        let model = config
            .general
            .as_ref()
            .and_then(|g| g.model.clone())
            .unwrap_or_else(|| crate::llm::MODEL.to_string());
        let settings_draft = config.clone();
        Self {
            input: String::new(),
            cursor_pos: 0,
            messages: vec![
                Message {
                    role: Role::System,
                    content: SYSTEM_PROMPT.into(),
                    reasoning: None,
                    tool_calls: None,
                    tool_group_id: None,
                },
                Message {
                    role: Role::Agent,
                    content: "Lorikeet ready. I can execute commands, read/write files, and help you build things. What would you like to do?".into(),
                    reasoning: None,
                    tool_calls: None,
                    tool_group_id: None,
                },
            ],
            messages_scroll: 0,
            messages_auto_scroll: true,
            tool_outputs: vec![],
            active_pane: Pane::Chat,
            should_quit: false,
            is_processing: false,
            processing_start: None,
            current_response: String::new(),
            current_reasoning: String::new(),
            spinner_frame: 0,
            tool_spinner_frame: 0,
            event_tx,
            api_key,
            model,
            sandbox_policy,
            memory,
            config,
            workspace_root,
            chat_area: Rect::default(),
            context_area: Rect::default(),
            root_area: Rect::default(),
            splitter_area: Rect::default(),
            split_ratio,
            dragging_splitter: false,
            settings_open: false,
            settings_selected: 0,
            settings_input: String::new(),
            settings_cursor: 0,
            settings_draft,
            turn_user_message: None,
            turn_tool_start_idx: 0,
            current_turn_id: 0,
            next_tool_group_id: 1,
            last_tool_group_id: None,
            tool_group_by_call_id: HashMap::new(),
            tool_trace_expanded: HashMap::new(),
            tool_trace_show_details: HashMap::new(),
            tool_index_by_call_id: HashMap::new(),
            tool_failure_counts: HashMap::new(),
            tool_loop_abort: None,
            recent_files: VecDeque::new(),
            last_searches: VecDeque::new(),
            indexing_status: load_existing_index_status(),
            indexing_spinner_frame: 0,
            verify_suggestions: Vec::new(),
            session: None,
        }
    }

    pub fn init_session(&mut self, resume: bool) {
        // Decide whether to resume
        if resume {
            if let Ok(Some(store)) = SessionStore::open_latest(&self.workspace_root) {
                if let Ok(events) = store.load_events() {
                    // Rebuild app state from events.
                    self.messages.clear();
                    self.tool_outputs.clear();
                    self.tool_trace_expanded.clear();
                    self.tool_trace_show_details.clear();
                    self.tool_index_by_call_id.clear();
                    self.tool_failure_counts.clear();
                    self.tool_loop_abort = None;
                    self.tool_group_by_call_id.clear();
                    self.last_tool_group_id = None;
                    self.next_tool_group_id = 1;
                    self.recent_files.clear();
                    replay_into(&events, &mut self.messages, &mut self.tool_outputs);

                    // Rebuild call_id -> group_id mapping from persisted tool events (best-effort).
                    for t in &self.tool_outputs {
                        self.tool_group_by_call_id
                            .insert(t.call_id.clone(), t.group_id);
                    }

                    // Restore group id counters so new tool calls stay in-order after resume.
                    let max_group_msg = self
                        .messages
                        .iter()
                        .filter_map(|m| m.tool_group_id)
                        .max()
                        .unwrap_or(0);
                    let max_group_tool = self
                        .tool_outputs
                        .iter()
                        .map(|t| t.group_id)
                        .max()
                        .unwrap_or(0);
                    let max_group_id = max_group_msg.max(max_group_tool);
                    self.next_tool_group_id = max_group_id.saturating_add(1).max(1);
                    self.last_tool_group_id = if max_group_id > 0 {
                        Some(max_group_id)
                    } else {
                        None
                    };

                    for (i, t) in self.tool_outputs.iter().enumerate() {
                        self.tool_index_by_call_id.insert(t.call_id.clone(), i);
                    }

                    // Sync turn counter with the restored transcript so new tool calls
                    // attach to the correct user turn.
                    let user_turns = self
                        .messages
                        .iter()
                        .filter(|m| m.role == Role::User)
                        .count() as u64;
                    let max_tool_turn = self
                        .tool_outputs
                        .iter()
                        .map(|t| t.turn_id)
                        .max()
                        .unwrap_or(0);
                    self.current_turn_id = user_turns.max(max_tool_turn);

                    // Rebuild a small context list from restored tools.
                    let recent_paths: Vec<String> = self
                        .tool_outputs
                        .iter()
                        .filter(|t| {
                            t.status != ToolStatus::Running
                                && (t.tool == "read_file"
                                    || t.tool == "write_file"
                                    || t.tool == "edit_file")
                        })
                        .map(|t| t.target.clone())
                        .collect();
                    for p in recent_paths {
                        self.push_recent_file(&p);
                    }

                    // Ensure we always have a system prompt to guide the agent.
                    if !self.messages.iter().any(|m| m.role == Role::System) {
                        self.messages.insert(
                            0,
                            Message {
                                role: Role::System,
                                content: SYSTEM_PROMPT.into(),
                                reasoning: None,
                                tool_calls: None,
                                tool_group_id: None,
                            },
                        );
                    }

                    self.session = Some(store);
                    self.refresh_verify_suggestions();
                    // User-visible notice.
                    self.messages.push(Message {
                        role: Role::Agent,
                        content: "(resumed previous session)".into(),
                        reasoning: None,
                        tool_calls: None,
                        tool_group_id: None,
                    });
                    return;
                }
            }
        }

        self.new_session();
    }

    pub fn new_session(&mut self) {
        let session_id = format!("{}", crate::memory::types::unix_ts());
        if let Ok(store) = SessionStore::new(&self.workspace_root, session_id) {
            store.init_file();
            self.session = Some(store);
        }

        self.refresh_verify_suggestions();

        // Persist the current system/hello messages.
        if let Some(store) = &self.session {
            for m in &self.messages {
                store.record_message(m);
            }
        }
    }

    fn session_record_message(&self, msg: &Message) {
        if let Some(store) = &self.session {
            store.record_message(msg);
        }
    }

    fn session_record_tool(&self, tool: &ToolOutput) {
        if let Some(store) = &self.session {
            if tool.status != ToolStatus::Running {
                store.record_tool(tool);
            }
        }
    }
    fn refresh_verify_suggestions(&mut self) {
        self.verify_suggestions = detect_suggestions(&self.workspace_root);
    }

    /// Start background indexing of the current directory
    pub fn start_background_indexing(&self) {
        let tx = self.event_tx.clone();
        let policy = self.sandbox_policy.clone();
        let workspace_root = self.workspace_root.clone();
        tokio::spawn(async move {
            let _ = tx.send(AppEvent::IndexingStarted);

            // Clone tx for the blocking task, keep original for error handling
            let tx_for_blocking = tx.clone();

            // Run indexing in a blocking task since it's CPU-intensive
            let result = tokio::task::spawn_blocking(move || {
                run_background_index(tx_for_blocking, policy, workspace_root)
            })
            .await;

            if let Err(e) = result {
                // Task panicked
                let _ = tx.send(AppEvent::IndexingError(format!(
                    "Indexing task failed: {}",
                    e
                )));
            }
            // Success/error cases already send events from within run_background_index
        });
    }

    fn adjust_split_ratio(&mut self, delta: i16) {
        const MIN_SPLIT: i16 = 20;
        const MAX_SPLIT: i16 = 80;
        let next = (self.split_ratio as i16 + delta).clamp(MIN_SPLIT, MAX_SPLIT);
        self.split_ratio = next as u16;
    }

    fn submit_message(&mut self) {
        if self.input.trim().is_empty() || self.is_processing {
            return;
        }

        let user_msg = std::mem::take(&mut self.input);
        let user_msg_for_mem = user_msg.clone();
        self.cursor_pos = 0;

        // Each submitted user message is a new turn.
        self.current_turn_id = self.current_turn_id.saturating_add(1);
        self.turn_tool_start_idx = self.tool_outputs.len();
        self.turn_user_message = Some(user_msg_for_mem.clone());

        self.messages.push(Message {
            role: Role::User,
            content: user_msg,
            reasoning: None,
            tool_calls: None,
            tool_group_id: None,
        });

        if let Some(last) = self.messages.last() {
            self.session_record_message(last);
        }

        let last_user = self
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        if self.maybe_handle_command(&last_user) {
            self.scroll_messages_to_bottom();
            return;
        }

        let memory_enabled = self
            .config
            .memory
            .as_ref()
            .and_then(|m| m.enabled)
            .unwrap_or(true);
        let learn_user = memory_enabled
            && self
                .config
                .memory
                .as_ref()
                .and_then(|m| m.auto_learn_user)
                .unwrap_or(true);

        if learn_user {
            let prev_agent = self
                .messages
                .iter()
                .rev()
                .find(|m| m.role == Role::Agent)
                .map(|m| m.content.clone());

            // Best-effort user preference learning (async, does not block UI).
            let memory = self.memory.clone();
            tokio::spawn(async move {
                memory
                    .on_user_message(&user_msg_for_mem, prev_agent.as_deref())
                    .await;
            });
        }

        self.scroll_messages_to_bottom();

        self.start_llm_call();
    }

    fn start_llm_call(&mut self) {
        self.is_processing = true;
        self.processing_start = Some(Instant::now());
        self.current_response.clear();
        self.current_reasoning.clear();
        self.messages_auto_scroll = true;

        let user_message = self
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let memory_enabled = self
            .config
            .memory
            .as_ref()
            .and_then(|m| m.enabled)
            .unwrap_or(true);
        let inject_memory = memory_enabled
            && self
                .config
                .memory
                .as_ref()
                .and_then(|m| m.auto_inject)
                .unwrap_or(true);

        // Build a per-turn LLM message list with ephemeral memory injection.
        // This keeps memory out of the persisted transcript and avoids blocking the UI.
        let base_chat_messages: Vec<ChatMessage> = self
            .messages
            .iter()
            .filter(|m| !(m.role == Role::System && m.content.starts_with("\n[Memory]\n")))
            .map(|m| m.to_chat_message())
            .collect();

        let tx = self.event_tx.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        let memory = self.memory.clone();

        tokio::spawn(async move {
            let mut chat_messages = base_chat_messages;

            if inject_memory {
                let memory_context = memory.build_injection_context(&user_message, &[]).await;

                if !memory_context.is_empty() {
                    // Insert right after the first system prompt (if present).
                    let insert_at = chat_messages
                        .iter()
                        .position(|m| m.role == "system")
                        .map(|idx| idx + 1)
                        .unwrap_or(0);

                    chat_messages.insert(
                        insert_at,
                        ChatMessage {
                            role: "system".into(),
                            content: Some(memory_context),
                            tool_calls: None,
                            tool_call_id: None,
                            name: None,
                        },
                    );
                }
            }

            call_llm(tx, api_key, model, chat_messages).await;
        });
    }

    fn scroll_messages_to_bottom(&mut self) {
        self.messages_auto_scroll = true;
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if self.settings_open {
            self.handle_settings_key(key);
            return;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Left => {
                    self.adjust_split_ratio(-2);
                    return;
                }
                KeyCode::Right => {
                    self.adjust_split_ratio(2);
                    return;
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Esc => self.should_quit = true,
            KeyCode::Tab => {
                self.active_pane = match self.active_pane {
                    Pane::Chat => Pane::Context,
                    Pane::Context => Pane::Chat,
                };
            }
            KeyCode::Enter => self.submit_message(),
            KeyCode::Backspace => {
                if self.cursor_pos > 0 {
                    self.cursor_pos -= 1;
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Delete => {
                if self.cursor_pos < self.input.len() {
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
            }
            KeyCode::Right => {
                self.cursor_pos = (self.cursor_pos + 1).min(self.input.len());
            }
            KeyCode::Home => self.cursor_pos = 0,
            KeyCode::End => self.cursor_pos = self.input.len(),
            KeyCode::Up => {
                if key.modifiers.contains(KeyModifiers::SHIFT) {
                    match self.active_pane {
                        Pane::Chat => {
                            self.messages_auto_scroll = false;
                            self.messages_scroll = self.messages_scroll.saturating_sub(1);
                        }
                        Pane::Context => {}
                    }
                }
            }
            KeyCode::Down => {
                if key.modifiers.contains(KeyModifiers::SHIFT) {
                    match self.active_pane {
                        Pane::Chat => {
                            self.messages_scroll = self.messages_scroll.saturating_add(1);
                            // Re-enable auto-scroll if at bottom (will be clamped in UI)
                        }
                        Pane::Context => {}
                    }
                }
            }
            KeyCode::PageUp => match self.active_pane {
                Pane::Chat => {
                    self.messages_auto_scroll = false;
                    self.messages_scroll = self.messages_scroll.saturating_sub(20);
                }
                Pane::Context => {}
            },
            KeyCode::PageDown => match self.active_pane {
                Pane::Chat => {
                    self.messages_scroll = self.messages_scroll.saturating_add(20);
                }
                Pane::Context => {}
            },
            KeyCode::Char('e') => {
                if key.modifiers.contains(KeyModifiers::CONTROL) && self.active_pane == Pane::Chat {
                    if let Some(group_id) = self.last_tool_group_id {
                        let has_trace = self.tool_outputs.iter().any(|t| t.group_id == group_id);
                        if has_trace {
                            let cur = self
                                .tool_trace_expanded
                                .get(&group_id)
                                .copied()
                                .unwrap_or(false);
                            self.tool_trace_expanded.insert(group_id, !cur);
                        }
                    }
                    return;
                }
                self.input.insert(self.cursor_pos, 'e');
                self.cursor_pos += 1;
            }
            KeyCode::Char('i') => {
                if key.modifiers.contains(KeyModifiers::CONTROL) && self.active_pane == Pane::Chat {
                    if let Some(group_id) = self.last_tool_group_id {
                        let has_trace = self.tool_outputs.iter().any(|t| t.group_id == group_id);
                        if has_trace {
                            let cur = self
                                .tool_trace_show_details
                                .get(&group_id)
                                .copied()
                                .unwrap_or(true);
                            self.tool_trace_show_details.insert(group_id, !cur);
                        }
                    }
                    return;
                }
                self.input.insert(self.cursor_pos, 'i');
                self.cursor_pos += 1;
            }
            KeyCode::Char(c) => {
                self.input.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
            }
            _ => {}
        }
    }

    fn handle_settings_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.settings_open = false;
                self.settings_input.clear();
                self.settings_cursor = 0;
            }
            KeyCode::Up => {
                if self.settings_selected > 0 {
                    self.settings_selected -= 1;
                    self.load_settings_input();
                }
            }
            KeyCode::Down => {
                if self.settings_selected + 1 < self.settings_items().len() {
                    self.settings_selected += 1;
                    self.load_settings_input();
                }
            }
            KeyCode::Tab => {
                self.settings_selected = (self.settings_selected + 1) % self.settings_items().len();
                self.load_settings_input();
            }
            KeyCode::Backspace => {
                if self.settings_cursor > 0 {
                    self.settings_cursor -= 1;
                    self.settings_input.remove(self.settings_cursor);
                }
            }
            KeyCode::Left => {
                self.settings_cursor = self.settings_cursor.saturating_sub(1);
            }
            KeyCode::Right => {
                if self.settings_cursor < self.settings_input.len() {
                    self.settings_cursor += 1;
                }
            }
            KeyCode::Char(c) => {
                self.settings_input.insert(self.settings_cursor, c);
                self.settings_cursor += 1;
            }
            KeyCode::Enter => {
                self.apply_settings_input();
                let _ = self.settings_draft.save();
                self.config = self.settings_draft.clone();
                self.sandbox_policy = Arc::new(SandboxPolicy::from_config(
                    self.config.clone(),
                    self.workspace_root.clone(),
                    crate::tools::TOOL_NAMES,
                ));
                if self
                    .config
                    .general
                    .as_ref()
                    .and_then(|g| g.auto_index)
                    .unwrap_or(true)
                    && matches!(self.indexing_status, IndexingStatus::NotStarted)
                    && !index_file_exists(&self.workspace_root)
                {
                    self.start_background_indexing();
                }
                self.settings_open = false;
            }
            _ => {}
        }
    }

    fn settings_items(&self) -> Vec<SettingsItem> {
        vec![
            SettingsItem::Model,
            SettingsItem::SplitRatio,
            SettingsItem::AutoIndex,
            SettingsItem::ResumeLastSession,
            SettingsItem::MemoryEnabled,
            SettingsItem::MemoryAutoInject,
            SettingsItem::MemoryAutoLearnFailures,
            SettingsItem::MemoryAutoLearnUser,
            SettingsItem::MemoryAutoExtract,
            SettingsItem::MemoryExtractionModel,
            SettingsItem::SandboxEnabled,
            SettingsItem::SandboxRoot,
            SettingsItem::SandboxAllowPaths,
            SettingsItem::SandboxDenyPaths,
            SettingsItem::SandboxAllowCommands,
            SettingsItem::SandboxAllowTools,
        ]
    }

    fn load_settings_input(&mut self) {
        let current = self.read_settings_value(self.settings_items()[self.settings_selected]);
        self.settings_input = current;
        self.settings_cursor = self.settings_input.len();
    }

    fn read_settings_value(&self, item: SettingsItem) -> String {
        match item {
            SettingsItem::Model => self
                .settings_draft
                .general
                .as_ref()
                .and_then(|g| g.model.clone())
                .unwrap_or_else(|| crate::llm::MODEL.to_string()),
            SettingsItem::SplitRatio => self
                .settings_draft
                .general
                .as_ref()
                .and_then(|g| g.split_ratio)
                .map(|v| v.to_string())
                .unwrap_or_else(|| self.split_ratio.to_string()),
            SettingsItem::AutoIndex => self
                .settings_draft
                .general
                .as_ref()
                .and_then(|g| g.auto_index)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "true".to_string()),
            SettingsItem::ResumeLastSession => self
                .settings_draft
                .general
                .as_ref()
                .and_then(|g| g.resume_last)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "true".to_string()),
            SettingsItem::MemoryEnabled => self
                .settings_draft
                .memory
                .as_ref()
                .and_then(|m| m.enabled)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "true".to_string()),
            SettingsItem::MemoryAutoInject => self
                .settings_draft
                .memory
                .as_ref()
                .and_then(|m| m.auto_inject)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "true".to_string()),
            SettingsItem::MemoryAutoLearnFailures => self
                .settings_draft
                .memory
                .as_ref()
                .and_then(|m| m.auto_learn_failures)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "true".to_string()),
            SettingsItem::MemoryAutoLearnUser => self
                .settings_draft
                .memory
                .as_ref()
                .and_then(|m| m.auto_learn_user)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "true".to_string()),
            SettingsItem::MemoryAutoExtract => self
                .settings_draft
                .memory
                .as_ref()
                .and_then(|m| m.auto_extract)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "false".to_string()),
            SettingsItem::MemoryExtractionModel => self
                .settings_draft
                .memory
                .as_ref()
                .and_then(|m| m.extraction_model.clone())
                .unwrap_or_default(),
            SettingsItem::SandboxEnabled => self
                .settings_draft
                .sandbox
                .as_ref()
                .and_then(|s| s.enabled)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "true".to_string()),
            SettingsItem::SandboxRoot => self
                .settings_draft
                .sandbox
                .as_ref()
                .and_then(|s| s.root.clone())
                .map(|v| v.display().to_string())
                .unwrap_or_default(),
            SettingsItem::SandboxAllowPaths => self
                .settings_draft
                .sandbox
                .as_ref()
                .and_then(|s| s.allow_paths.clone())
                .map(|paths| join_paths(&paths))
                .unwrap_or_default(),
            SettingsItem::SandboxDenyPaths => self
                .settings_draft
                .sandbox
                .as_ref()
                .and_then(|s| s.deny_paths.clone())
                .map(|paths| join_paths(&paths))
                .unwrap_or_default(),
            SettingsItem::SandboxAllowCommands => self
                .settings_draft
                .sandbox
                .as_ref()
                .and_then(|s| s.allow_commands.clone())
                .map(|list| list.join(", "))
                .unwrap_or_default(),
            SettingsItem::SandboxAllowTools => self
                .settings_draft
                .sandbox
                .as_ref()
                .and_then(|s| s.allow_tools.clone())
                .map(|list| list.join(", "))
                .unwrap_or_default(),
        }
    }

    fn apply_settings_input(&mut self) {
        let item = self.settings_items()[self.settings_selected];
        match item {
            SettingsItem::Model => {
                let mut general = self.settings_draft.general.clone().unwrap_or_default();
                let value = self.settings_input.trim().to_string();
                general.model = Some(value.clone());
                self.settings_draft.general = Some(general);
                self.model = value;
            }
            SettingsItem::SplitRatio => {
                if let Ok(val) = self.settings_input.trim().parse::<u16>() {
                    let mut general = self.settings_draft.general.clone().unwrap_or_default();
                    general.split_ratio = Some(val.clamp(20, 80));
                    self.settings_draft.general = Some(general);
                    self.split_ratio = val.clamp(20, 80);
                }
            }
            SettingsItem::AutoIndex => {
                if let Ok(val) = parse_bool(&self.settings_input) {
                    let mut general = self.settings_draft.general.clone().unwrap_or_default();
                    general.auto_index = Some(val);
                    self.settings_draft.general = Some(general);
                }
            }
            SettingsItem::ResumeLastSession => {
                if let Ok(val) = parse_bool(&self.settings_input) {
                    let mut general = self.settings_draft.general.clone().unwrap_or_default();
                    general.resume_last = Some(val);
                    self.settings_draft.general = Some(general);
                }
            }
            SettingsItem::MemoryEnabled => {
                let mut mem = self.settings_draft.memory.clone().unwrap_or_default();
                if let Ok(v) = parse_bool(&self.settings_input) {
                    mem.enabled = Some(v);
                }
                self.settings_draft.memory = Some(mem);
            }
            SettingsItem::MemoryAutoInject => {
                let mut mem = self.settings_draft.memory.clone().unwrap_or_default();
                if let Ok(v) = parse_bool(&self.settings_input) {
                    mem.auto_inject = Some(v);
                }
                self.settings_draft.memory = Some(mem);
            }
            SettingsItem::MemoryAutoLearnFailures => {
                let mut mem = self.settings_draft.memory.clone().unwrap_or_default();
                if let Ok(v) = parse_bool(&self.settings_input) {
                    mem.auto_learn_failures = Some(v);
                }
                self.settings_draft.memory = Some(mem);
            }
            SettingsItem::MemoryAutoLearnUser => {
                let mut mem = self.settings_draft.memory.clone().unwrap_or_default();
                if let Ok(v) = parse_bool(&self.settings_input) {
                    mem.auto_learn_user = Some(v);
                }
                self.settings_draft.memory = Some(mem);
            }
            SettingsItem::MemoryAutoExtract => {
                let mut mem = self.settings_draft.memory.clone().unwrap_or_default();
                if let Ok(v) = parse_bool(&self.settings_input) {
                    mem.auto_extract = Some(v);
                }
                self.settings_draft.memory = Some(mem);
            }
            SettingsItem::MemoryExtractionModel => {
                let mut mem = self.settings_draft.memory.clone().unwrap_or_default();
                let v = self.settings_input.trim();
                if v.is_empty() {
                    mem.extraction_model = None;
                } else {
                    mem.extraction_model = Some(v.to_string());
                }
                self.settings_draft.memory = Some(mem);
            }

            SettingsItem::SandboxEnabled => {
                if let Ok(val) = parse_bool(&self.settings_input) {
                    let mut sandbox = self.settings_draft.sandbox.clone().unwrap_or_default();
                    sandbox.enabled = Some(val);
                    self.settings_draft.sandbox = Some(sandbox);
                }
            }
            SettingsItem::SandboxRoot => {
                let mut sandbox = self.settings_draft.sandbox.clone().unwrap_or_default();
                if self.settings_input.trim().is_empty() {
                    sandbox.root = None;
                } else {
                    sandbox.root = Some(PathBuf::from(self.settings_input.trim()));
                }
                self.settings_draft.sandbox = Some(sandbox);
            }
            SettingsItem::SandboxAllowPaths => {
                let mut sandbox = self.settings_draft.sandbox.clone().unwrap_or_default();
                sandbox.allow_paths = Some(split_paths(&self.settings_input));
                self.settings_draft.sandbox = Some(sandbox);
            }
            SettingsItem::SandboxDenyPaths => {
                let mut sandbox = self.settings_draft.sandbox.clone().unwrap_or_default();
                sandbox.deny_paths = Some(split_paths(&self.settings_input));
                self.settings_draft.sandbox = Some(sandbox);
            }
            SettingsItem::SandboxAllowCommands => {
                let mut sandbox = self.settings_draft.sandbox.clone().unwrap_or_default();
                sandbox.allow_commands = Some(split_list(&self.settings_input));
                self.settings_draft.sandbox = Some(sandbox);
            }
            SettingsItem::SandboxAllowTools => {
                let mut sandbox = self.settings_draft.sandbox.clone().unwrap_or_default();
                sandbox.allow_tools = Some(split_list(&self.settings_input));
                self.settings_draft.sandbox = Some(sandbox);
            }
        }
    }

    pub fn open_settings(&mut self) {
        self.settings_open = true;
        self.settings_selected = 0;
        self.settings_draft = self.config.clone();
        self.settings_input.clear();
        self.settings_cursor = 0;
        self.load_settings_input();
    }

    pub fn maybe_handle_command(&mut self, input: &str) -> bool {
        let trimmed = input.trim();
        if trimmed == "/settings" || trimmed == "/ settings" {
            self.open_settings();
            return true;
        }
        if trimmed == "/new" {
            // Start a fresh session but keep the system prompt.
            self.messages.retain(|m| m.role == Role::System);
            self.tool_outputs.clear();
            self.tool_index_by_call_id.clear();
            self.tool_group_by_call_id.clear();
            self.tool_trace_expanded.clear();
            self.tool_trace_show_details.clear();
            self.tool_failure_counts.clear();
            self.tool_loop_abort = None;
            self.recent_files.clear();
            self.last_searches.clear();
            self.turn_user_message = None;
            self.turn_tool_start_idx = 0;
            self.current_turn_id = 0;
            self.next_tool_group_id = 1;
            self.last_tool_group_id = None;
            self.new_session();
            self.messages.push(Message {
                role: Role::Agent,
                content: "(new session)".into(),
                reasoning: None,
                tool_calls: None,
                tool_group_id: None,
            });
            self.scroll_messages_to_bottom();
            return true;
        }

        if trimmed == "/resume" {
            self.init_session(true);
            self.scroll_messages_to_bottom();
            return true;
        }

        if trimmed == "/verify" {
            self.refresh_verify_suggestions();
            if self.verify_suggestions.is_empty() {
                self.messages.push(Message {
                    role: Role::Agent,
                    content: "No verify suggestions for this workspace yet.".into(),
                    reasoning: None,
                    tool_calls: None,
                    tool_group_id: None,
                });
                self.scroll_messages_to_bottom();
                if let Some(last) = self.messages.last() {
                    self.session_record_message(last);
                }
                return true;
            }

            let cmds = self
                .verify_suggestions
                .iter()
                .take(2)
                .map(|s| s.command.clone())
                .collect::<Vec<_>>()
                .join(" && ");

            let group_id = self.next_tool_group_id;
            self.next_tool_group_id = self.next_tool_group_id.saturating_add(1);
            self.last_tool_group_id = Some(group_id);

            self.messages.push(Message {
                role: Role::Agent,
                content: format!("Running verify: {}", cmds),
                reasoning: None,
                tool_calls: None,
                tool_group_id: Some(group_id),
            });
            self.scroll_messages_to_bottom();
            if let Some(last) = self.messages.last() {
                self.session_record_message(last);
            }

            // Execute via bash tool (respects sandbox).
            let tx = self.event_tx.clone();
            let policy = self.sandbox_policy.clone();
            let call_id = format!("internal:verify:{}", crate::memory::types::unix_ts());
            self.tool_group_by_call_id.insert(call_id.clone(), group_id);
            let args_raw = serde_json::json!({"command": cmds}).to_string();
            let args_val: serde_json::Value = serde_json::from_str(&args_raw)
                .unwrap_or_else(|_| serde_json::json!({"command": cmds}));
            let args_summary = summarize_tool_call("bash", &args_val);
            let sandbox = sandbox_decision_for_tool("bash", &args_val, &policy);

            let _ = tx.send(AppEvent::ToolStart(crate::events::ToolStartEvent {
                call_id: call_id.clone(),
                tool: "bash".to_string(),
                args_raw: args_raw.clone(),
                args_summary,
                cwd: policy.root.clone(),
                sandbox: sandbox.clone(),
            }));

            if !sandbox.allowed {
                let msg = sandbox
                    .reason
                    .clone()
                    .unwrap_or_else(|| "Sandbox: blocked".to_string());
                let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                    call_id: call_id.clone(),
                    chunk: msg.clone(),
                }));
                let _ = tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                    call_id,
                    success: false,
                }));
                return true;
            }

            tokio::spawn(async move {
                let _ = crate::tools::execute_tool("bash", &args_raw, &call_id, &tx, &policy).await;
            });
            return true;
        }

        if trimmed == "/sessions" {
            let msg = if let Some(store) = &self.session {
                format!(
                    "Sessions dir: {}",
                    store
                        .events_path
                        .parent()
                        .unwrap_or_else(|| std::path::Path::new(""))
                        .display()
                )
            } else {
                "No session store initialized".to_string()
            };
            self.messages.push(Message {
                role: Role::Agent,
                content: msg,
                reasoning: None,
                tool_calls: None,
                tool_group_id: None,
            });
            self.scroll_messages_to_bottom();
            if let Some(last) = self.messages.last() {
                self.session_record_message(last);
            }
            return true;
        }
        false
    }

    pub fn settings_rows(&self) -> Vec<(String, String)> {
        self.settings_items()
            .iter()
            .map(|item| (item.label().to_string(), self.read_settings_value(*item)))
            .collect()
    }

    fn handle_mouse(&mut self, mouse: MouseEvent, chat_area: Rect, context_area: Rect) {
        let in_chat = mouse.column >= chat_area.x
            && mouse.column < chat_area.right()
            && mouse.row >= chat_area.y
            && mouse.row < chat_area.bottom();
        let in_context = mouse.column >= context_area.x
            && mouse.column < context_area.right()
            && mouse.row >= context_area.y
            && mouse.row < context_area.bottom();
        let in_splitter = mouse.column >= self.splitter_area.x
            && mouse.column < self.splitter_area.right()
            && mouse.row >= self.splitter_area.y
            && mouse.row < self.splitter_area.bottom();

        match mouse.kind {
            MouseEventKind::ScrollUp => {
                if in_chat {
                    self.messages_auto_scroll = false;
                    self.messages_scroll = self.messages_scroll.saturating_sub(1);
                } else if in_context {
                }
            }
            MouseEventKind::ScrollDown => {
                if in_chat {
                    self.messages_scroll = self.messages_scroll.saturating_add(1);
                } else if in_context {
                }
            }
            MouseEventKind::Down(_) => {
                // Click to focus pane
                if in_chat {
                    self.active_pane = Pane::Chat;
                } else if in_context {
                    self.active_pane = Pane::Context;
                } else if in_splitter {
                    self.dragging_splitter = true;
                }
            }
            MouseEventKind::Up(MouseButton::Left) => {
                self.dragging_splitter = false;
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if self.dragging_splitter && self.root_area.width > 0 {
                    let x = mouse.column.saturating_sub(self.root_area.x);
                    let ratio = (x as f32 / self.root_area.width as f32 * 100.0).round() as i16;
                    self.split_ratio = ratio.clamp(20, 80) as u16;
                }
            }
            _ => {}
        }
    }

    pub fn handle_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Input(key) => self.handle_key(key),
            AppEvent::Mouse(mouse) => {
                self.handle_mouse(mouse, self.chat_area, self.context_area);
            }
            AppEvent::AgentChunk(chunk) => {
                self.current_response.push_str(&chunk);
            }
            AppEvent::AgentReasoning(reasoning) => {
                self.current_reasoning.push_str(&reasoning);
            }
            AppEvent::AgentDone => {
                let response = std::mem::take(&mut self.current_response);
                let reasoning = std::mem::take(&mut self.current_reasoning);
                let response_for_mem = response.clone();

                if !response.is_empty() || !reasoning.is_empty() {
                    self.messages.push(Message {
                        role: Role::Agent,
                        content: response,
                        reasoning: if reasoning.is_empty() {
                            None
                        } else {
                            Some(reasoning)
                        },
                        tool_calls: None,
                        tool_group_id: None,
                    });
                    self.scroll_messages_to_bottom();
                    if let Some(last) = self.messages.last() {
                        self.session_record_message(last);
                    }
                }

                // Optional: LLM-based extraction of durable long-term memories.
                let memory_enabled = self
                    .config
                    .memory
                    .as_ref()
                    .and_then(|m| m.enabled)
                    .unwrap_or(true);
                let auto_extract = memory_enabled
                    && self
                        .config
                        .memory
                        .as_ref()
                        .and_then(|m| m.auto_extract)
                        .unwrap_or(false);

                let user_message = self.turn_user_message.take().unwrap_or_else(|| {
                    self.messages
                        .iter()
                        .rev()
                        .find(|m| m.role == Role::User)
                        .map(|m| m.content.clone())
                        .unwrap_or_default()
                });

                let tool_outputs: Vec<ToolOutput> =
                    if self.turn_tool_start_idx < self.tool_outputs.len() {
                        self.tool_outputs[self.turn_tool_start_idx..].to_vec()
                    } else {
                        Vec::new()
                    };
                self.turn_tool_start_idx = self.tool_outputs.len();

                if auto_extract
                    && !user_message.trim().is_empty()
                    && (!response_for_mem.trim().is_empty() || !tool_outputs.is_empty())
                {
                    let summary =
                        build_turn_summary(&user_message, &response_for_mem, &tool_outputs);
                    let extraction_model = self
                        .config
                        .memory
                        .as_ref()
                        .and_then(|m| m.extraction_model.clone())
                        .unwrap_or_else(|| self.model.clone());
                    let api_key = self.api_key.clone();
                    let memory = self.memory.clone();

                    tokio::spawn(async move {
                        let _ = memory
                            .llm_extract_and_save(api_key, extraction_model, summary)
                            .await;
                    });
                }

                self.is_processing = false;
                self.processing_start = None;
            }
            AppEvent::AgentToolCalls(tool_calls) => {
                // Save assistant message with tool calls
                let response = std::mem::take(&mut self.current_response);
                let reasoning = std::mem::take(&mut self.current_reasoning);

                // Each tool-call phase gets its own group id so tool traces can be interleaved
                // with the assistant's narrative without duplicating previous tool phases.
                let group_id = self.next_tool_group_id;
                self.next_tool_group_id = self.next_tool_group_id.saturating_add(1);
                self.last_tool_group_id = Some(group_id);
                for tc in &tool_calls {
                    self.tool_group_by_call_id.insert(tc.id.clone(), group_id);
                }

                self.messages.push(Message {
                    role: Role::Agent,
                    content: response,
                    reasoning: if reasoning.is_empty() {
                        None
                    } else {
                        Some(reasoning)
                    },
                    tool_calls: Some(tool_calls.clone()),
                    tool_group_id: Some(group_id),
                });
                self.scroll_messages_to_bottom();
                if let Some(last) = self.messages.last() {
                    self.session_record_message(last);
                }

                // Execute tools and continue
                let tx = self.event_tx.clone();
                let policy = self.sandbox_policy.clone();
                let memory = self.memory.clone();

                tokio::spawn(async move {
                    let mut tool_results = Vec::new();

                    for tool_call in &tool_calls {
                        let call_id = tool_call.id.clone();
                        let name = tool_call.function.name.as_str();
                        let args_raw = tool_call.function.arguments.clone();

                        let args_val: serde_json::Value = match serde_json::from_str(&args_raw) {
                            Ok(v) => v,
                            Err(e) => {
                                // Still show the invocation row for auditability.
                                let _ =
                                    tx.send(AppEvent::ToolStart(crate::events::ToolStartEvent {
                                        call_id: call_id.clone(),
                                        tool: name.to_string(),
                                        args_raw: args_raw.clone(),
                                        args_summary: "<invalid json>".to_string(),
                                        cwd: policy.root.clone(),
                                        sandbox: crate::sandbox::SandboxDecision::allow(),
                                    }));
                                let msg = format!("Error parsing arguments: {}", e);
                                let _ =
                                    tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                                        call_id: call_id.clone(),
                                        chunk: msg.clone(),
                                    }));
                                let _ = tx.send(AppEvent::ToolComplete(
                                    crate::events::ToolCompleteEvent {
                                        call_id: call_id.clone(),
                                        success: false,
                                    },
                                ));
                                tool_results.push((call_id, msg));
                                continue;
                            }
                        };

                        let args_summary = summarize_tool_call(name, &args_val);
                        let sandbox = sandbox_decision_for_tool(name, &args_val, &policy);

                        let _ = tx.send(AppEvent::ToolStart(crate::events::ToolStartEvent {
                            call_id: call_id.clone(),
                            tool: name.to_string(),
                            args_raw: args_raw.clone(),
                            args_summary: args_summary.clone(),
                            cwd: policy.root.clone(),
                            sandbox: sandbox.clone(),
                        }));

                        if !sandbox.allowed {
                            let msg = sandbox
                                .reason
                                .clone()
                                .unwrap_or_else(|| "Sandbox: blocked".to_string());
                            let _ = tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                                call_id: call_id.clone(),
                                chunk: msg.clone(),
                            }));
                            let _ =
                                tx.send(AppEvent::ToolComplete(crate::events::ToolCompleteEvent {
                                    call_id: call_id.clone(),
                                    success: false,
                                }));
                            tool_results.push((call_id, msg));
                            continue;
                        }

                        let result = match name {
                            "memory_recall" | "memory_save" | "memory_list" | "memory_forget" => {
                                let out = match name {
                                    "memory_recall" => {
                                        let query = args_val
                                            .get("query")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let limit = args_val
                                            .get("limit")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(8)
                                            as usize;
                                        let types = args_val
                                            .get("types")
                                            .and_then(|v| v.as_array())
                                            .map(|arr| {
                                                arr.iter()
                                                    .filter_map(|x| x.as_str())
                                                    .filter_map(|s| s.parse().ok())
                                                    .collect::<Vec<crate::memory::MemoryType>>()
                                            });
                                        let results = memory
                                            .recall(query, limit, types)
                                            .await
                                            .unwrap_or_default();
                                        if results.is_empty() {
                                            "No memories.".to_string()
                                        } else {
                                            let mut out = String::new();
                                            for (i, sm) in results.iter().enumerate() {
                                                let m = &sm.memory;
                                                out.push_str(&format!(
                                                    "{}. {} [{}] ({:.2}) {}
",
                                                    i + 1,
                                                    m.id,
                                                    m.memory_type.as_str(),
                                                    sm.score,
                                                    m.content.replace('\n', " ")
                                                ));
                                            }
                                            out
                                        }
                                    }
                                    "memory_save" => {
                                        let mem_type = args_val
                                            .get("type")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("fact")
                                            .parse::<crate::memory::MemoryType>()
                                            .unwrap_or(crate::memory::MemoryType::Fact);
                                        let content = args_val
                                            .get("content")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let why = args_val.get("why").and_then(|v| v.as_str());
                                        let context =
                                            args_val.get("context").and_then(|v| v.as_str());
                                        let tags = args_val
                                            .get("tags")
                                            .and_then(|v| v.as_array())
                                            .map(|arr| {
                                                arr.iter()
                                                    .filter_map(|x| x.as_str())
                                                    .map(|s| s.to_string())
                                                    .collect::<Vec<String>>()
                                            })
                                            .unwrap_or_default();
                                        let scope = args_val
                                            .get("scope")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("project")
                                            .parse::<crate::memory::MemoryScope>()
                                            .unwrap_or(crate::memory::MemoryScope::Project);
                                        let confidence = args_val
                                            .get("confidence")
                                            .and_then(|v| v.as_f64())
                                            .map(|v| v as f32);
                                        let importance = args_val
                                            .get("importance")
                                            .and_then(|v| v.as_f64())
                                            .map(|v| v as f32);

                                        match memory
                                            .save_explicit(
                                                mem_type,
                                                content,
                                                why,
                                                context,
                                                tags,
                                                scope,
                                                None,
                                                crate::memory::MemorySource::User,
                                                confidence,
                                                importance,
                                            )
                                            .await
                                        {
                                            Ok(mem) => format!(
                                                "Saved memory {} [{}]",
                                                mem.id,
                                                mem.memory_type.as_str()
                                            ),
                                            Err(e) => format!("Error: {}", e),
                                        }
                                    }
                                    "memory_list" => {
                                        let limit = args_val
                                            .get("limit")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(30)
                                            as usize;
                                        let t =
                                            args_val.get("type").and_then(|v| v.as_str()).and_then(
                                                |s| s.parse::<crate::memory::MemoryType>().ok(),
                                            );
                                        let memories =
                                            memory.list(limit, t).await.unwrap_or_default();
                                        if memories.is_empty() {
                                            "No memories.".to_string()
                                        } else {
                                            let mut out = String::new();
                                            for m in memories {
                                                out.push_str(&format!(
                                                    "- {} [{}] {}
",
                                                    m.id,
                                                    m.memory_type.as_str(),
                                                    m.content.replace('\n', " ")
                                                ));
                                            }
                                            out
                                        }
                                    }
                                    "memory_forget" => {
                                        let id = args_val
                                            .get("id")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        match memory.forget(id).await {
                                            Ok(true) => format!("Forgot {}", id),
                                            Ok(false) => format!("Not found: {}", id),
                                            Err(e) => format!("Error: {}", e),
                                        }
                                    }
                                    _ => "Error: unknown memory tool".to_string(),
                                };
                                let success = !out.starts_with("Error:");
                                let _ =
                                    tx.send(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                                        call_id: call_id.clone(),
                                        chunk: out.clone(),
                                    }));
                                let _ = tx.send(AppEvent::ToolComplete(
                                    crate::events::ToolCompleteEvent {
                                        call_id: call_id.clone(),
                                        success,
                                    },
                                ));
                                out
                            }
                            _ => execute_tool(name, &args_raw, &call_id, &tx, &policy).await,
                        };

                        tool_results.push((call_id, result));
                    }

                    // Send tool results back for next LLM call
                    let _ = tx.send(AppEvent::ToolResultsReady(tool_results));
                });
            }
            AppEvent::ToolResultsReady(results) => {
                // Add tool result messages
                for (tool_call_id, result) in results {
                    self.messages.push(Message {
                        role: Role::Tool,
                        content: result,
                        reasoning: Some(tool_call_id), // Store tool_call_id here
                        tool_calls: None,
                        tool_group_id: None,
                    });
                }

                // If we detected an infinite tool retry loop, stop here and require user input.
                if let Some((turn_id, msg)) = self.tool_loop_abort.take() {
                    if turn_id == self.current_turn_id {
                        self.messages.push(Message {
                            role: Role::Agent,
                            content: msg,
                            reasoning: None,
                            tool_calls: None,
                            tool_group_id: None,
                        });
                        self.scroll_messages_to_bottom();
                        self.is_processing = false;
                        self.processing_start = None;
                        return;
                    }
                }

                // Continue the conversation
                self.start_llm_call();
            }
            AppEvent::AgentError(err) => {
                self.messages.push(Message {
                    role: Role::Agent,
                    content: format!("[Error: {}]", err),
                    reasoning: None,
                    tool_calls: None,
                    tool_group_id: None,
                });
                self.scroll_messages_to_bottom();
                self.is_processing = false;
                self.processing_start = None;
            }

            AppEvent::ToolStart(ev) => {
                let turn_id = self.current_turn_id;
                let group_id = self
                    .tool_group_by_call_id
                    .get(&ev.call_id)
                    .copied()
                    .unwrap_or(0);
                if group_id > 0 {
                    self.last_tool_group_id = Some(group_id);
                }

                let idx = self.tool_outputs.len();
                let tool_run = ToolOutput::new(
                    ev.call_id.clone(),
                    ev.tool,
                    ev.args_raw,
                    ev.args_summary,
                    ev.cwd,
                    ev.sandbox,
                    turn_id,
                    group_id,
                );
                self.tool_outputs.push(tool_run);
                self.tool_index_by_call_id.insert(ev.call_id, idx);

                // Default collapsed: show live one-line tails; expand with Ctrl+E / Ctrl+I when needed.
                if group_id > 0 {
                    self.tool_trace_expanded.entry(group_id).or_insert(false);
                    self.tool_trace_show_details.entry(group_id).or_insert(true);
                }
            }
            AppEvent::ToolOutput(ev) => {
                if let Some(&idx) = self.tool_index_by_call_id.get(&ev.call_id) {
                    if let Some(t) = self.tool_outputs.get_mut(idx) {
                        t.append_chunk(ev.chunk);
                    }
                }
            }
            AppEvent::ToolComplete(ev) => {
                let Some(&idx) = self.tool_index_by_call_id.get(&ev.call_id) else {
                    return;
                };

                let mut snapshot: Option<(String, String, String, bool, u64, u64)> = None;
                if let Some(t) = self.tool_outputs.get_mut(idx) {
                    t.complete(ev.success);
                    snapshot = Some((
                        t.tool.clone(),
                        t.target.clone(),
                        t.output.clone(),
                        ev.success,
                        t.turn_id,
                        t.group_id,
                    ));

                    // Persist completed tool output (best-effort).
                    let tool_snapshot = t.clone();
                    self.session_record_tool(&tool_snapshot);
                }

                let memory_enabled = self
                    .config
                    .memory
                    .as_ref()
                    .and_then(|m| m.enabled)
                    .unwrap_or(true);
                let learn_failures = memory_enabled
                    && self
                        .config
                        .memory
                        .as_ref()
                        .and_then(|m| m.auto_learn_failures)
                        .unwrap_or(true);

                self.refresh_verify_suggestions();

                // Auto-collapse the tool trace group when no tools are running for it.
                if let Some((tool, target, output, success, _turn_id, group_id)) = snapshot {
                    if group_id > 0 {
                        let any_running = self
                            .tool_outputs
                            .iter()
                            .any(|t| t.group_id == group_id && t.status == ToolStatus::Running);
                        if !any_running {
                            self.tool_trace_expanded.insert(group_id, false);
                        }
                    }

                    // Tool loop guard: if the same tool keeps failing with the same target in one turn,
                    // stop retrying and force the user/model to correct course.
                    if !success {
                        let key = (self.current_turn_id, format!("{}|{}", tool, target));
                        let count = self.tool_failure_counts.entry(key).or_insert(0);
                        *count = count.saturating_add(1);
                        if *count >= 3 && self.tool_loop_abort.is_none() {
                            self.tool_loop_abort = Some((
                                self.current_turn_id,
                                format!(
                                    "Tool loop detected: `{}` kept failing on `{}`. Stopping retries. Please provide the exact path/command or clarify the request.",
                                    tool, target
                                ),
                            ));
                        }
                    }

                    // Track recent files for the context sidebar.
                    if tool == "read_file" || tool == "write_file" || tool == "edit_file" {
                        self.push_recent_file(&target);
                    }

                    if learn_failures {
                        let memory = self.memory.clone();
                        tokio::spawn(async move {
                            memory
                                .on_tool_complete(&tool, &target, &output, success)
                                .await;
                        });
                    }
                }
            }

            // Indexing events
            AppEvent::IndexingStarted => {
                self.indexing_status = IndexingStatus::Indexing {
                    files_done: 0,
                    total_files: 0,
                };
            }
            AppEvent::IndexingProgress(files_done, total_files) => {
                self.indexing_status = IndexingStatus::Indexing {
                    files_done,
                    total_files,
                };
            }
            AppEvent::IndexingComplete(chunks, files) => {
                self.indexing_status = IndexingStatus::Complete { chunks, files };
            }
            AppEvent::IndexingError(err) => {
                self.indexing_status = IndexingStatus::Error(err);
            }
        }
    }

    fn push_recent_file(&mut self, path: &str) {
        if path.trim().is_empty() {
            return;
        }

        if let Some(pos) = self.recent_files.iter().position(|p| p == path) {
            self.recent_files.remove(pos);
        }
        self.recent_files.push_front(path.to_string());
        while self.recent_files.len() > 20 {
            self.recent_files.pop_back();
        }
    }

    pub fn display_messages(&self) -> impl Iterator<Item = &Message> {
        self.messages
            .iter()
            .filter(|m| m.role != Role::System && m.role != Role::Tool)
    }

    pub fn workspace_root_display(&self) -> String {
        self.workspace_root.display().to_string()
    }
}

#[derive(Debug, Clone, Copy)]
enum SettingsItem {
    Model,
    SplitRatio,
    AutoIndex,
    ResumeLastSession,
    SandboxEnabled,
    SandboxRoot,
    SandboxAllowPaths,
    SandboxDenyPaths,
    SandboxAllowCommands,
    SandboxAllowTools,
    MemoryEnabled,
    MemoryAutoInject,
    MemoryAutoLearnFailures,
    MemoryAutoLearnUser,
    MemoryAutoExtract,
    MemoryExtractionModel,
}

impl SettingsItem {
    fn label(&self) -> &'static str {
        match self {
            SettingsItem::Model => "Model",
            SettingsItem::SplitRatio => "Split ratio",
            SettingsItem::AutoIndex => "Auto index",
            SettingsItem::ResumeLastSession => "Resume last session",
            SettingsItem::SandboxEnabled => "Sandbox enabled",
            SettingsItem::SandboxRoot => "Sandbox root",
            SettingsItem::SandboxAllowPaths => "Sandbox allow paths",
            SettingsItem::SandboxDenyPaths => "Sandbox deny paths",
            SettingsItem::SandboxAllowCommands => "Sandbox allow commands",
            SettingsItem::SandboxAllowTools => "Sandbox allow tools",
            SettingsItem::MemoryEnabled => "Memory enabled",
            SettingsItem::MemoryAutoInject => "Memory auto inject",
            SettingsItem::MemoryAutoLearnFailures => "Memory learn failures",
            SettingsItem::MemoryAutoLearnUser => "Memory learn user",
            SettingsItem::MemoryAutoExtract => "Memory auto extract",
            SettingsItem::MemoryExtractionModel => "Memory extraction model",
        }
    }
}

fn parse_bool(input: &str) -> Result<bool, ()> {
    let v = input.trim().to_lowercase();
    match v.as_str() {
        "true" | "1" | "yes" | "y" | "on" => Ok(true),
        "false" | "0" | "no" | "n" | "off" => Ok(false),
        _ => Err(()),
    }
}

fn split_list(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn split_paths(input: &str) -> Vec<PathBuf> {
    input
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn join_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn pretty_args_lines(raw: &str) -> Vec<String> {
    // Avoid spewing huge JSON blobs (e.g. write_file content) into the UI details.
    // We parse JSON, replace very large strings with a placeholder, and pretty-print.
    const MAX_STRING: usize = 240;

    fn redact(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::String(s) => {
                if s.len() > MAX_STRING {
                    *s = format!("<{} chars>", s.len());
                }
            }
            serde_json::Value::Array(arr) => {
                for x in arr {
                    redact(x);
                }
            }
            serde_json::Value::Object(map) => {
                for (_k, x) in map.iter_mut() {
                    redact(x);
                }
            }
            _ => {}
        }
    }

    let mut v: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return vec![raw.to_string()],
    };
    redact(&mut v);
    match serde_json::to_string_pretty(&v) {
        Ok(s) => s.lines().map(|l| l.to_string()).collect(),
        Err(_) => vec![raw.to_string()],
    }
}

fn summarize_tool_call(name: &str, args: &serde_json::Value) -> String {
    fn trunc(s: &str, max: usize) -> String {
        let t = s.trim();
        if t.len() <= max {
            return t.to_string();
        }
        let mut out = t.to_string();
        out.truncate(max.saturating_sub(3));
        out.push_str("...");
        out
    }

    match name {
        "bash" => {
            let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
            trunc(cmd, 120)
        }
        "verify" => {
            let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
            if cmd.trim().is_empty() {
                "auto".to_string()
            } else {
                trunc(cmd, 120)
            }
        }
        "rg" => {
            let q = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            if q.is_empty() {
                format!("in {}", path)
            } else {
                trunc(&format!("{} in {}", q, path), 120)
            }
        }
        "read_file" | "write_file" | "list_files" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            trunc(path, 140)
        }
        "open_at" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            let line = args.get("line").and_then(|v| v.as_u64()).unwrap_or(1);
            trunc(&format!("{}:{}", path, line), 140)
        }
        "apply_patch" => {
            let patch = args.get("patch").and_then(|v| v.as_str()).unwrap_or("");
            if patch.trim().is_empty() {
                "patch".to_string()
            } else {
                // Keep this short; full patch is visible in details with redaction.
                "patch".to_string()
            }
        }
        "edit_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let old_str = args
                .get("old_string")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if old_str.is_empty() {
                trunc(path, 140)
            } else {
                trunc(
                    &format!("{} (replace: {})", path, old_str.replace('\n', " ")),
                    140,
                )
            }
        }
        "semantic_search" => {
            let q = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            trunc(q, 140)
        }
        "smart_search" => {
            let q = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            if q.is_empty() {
                format!("in {}", path)
            } else {
                trunc(&format!("{} in {}", q, path), 140)
            }
        }
        "memory_recall" => {
            let q = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            trunc(&format!("recall: {}", q), 140)
        }
        "memory_save" => {
            let t = args.get("type").and_then(|v| v.as_str()).unwrap_or("fact");
            let c = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            trunc(&format!("save({}): {}", t, c), 140)
        }
        "memory_list" => "list".to_string(),
        "memory_forget" => {
            let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("");
            trunc(&format!("forget: {}", id), 140)
        }
        _ => trunc(&args.to_string(), 120),
    }
}

fn sandbox_decision_for_tool(
    name: &str,
    args: &serde_json::Value,
    policy: &SandboxPolicy,
) -> crate::sandbox::SandboxDecision {
    use crate::sandbox::SandboxDecision;
    use std::path::Path;

    if let Err(e) = policy.check_tool_allowed(name) {
        return SandboxDecision::deny(e.to_string());
    }

    match name {
        "bash" => {
            let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
            if let Err(e) = policy.check_command_allowed(cmd) {
                return SandboxDecision::deny(e.to_string());
            }
            if let Err(e) = policy.check_bash_paths(cmd) {
                return SandboxDecision::deny(e.to_string());
            }
            SandboxDecision::allow()
        }
        "verify" => {
            let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
            // If omitted, tool will auto-detect; allow the tool to decide at runtime.
            if cmd.trim().is_empty() {
                return SandboxDecision::allow();
            }
            if let Err(e) = policy.check_command_allowed(cmd) {
                return SandboxDecision::deny(e.to_string());
            }
            if let Err(e) = policy.check_bash_paths(cmd) {
                return SandboxDecision::deny(e.to_string());
            }
            SandboxDecision::allow()
        }
        "rg" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            match policy.check_path_allowed(Path::new(path)) {
                Ok(_) => SandboxDecision::allow(),
                Err(e) => SandboxDecision::deny(e.to_string()),
            }
        }
        "open_at" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            match policy.check_path_allowed(Path::new(path)) {
                Ok(_) => SandboxDecision::allow(),
                Err(e) => SandboxDecision::deny(e.to_string()),
            }
        }
        "smart_search" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            match policy.check_path_allowed(Path::new(path)) {
                Ok(_) => SandboxDecision::allow(),
                Err(e) => SandboxDecision::deny(e.to_string()),
            }
        }
        "read_file" | "write_file" | "list_files" | "edit_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            match policy.check_path_allowed(Path::new(path)) {
                Ok(_) => SandboxDecision::allow(),
                Err(e) => SandboxDecision::deny(e.to_string()),
            }
        }
        "apply_patch" => {
            // Best-effort: scan for paths in the patch headers and validate them.
            let patch = args.get("patch").and_then(|v| v.as_str()).unwrap_or("");
            for line in patch.lines() {
                let path = if let Some(p) = line.strip_prefix("*** Add File: ") {
                    Some(p.trim())
                } else if let Some(p) = line.strip_prefix("*** Update File: ") {
                    Some(p.trim())
                } else if let Some(p) = line.strip_prefix("*** Delete File: ") {
                    Some(p.trim())
                } else {
                    None
                };
                if let Some(p) = path {
                    if let Err(e) = policy.check_path_allowed(Path::new(p)) {
                        return SandboxDecision::deny(e.to_string());
                    }
                }
            }
            SandboxDecision::allow()
        }
        _ => SandboxDecision::allow(),
    }
}

fn build_turn_summary(user_message: &str, agent_response: &str, tools: &[ToolOutput]) -> String {
    let mut out = String::new();

    out.push_str("## User Message\n");
    out.push_str(user_message.trim());
    out.push_str("\n\n");

    if !tools.is_empty() {
        out.push_str("## Tool Calls\n");
        for t in tools {
            let status = match t.status {
                ToolStatus::Running => "RUNNING",
                ToolStatus::Success => "SUCCESS",
                ToolStatus::Error => "FAILED",
            };

            out.push_str(&format!(
                "- {} ({}) {}\n",
                t.tool,
                status,
                truncate_for_summary(&t.target, 140)
            ));

            if !t.output.is_empty() {
                let first = t.output.lines().next().unwrap_or("").trim();
                if !first.is_empty() {
                    out.push_str(&format!("  Output: {}\n", truncate_for_summary(first, 180)));
                }
            }
        }
        out.push('\n');
    }

    out.push_str("## Agent Response\n");
    out.push_str(agent_response.trim());

    out
}

fn truncate_for_summary(s: &str, max: usize) -> String {
    let t = s.trim();
    if t.len() <= max {
        return t.to_string();
    }
    if max <= 3 {
        return "...".to_string();
    }
    let mut out = t.to_string();
    out.truncate(max.saturating_sub(3));
    out.push_str("...");
    out
}

/// Run background indexing - called from a blocking task
fn run_background_index(
    tx: mpsc::UnboundedSender<AppEvent>,
    policy: Arc<SandboxPolicy>,
    workspace_root: PathBuf,
) -> Result<(usize, usize), String> {
    // Catch any panics from dependencies
    let result = std::panic::catch_unwind(|| {
        let cwd = match std::env::current_dir() {
            Ok(d) => d,
            Err(e) => return Err(format!("cwd: {}", e)),
        };

        let checked_root = match policy.check_path_allowed(&cwd) {
            Ok(p) => p,
            Err(e) => return Err(e.to_string()),
        };

        let cfg = SearchConfig::for_workspace(&workspace_root);
        let search = match SemanticSearch::new(cfg) {
            Ok(s) => s,
            Err(e) => return Err(format!("init: {}", e)),
        };

        match search.index_directory(&checked_root) {
            Ok(stats) => Ok((stats.total_chunks, stats.total_files)),
            Err(e) => Err(format!("index: {}", e)),
        }
    });

    match result {
        Ok(Ok((chunks, files))) => {
            let _ = tx.send(AppEvent::IndexingComplete(chunks, files));
            Ok((chunks, files))
        }
        Ok(Err(e)) => {
            let _ = tx.send(AppEvent::IndexingError(e.clone()));
            Err(e)
        }
        Err(_) => {
            let err = "indexing panicked".to_string();
            let _ = tx.send(AppEvent::IndexingError(err.clone()));
            Err(err)
        }
    }
}

fn index_file_exists(workspace_root: &std::path::Path) -> bool {
    let index_dir = index_dir_for_workspace(workspace_root);
    let index_path = index_dir.join("index.bin");
    std::fs::metadata(index_path)
        .map(|m| m.len() > 0)
        .unwrap_or(false)
}

fn load_existing_index_status() -> IndexingStatus {
    // Note: this reads *existing* index metadata only; it does not start indexing.
    // The app will kick off background indexing from main.rs depending on config.
    let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if !index_file_exists(&workspace_root) {
        return IndexingStatus::NotStarted;
    }

    let cfg = SearchConfig::for_workspace(&workspace_root);
    match SemanticSearch::new(cfg) {
        Ok(search) => {
            let stats = search.stats();
            IndexingStatus::Complete {
                chunks: stats.total_chunks,
                files: stats.total_files,
            }
        }
        Err(_) => IndexingStatus::NotStarted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_runs_are_linked_by_call_id() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let tmp = std::env::temp_dir()
                .join(format!("lorikeet-test-{}", crate::memory::types::unix_ts()));
            let _ = std::fs::create_dir_all(&tmp);

            let (tx, _rx) = mpsc::unbounded_channel::<AppEvent>();
            let config = AppConfig::default();
            let policy = Arc::new(SandboxPolicy::from_config(
                config.clone(),
                tmp.clone(),
                crate::tools::TOOL_NAMES,
            ));
            let memory = Arc::new(MemoryManager::init(&tmp).await.unwrap());

            let mut app = App::new(tx, "k".into(), policy.clone(), config, tmp.clone(), memory);
            app.current_turn_id = 1;

            app.handle_event(AppEvent::ToolStart(crate::events::ToolStartEvent {
                call_id: "call-a".into(),
                tool: "bash".into(),
                args_raw: r#"{"command":"echo a"}"#.into(),
                args_summary: "echo a".into(),
                cwd: tmp.clone(),
                sandbox: crate::sandbox::SandboxDecision::allow(),
            }));
            app.handle_event(AppEvent::ToolStart(crate::events::ToolStartEvent {
                call_id: "call-b".into(),
                tool: "bash".into(),
                args_raw: r#"{"command":"echo b"}"#.into(),
                args_summary: "echo b".into(),
                cwd: tmp.clone(),
                sandbox: crate::sandbox::SandboxDecision::allow(),
            }));

            app.handle_event(AppEvent::ToolOutput(crate::events::ToolOutputEvent {
                call_id: "call-a".into(),
                chunk: "A
"
                .into(),
            }));

            assert_eq!(app.tool_outputs.len(), 2);
            assert!(app.tool_outputs[0].output.contains('A'));
            assert!(app.tool_outputs[1].output.is_empty());
        });
    }
}
