use std::path::PathBuf;

use crate::sandbox::SandboxDecision;
use crate::types::ToolCallMessage;

#[derive(Debug, Clone)]
pub struct ToolStartEvent {
    pub call_id: String,
    pub tool: String,
    pub args_raw: String,
    pub args_summary: String,
    pub cwd: PathBuf,
    pub sandbox: SandboxDecision,
}

#[derive(Debug, Clone)]
pub struct ToolOutputEvent {
    pub call_id: String,
    pub chunk: String,
}

#[derive(Debug, Clone)]
pub struct ToolCompleteEvent {
    pub call_id: String,
    pub success: bool,
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    Input(crossterm::event::KeyEvent),
    Mouse(crossterm::event::MouseEvent),
    AgentChunk(String),
    AgentReasoning(String),
    AgentDone,
    AgentToolCalls(Vec<ToolCallMessage>),
    ToolResultsReady(Vec<(String, String)>), // (tool_call_id, result)
    AgentError(String),

    ToolStart(ToolStartEvent),
    /// A chunk of streaming tool output to append.
    ToolOutput(ToolOutputEvent),
    ToolComplete(ToolCompleteEvent),

    // Indexing events
    IndexingStarted,
    #[allow(dead_code)]
    IndexingProgress(usize, usize), // (files_indexed, total_files)
    IndexingComplete(usize, usize), // (chunks, files)
    IndexingError(String),
}
