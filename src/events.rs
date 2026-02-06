use crate::types::ToolCallMessage;

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
    ToolStart(String, String),
    /// A chunk of streaming tool output to append.
    ToolOutput(String),
    ToolComplete(bool),
    // Indexing events
    IndexingStarted,
    #[allow(dead_code)]
    IndexingProgress(usize, usize), // (files_indexed, total_files)
    IndexingComplete(usize, usize), // (chunks, files)
    IndexingError(String),
}
