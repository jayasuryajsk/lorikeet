use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::prelude::Stylize;

use unicode_width::UnicodeWidthStr;

use crate::app::{App, Role};
use crate::markdown;
use crate::theme::{self, UiTheme};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum RenderedBlockId {
    Message(u64),
    ToolGroup(u64),
    Streaming,
    Spacer(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    Message,
    ToolGroup,
    Streaming,
    Spacer,
}

#[derive(Debug, Clone)]
struct BlockKey {
    width: u16,
    theme_key: u64,
    content_hash: u64,
}

#[derive(Debug, Clone)]
pub struct RenderedBlock {
    pub id: RenderedBlockId,
    pub kind: BlockKind,
    pub lines: Arc<Vec<Line<'static>>>,
    pub height: usize,
    key: BlockKey,
}

#[derive(Debug, Default)]
pub struct RenderStore {
    blocks: Vec<RenderedBlock>,
    index: HashMap<RenderedBlockId, usize>,
    dirty: HashSet<RenderedBlockId>,
    total_height: usize,
    last_width: u16,
    last_theme_key: u64,
}

impl RenderStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total_height(&self) -> usize {
        self.total_height
    }

    pub fn mark_all_dirty(&mut self) {
        self.dirty.clear();
        for b in &self.blocks {
            self.dirty.insert(b.id.clone());
        }
    }

    pub fn mark_dirty(&mut self, id: RenderedBlockId) {
        self.dirty.insert(id);
    }

    pub fn mark_tool_group_dirty(&mut self, group_id: u64) {
        self.dirty.insert(RenderedBlockId::ToolGroup(group_id));
    }

    pub fn ensure_up_to_date(&mut self, app: &App, ui_theme: &UiTheme, chat_width: usize) {
        let theme_key = hash64(&format!("{ui_theme:?}"));
        let width = chat_width.min(u16::MAX as usize) as u16;

        if self.last_width != width || self.last_theme_key != theme_key {
            self.last_width = width;
            self.last_theme_key = theme_key;
            self.dirty.clear();
            // Width/theme changes invalidate everything.
            for b in &self.blocks {
                self.dirty.insert(b.id.clone());
            }
        }

        // Desired block order.
        let mut desired: Vec<(RenderedBlockId, BlockKind)> = Vec::new();
        let display_messages: Vec<_> = app.display_messages().cloned().collect();
        for msg in &display_messages {
            desired.push((RenderedBlockId::Message(msg.id), BlockKind::Message));
            desired.push((RenderedBlockId::Spacer(msg.id), BlockKind::Spacer));
            if msg.role == Role::Agent {
                if let Some(group_id) = msg.tool_group_id {
                    desired.push((RenderedBlockId::ToolGroup(group_id), BlockKind::ToolGroup));
                }
            }
        }
        if app.is_processing {
            desired.push((RenderedBlockId::Streaming, BlockKind::Streaming));
        }

        // Always reconcile: streaming, spinners, and tool tails can change without structural changes.

        let mut old_blocks: HashMap<RenderedBlockId, RenderedBlock> = HashMap::new();
        for b in self.blocks.drain(..) {
            old_blocks.insert(b.id.clone(), b);
        }
        self.index.clear();
        self.total_height = 0;

        for (id, kind) in desired {
            let (content_hash, lines) = match (&id, kind) {
                (RenderedBlockId::Message(msg_id), BlockKind::Message) => match display_messages
                    .iter()
                    .find(|m| m.id == *msg_id)
                {
                    Some(msg) => {
                        let h = hash64(&format!(
                            "{:?}::{:?}::{:?}::{:?}",
                            msg.role, msg.content, msg.reasoning, msg.tool_group_id
                        ));
                        let lines = render_message_block(msg, ui_theme, chat_width);
                        (h, Arc::new(lines))
                    }
                    None => (0u64, Arc::new(Vec::new())),
                },
                (RenderedBlockId::ToolGroup(group_id), BlockKind::ToolGroup) => {
                    let h = tool_group_hash(app, *group_id, chat_width, theme_key);
                    let lines = render_tool_group_block(app, ui_theme, *group_id, chat_width);
                    (h, Arc::new(lines))
                }
                (RenderedBlockId::Streaming, BlockKind::Streaming) => {
                    let h = hash64(&format!(
                        "{}::{:?}::{:?}",
                        app.spinner_frame, app.current_reasoning, app.current_response
                    ));
                    let lines = render_streaming_block(app, ui_theme, chat_width);
                    (h, Arc::new(lines))
                }
                (RenderedBlockId::Spacer(_), BlockKind::Spacer) => {
                    // Always 1 empty line.
                    (0u64, Arc::new(vec![Line::from("")]))
                }
                _ => (0u64, Arc::new(Vec::new())),
            };

            let key = BlockKey {
                width,
                theme_key,
                content_hash,
            };

            let reuse = old_blocks.get(&id).and_then(|old| {
                if !self.dirty.contains(&id) && old.key.width == width && old.key.theme_key == theme_key
                    && old.key.content_hash == content_hash
                {
                    Some(old.lines.clone())
                } else {
                    None
                }
            });

            let final_lines = reuse.unwrap_or(lines);
            let height = final_lines.len();
            let block = RenderedBlock {
                id: id.clone(),
                kind,
                lines: final_lines,
                height,
                key,
            };
            self.index.insert(id, self.blocks.len());
            self.total_height += height;
            self.blocks.push(block);
        }

        // Rebuild done; clear dirt for ids that no longer exist as well.
        self.dirty.retain(|id| self.index.contains_key(id));
    }

    pub fn visible_lines(&self, scroll_rows: usize, viewport_height: usize) -> Vec<Line<'static>> {
        if viewport_height == 0 || self.total_height == 0 {
            return Vec::new();
        }

        let start = scroll_rows.min(self.total_height);
        let end = (start + viewport_height).min(self.total_height);

        let mut out: Vec<Line<'static>> = Vec::with_capacity(end.saturating_sub(start));
        let mut row_cursor: usize = 0;

        for b in &self.blocks {
            let block_start = row_cursor;
            let block_end = row_cursor + b.height;
            row_cursor = block_end;

            if block_end <= start {
                continue;
            }
            if block_start >= end {
                break;
            }

            let slice_start = start.saturating_sub(block_start);
            let slice_end = (end - block_start).min(b.height);
            for line in b.lines[slice_start..slice_end].iter().cloned() {
                out.push(line);
            }
        }

        out
    }
}

fn hash64(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn render_message_block(msg: &crate::app::Message, ui_theme: &UiTheme, chat_width: usize) -> Vec<Line<'static>> {
    let pal = ui_theme.palette;
    let (prefix, md_theme, prefix_style) = match msg.role {
        Role::User => (
            "▶ ",
            theme::user_markdown_theme(ui_theme),
            Style::default().fg(Color::Cyan).bold(),
        ),
        Role::Agent => (
            "● ",
            ui_theme.markdown,
            Style::default().fg(pal.accent).bold(),
        ),
        Role::System => (
            "◆ ",
            ui_theme.markdown,
            Style::default().fg(pal.warn).bold(),
        ),
        Role::Tool => (
            "⚙ ",
            ui_theme.markdown,
            Style::default().fg(pal.accent).bold(),
        ),
    };

    let mut lines: Vec<Line<'static>> = Vec::new();

    if let Some(reasoning) = &msg.reasoning {
        for line in wrap_lines(reasoning, chat_width.saturating_sub(2)) {
            if line.trim().is_empty() {
                continue;
            }
            lines.push(Line::from(Span::styled(line, pal.ghost())));
        }
    }

    let md_lines = markdown::render(
        &msg.content,
        md_theme,
        ui_theme.syntax,
        chat_width.saturating_sub(2),
    );

    let mut prev_empty = false;
    let mut is_first = true;
    for line in md_lines.into_iter() {
        let line_text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        let is_empty = line_text.trim().is_empty();
        if is_empty {
            if prev_empty {
                continue;
            }
            prev_empty = true;
        } else {
            prev_empty = false;
        }

        if is_first {
            is_first = false;
            let mut spans: Vec<Span<'static>> =
                vec![Span::styled(prefix.to_string(), prefix_style)];
            spans.extend(line.spans);
            lines.push(Line::from(spans));
        } else {
            let mut spans: Vec<Span<'static>> = vec![Span::raw("  ")];
            spans.extend(line.spans);
            lines.push(Line::from(spans));
        }
    }

    // Trim trailing empty line and keep one spacer via Spacer blocks.
    while let Some(last) = lines.last() {
        let last_text: String = last.spans.iter().map(|s| s.content.as_ref()).collect();
        if last_text.trim().is_empty() {
            lines.pop();
        } else {
            break;
        }
    }

    lines
}

fn tool_group_hash(app: &App, group_id: u64, chat_width: usize, theme_key: u64) -> u64 {
    let expanded = app.tool_trace_expanded.get(&group_id).copied().unwrap_or(false);
    let details = app.tool_trace_show_details.get(&group_id).copied().unwrap_or(true);
    let tools: Vec<_> = app.tool_outputs.iter().filter(|t| t.group_id == group_id).collect();
    let mut s = format!("{group_id}::{expanded}::{details}::{chat_width}::{theme_key}");
    for t in tools {
        let tail_hash = output_tail_hash(t, expanded);
        s.push_str(&format!(
            "::{}:{}:{}:{:?}:{tail_hash}",
            t.call_id, t.tool, t.args_summary, t.status
        ));
    }
    hash64(&s)
}

fn render_tool_group_block(app: &App, ui_theme: &UiTheme, group_id: u64, chat_width: usize) -> Vec<Line<'static>> {
    // Reuse the existing inline renderer (it builds Lines) by capturing into a vec.
    let pal = ui_theme.palette;
    let mut out: Vec<Line<'static>> = Vec::new();
    let tool_spinner_frames: &[&str] = &["◐", "◓", "◑", "◒"];
    let tool_spinner = tool_spinner_frames[app.tool_spinner_frame % tool_spinner_frames.len()];

    // Header
    let tools: Vec<&crate::app::ToolOutput> = app
        .tool_outputs
        .iter()
        .filter(|t| t.group_id == group_id)
        .collect();
    if tools.is_empty() {
        return out;
    }
    let any_running = tools.iter().any(|t| matches!(t.status, crate::app::ToolStatus::Running));
    let expanded = app.tool_trace_expanded.get(&group_id).copied().unwrap_or(any_running);
    let glyph = if expanded { "▾" } else { "▸" };
    let status = if any_running { "running…" } else { "done" };
    let status_glyph = if any_running { tool_spinner } else { "●" };
    let status_style = if any_running {
        Style::default().fg(pal.warn)
    } else {
        Style::default().fg(pal.ok)
    };

    out.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            format!("{glyph} Tool Trace ({}) ", tools.len()),
            pal.meta(),
        ),
        Span::styled(format!("{status_glyph} {status}"), status_style),
    ]));

    for t in tools {
        render_tool_trace_item(app, ui_theme, t, expanded, tool_spinner, chat_width, &mut out);
    }

    out.push(Line::from(""));
    out
}

fn render_streaming_block(app: &App, ui_theme: &UiTheme, chat_width: usize) -> Vec<Line<'static>> {
    let pal = ui_theme.palette;
    let spinner_frames: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let spinner = spinner_frames[app.spinner_frame % spinner_frames.len()];
    let mut out: Vec<Line<'static>> = Vec::new();

    if !app.current_reasoning.is_empty() {
        for line in wrap_lines(&app.current_reasoning, chat_width.saturating_sub(2)) {
            if line.trim().is_empty() {
                continue;
            }
            out.push(Line::from(Span::styled(line, pal.ghost())));
        }
    }

    if app.current_response.is_empty() && app.current_reasoning.is_empty() {
        out.push(Line::from(vec![
            Span::styled(format!("{spinner} "), Style::default().fg(pal.warn)),
            Span::styled("…", pal.meta()),
        ]));
        return out;
    }

    if !app.current_response.is_empty() {
        let md_lines = markdown::render(
            &app.current_response,
            ui_theme.markdown,
            ui_theme.syntax,
            chat_width.saturating_sub(4),
        );
        if let Some(first) = md_lines.first() {
            let mut spans: Vec<Span<'static>> = vec![Span::styled(
                format!("{spinner} "),
                Style::default().fg(pal.warn),
            )];
            spans.extend(first.spans.iter().cloned());
            out.push(Line::from(spans));
        }
        for line in md_lines.into_iter().skip(1) {
            let mut spans: Vec<Span<'static>> = vec![Span::raw("  ")];
            spans.extend(line.spans);
            out.push(Line::from(spans));
        }
    } else {
        out.push(Line::from(Span::styled(
            format!("{spinner} "),
            Style::default().fg(pal.warn),
        )));
    }

    out
}

fn output_tail_hash(t: &crate::app::ToolOutput, expanded: bool) -> u64 {
    let k = if matches!(t.status, crate::app::ToolStatus::Running) {
        8usize
    } else if expanded {
        20usize
    } else {
        2usize
    };
    let mut s = String::new();
    for l in t.output_lines.iter().rev().take(k).rev() {
        s.push_str(l);
        s.push('\n');
    }
    hash64(&s)
}

fn render_tool_trace_item(
    app: &App,
    ui_theme: &UiTheme,
    tool: &crate::app::ToolOutput,
    group_expanded: bool,
    tool_spinner: &str,
    chat_width: usize,
    out: &mut Vec<Line<'static>>,
) {
    let pal = ui_theme.palette;
    let show_details = app
        .tool_trace_show_details
        .get(&tool.group_id)
        .copied()
        .unwrap_or(true);
    let (status_indicator, status_color) = match tool.status {
        crate::app::ToolStatus::Running => (tool_spinner, pal.warn),
        crate::app::ToolStatus::Success => ("●", pal.ok),
        crate::app::ToolStatus::Error => ("●", pal.err),
    };

    let elapsed = tool.elapsed();
    let elapsed_str = if elapsed.as_secs() >= 60 {
        format!(
            "{}m{:.1}s",
            elapsed.as_secs() / 60,
            (elapsed.as_secs() % 60) as f64 + elapsed.subsec_millis() as f64 / 1000.0
        )
    } else if elapsed.as_secs() >= 1 {
        format!("{:.1}s", elapsed.as_secs_f64())
    } else {
        format!("{}ms", elapsed.as_millis())
    };

    let cwd_display = tool
        .cwd
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(".");

    let suffix = format!(
        " [id={}] (cwd={}) ({})",
        tool.call_id, cwd_display, elapsed_str
    );
    let prefix = format!("  {} {}  ", status_indicator, tool.tool);

    let prefix_w = UnicodeWidthStr::width(prefix.as_str());
    let suffix_w = UnicodeWidthStr::width(suffix.as_str());
    let available = chat_width.saturating_sub(prefix_w + suffix_w);
    let args = truncate_to_width(&tool.args_summary, available);

    out.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("{} ", status_indicator), Style::default().fg(status_color)),
        Span::styled(
            format!("{} ", tool.tool),
            Style::default()
                .fg(match tool.status {
                    crate::app::ToolStatus::Running => pal.warn,
                    crate::app::ToolStatus::Success => ui_theme.tool_trace.invocation,
                    crate::app::ToolStatus::Error => pal.err,
                })
                .bold(),
        ),
        Span::styled(args, Style::default().fg(ui_theme.tool_trace.invocation)),
        Span::styled(
            suffix,
            Style::default()
                .fg(ui_theme.tool_trace.call_id)
                .add_modifier(Modifier::DIM),
        ),
    ]));

    let want_details = (group_expanded && show_details) || tool.status == crate::app::ToolStatus::Error;
    if want_details {
        let (label, style) = if tool.sandbox.allowed {
            ("allow".to_string(), Style::default().fg(ui_theme.tool_trace.sandbox_allow))
        } else {
            (
                format!(
                    "deny{}",
                    tool.sandbox
                        .reason
                        .as_ref()
                        .map(|r| format!(" ({})", truncate_to_width(r, chat_width.saturating_sub(14))))
                        .unwrap_or_default()
                ),
                Style::default().fg(ui_theme.tool_trace.sandbox_deny),
            )
        };

        out.push(Line::from(vec![
            Span::styled("  └ sandbox: ", Style::default().fg(ui_theme.tool_trace.details_key)),
            Span::styled(label, style),
        ]));

        const MAX_INPUT_LINES: usize = 8;
        if !tool.args_pretty_lines.is_empty() {
            out.push(Line::from(vec![
                Span::styled("  └ input: ", Style::default().fg(ui_theme.tool_trace.details_key)),
                Span::styled(
                    "{",
                    Style::default()
                        .fg(ui_theme.tool_trace.details_value)
                        .add_modifier(Modifier::DIM),
                ),
            ]));
            for l in tool.args_pretty_lines.iter().take(MAX_INPUT_LINES) {
                out.push(Line::from(vec![
                    Span::raw("    "),
                    Span::styled(
                        truncate_to_width(l, chat_width.saturating_sub(4)),
                        Style::default()
                            .fg(ui_theme.tool_trace.details_value)
                            .add_modifier(Modifier::DIM),
                    ),
                ]));
            }
            if tool.args_pretty_lines.len() > MAX_INPUT_LINES {
                out.push(Line::from(vec![
                    Span::raw("    "),
                    Span::styled(
                        format!("… {} more lines", tool.args_pretty_lines.len() - MAX_INPUT_LINES),
                        pal.meta(),
                    ),
                ]));
            }
            out.push(Line::from(vec![
                Span::raw("    "),
                Span::styled(
                    "}",
                    Style::default()
                        .fg(ui_theme.tool_trace.details_value)
                        .add_modifier(Modifier::DIM),
                ),
            ]));
        }
    }

    render_tool_output_lines(ui_theme, tool, group_expanded, chat_width, out);
}

fn render_tool_output_lines(
    ui_theme: &UiTheme,
    tool: &crate::app::ToolOutput,
    group_expanded: bool,
    chat_width: usize,
    out: &mut Vec<Line<'static>>,
) {
    let pal = ui_theme.palette;
    let k = if matches!(tool.status, crate::app::ToolStatus::Running) {
        8usize
    } else if group_expanded {
        20usize
    } else {
        2usize
    };

    let mut lines: Vec<String> = tool.output_lines.iter().cloned().collect();
    if lines.is_empty() && !tool.output.is_empty() {
        lines = tool.output.lines().map(|s| s.to_string()).collect();
    }
    let total = lines.len();
    let start = total.saturating_sub(k);
    let shown = &lines[start..];

    if !shown.is_empty() {
        out.push(Line::from(vec![
            Span::styled("  └ out: ", Style::default().fg(ui_theme.tool_trace.details_key)),
            Span::styled(
                truncate_to_width(&shown[0], chat_width.saturating_sub(10)),
                Style::default().fg(ui_theme.tool_trace.out_text),
            ),
        ]));
        for l in shown.iter().skip(1) {
            out.push(Line::from(vec![
                Span::styled("      ", Style::default().fg(ui_theme.tool_trace.out_prefix)),
                Span::styled(
                    truncate_to_width(l, chat_width.saturating_sub(6)),
                    Style::default().fg(ui_theme.tool_trace.out_text),
                ),
            ]));
        }
        if total > k {
            out.push(Line::from(vec![
                Span::raw("      "),
                Span::styled(format!("… {} more lines", total - k), pal.meta()),
            ]));
        }
    }
}

fn truncate_to_width(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if UnicodeWidthStr::width(s) <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let mut w = 0usize;
    for ch in s.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1);
        if w + cw + 1 > max {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out.push('…');
    out
}

fn wrap_lines(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![];
    }
    let mut out = Vec::new();
    for line in text.lines() {
        if line.chars().count() <= width {
            out.push(line.to_string());
            continue;
        }
        let chars: Vec<char> = line.chars().collect();
        for chunk in chars.chunks(width) {
            out.push(chunk.iter().collect::<String>());
        }
    }
    out
}
