use ratatui::{
    layout::Margin,
    prelude::*,
    widgets::{
        Block, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
    },
};

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::app::{App, IndexingStatus, Pane, Role, ToolOutput, ToolStatus};
use crate::markdown;
use crate::theme;

const INDEXING_SPINNER: &[&str] = &["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const TOOL_SPINNER_FRAMES: &[&str] = &["◐", "◓", "◑", "◒"];

// Styling helpers that work on both dark and light terminal themes.
fn ghost_style() -> Style {
    Style::default()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::ITALIC)
        .add_modifier(Modifier::DIM)
}

fn meta_style() -> Style {
    Style::default()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::DIM)
}

pub fn ui(frame: &mut Frame, app: &mut App) {
    app.root_area = frame.area();
    let split_ratio = app.split_ratio.max(20).min(80);

    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(split_ratio),
            Constraint::Length(1),
            Constraint::Percentage(100 - split_ratio),
        ])
        .split(frame.area());

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(main_chunks[0]);

    let splitter_style = Style::default().fg(Color::DarkGray);
    app.splitter_area = main_chunks[1];
    let splitter = Paragraph::new("│".repeat(main_chunks[1].height as usize)).style(splitter_style);
    frame.render_widget(splitter, main_chunks[1]);

    let chat_border_style = Style::default().fg(Color::DarkGray);

    let chat_width = left_chunks[0].width.saturating_sub(4) as usize;

    // Advance tool spinner (used by inline tool rendering)
    app.tool_spinner_frame = (app.tool_spinner_frame + 1) % TOOL_SPINNER_FRAMES.len();
    let tool_spinner = TOOL_SPINNER_FRAMES[app.tool_spinner_frame];

    // Advance indexing spinner (used by context sidebar)
    app.indexing_spinner_frame = (app.indexing_spinner_frame + 1) % INDEXING_SPINNER.len();

    // Render messages with markdown + inline tools
    let mut all_lines: Vec<Line> = Vec::new();

    // Copy the messages so we can mutably borrow `app` while rendering tools.
    let display_messages: Vec<_> = app.display_messages().cloned().collect();
    for msg in display_messages {
        let (prefix, theme) = match msg.role {
            Role::User => ("▶ ", markdown::Theme::user()),
            Role::Agent => ("● ", markdown::Theme::agent()),
            Role::System => ("◆ ", markdown::Theme::agent()),
            Role::Tool => ("⚙ ", markdown::Theme::agent()),
        };

        // Add prefix line
        let prefix_style = match msg.role {
            Role::User => Style::default().fg(Color::Cyan).bold(),
            Role::Agent => Style::default().fg(Color::Rgb(217, 119, 87)).bold(),
            Role::System => Style::default().fg(Color::Yellow).bold(),
            Role::Tool => Style::default().fg(Color::Magenta).bold(),
        };

        // Show reasoning as ghost text (if any) - wrap long lines
        if let Some(reasoning) = &msg.reasoning {
            for line in reasoning.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                // Wrap long lines instead of truncating
                let chars: Vec<char> = line.chars().collect();
                let wrap_width = chat_width.saturating_sub(2);
                for chunk in chars.chunks(wrap_width) {
                    let text: String = chunk.iter().collect();
                    all_lines.push(Line::from(Span::styled(text, ghost_style())));
                }
            }
        }

        let md_lines = markdown::render(&msg.content, theme, chat_width.saturating_sub(2));

        // Filter out excessive empty lines
        let mut prev_empty = false;
        let mut is_first = true;
        for line in md_lines.into_iter() {
            // Check if line is effectively empty
            let line_text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
            let is_empty = line_text.trim().is_empty();

            // Skip consecutive empty lines
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
                let mut first_spans = vec![Span::styled(prefix.to_string(), prefix_style)];
                first_spans.extend(line.spans);
                all_lines.push(Line::from(first_spans));
            } else {
                let mut indented = vec![Span::raw("  ")];
                indented.extend(line.spans);
                all_lines.push(Line::from(indented));
            }
        }

        // Remove trailing empty line if present, then add single spacing
        if let Some(last) = all_lines.last() {
            let last_text: String = last.spans.iter().map(|s| s.content.as_ref()).collect();
            if last_text.trim().is_empty() {
                all_lines.pop();
            }
        }
        all_lines.push(Line::from("")); // single line spacing between messages

        // Tool trace belongs to the assistant tool-call phase.
        // Render it immediately after the assistant message that initiated the tool group.
        if msg.role == Role::Agent {
            if let Some(group_id) = msg.tool_group_id {
                render_tool_group_inline(&*app, group_id, tool_spinner, chat_width, &mut all_lines);
            }
        }
    }

    // Show streaming response or loading spinner
    if app.is_processing {
        // Advance spinner
        app.spinner_frame = (app.spinner_frame + 1) % SPINNER_FRAMES.len();
        let spinner = SPINNER_FRAMES[app.spinner_frame];

        // Show reasoning tokens as ghost text (if any) - wrap long lines
        if !app.current_reasoning.is_empty() {
            for line in app.current_reasoning.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let chars: Vec<char> = line.chars().collect();
                let wrap_width = chat_width.saturating_sub(2);
                for chunk in chars.chunks(wrap_width) {
                    let text: String = chunk.iter().collect();
                    all_lines.push(Line::from(Span::styled(text, ghost_style())));
                }
            }
        }

        if app.current_response.is_empty() && app.current_reasoning.is_empty() {
            // Show spinner with elapsed time
            let elapsed = app
                .processing_start
                .map(|t| t.elapsed())
                .unwrap_or_default();
            let time_str = if elapsed.as_secs() >= 60 {
                format!("{}m {}s", elapsed.as_secs() / 60, elapsed.as_secs() % 60)
            } else if elapsed.as_secs() > 0 {
                format!(
                    "{}.{}s",
                    elapsed.as_secs(),
                    elapsed.as_millis() % 1000 / 100
                )
            } else {
                format!("{}ms", elapsed.as_millis())
            };
            all_lines.push(Line::from(vec![
                Span::styled(format!("{} ", spinner), Style::default().fg(Color::Yellow)),
                Span::styled(time_str, Style::default().fg(Color::DarkGray)),
            ]));
        } else if !app.current_response.is_empty() {
            // Show streaming content with spinner
            let theme = markdown::Theme::agent();
            let md_lines =
                markdown::render(&app.current_response, theme, chat_width.saturating_sub(4));

            if let Some(first) = md_lines.first() {
                let mut first_spans = vec![Span::styled(
                    format!("{} ", spinner),
                    Style::default().fg(Color::Yellow),
                )];
                first_spans.extend(first.spans.iter().cloned());
                all_lines.push(Line::from(first_spans));
            }

            for line in md_lines.into_iter().skip(1) {
                let mut indented = vec![Span::raw("  ")];
                indented.extend(line.spans);
                all_lines.push(Line::from(indented));
            }
        } else if !app.current_reasoning.is_empty() {
            // Has reasoning but no response yet - show spinner
            all_lines.push(Line::from(vec![Span::styled(
                format!("{} ", spinner),
                Style::default().fg(Color::Yellow),
            )]));
        }
    }

    // Calculate scroll
    let total_lines = all_lines.len();
    let visible_height = left_chunks[0].height.saturating_sub(2) as usize; // subtract borders
    let max_scroll = total_lines.saturating_sub(visible_height);

    // Auto-scroll to bottom if enabled
    if app.messages_auto_scroll {
        app.messages_scroll = max_scroll;
    } else {
        // Clamp scroll to valid range
        app.messages_scroll = app.messages_scroll.min(max_scroll);
    }

    // Re-enable auto-scroll if we're at the bottom
    if app.messages_scroll >= max_scroll.saturating_sub(1) {
        app.messages_auto_scroll = true;
    }

    // Store chat area for mouse handling
    app.chat_area = left_chunks[0];

    let chat_title = if app.active_pane == Pane::Chat {
        format!("* Lorikeet · {} ", app.model)
    } else {
        format!("  Lorikeet · {} ", app.model)
    };
    let messages_widget = Paragraph::new(all_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(chat_border_style)
                .title(chat_title),
        )
        .scroll((app.messages_scroll as u16, 0));
    frame.render_widget(messages_widget, left_chunks[0]);

    // Chat scrollbar - use max_scroll as content length so thumb reaches bottom
    let scrollbar_content = if max_scroll > 0 { max_scroll } else { 1 };
    let mut chat_scrollbar_state =
        ScrollbarState::new(scrollbar_content).position(app.messages_scroll);
    let chat_scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .begin_symbol(None)
        .end_symbol(None)
        .track_symbol(Some("░"))
        .thumb_symbol("█");
    frame.render_stateful_widget(
        chat_scrollbar,
        left_chunks[0].inner(Margin {
            vertical: 1,
            horizontal: 0,
        }),
        &mut chat_scrollbar_state,
    );

    // Input
    let input_title = " Input ";

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(chat_border_style)
        .title(input_title);

    let input_widget = Paragraph::new(app.input.as_str())
        .block(input_block)
        .wrap(Wrap { trim: false });
    frame.render_widget(input_widget, left_chunks[1]);

    if app.active_pane == Pane::Chat && !app.is_processing && !app.settings_open {
        let cursor_x = left_chunks[1].x + app.cursor_pos as u16 + 1;
        let cursor_y = left_chunks[1].y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    // Status bar
    let mut status_text =
        " ESC quit │ TAB switch │ CTRL+←/→ resize │ SHIFT+↑↓ scroll │ PgUp/PgDn │ ENTER send"
            .to_string();
    if let Some(group_id) = app.last_tool_group_id {
        let has_trace = app.tool_outputs.iter().any(|t| t.group_id == group_id);
        if has_trace {
            status_text.push_str(" │ Ctrl+E trace │ Ctrl+I details");
        }
    }
    if !app.verify_suggestions.is_empty() {
        let hint = app
            .verify_suggestions
            .iter()
            .take(1)
            .map(|s| format!("Verify: {}", s.command))
            .collect::<Vec<_>>()
            .join(" ");
        status_text.push_str(" │ ");
        status_text.push_str(&hint);
        status_text.push_str(" (/verify)");
    }

    let status = Paragraph::new(status_text).style(meta_style());
    frame.render_widget(status, left_chunks[2]);

    // Context sidebar (right pane)
    app.context_area = main_chunks[2];
    render_context_sidebar(frame, app, main_chunks[2]);
    if app.settings_open {
        render_settings_popup(frame, app);
    }
}

/// Render the indexing status bar
fn render_indexing_status(status: &IndexingStatus, spinner_frame: usize) -> String {
    let spinner = INDEXING_SPINNER[spinner_frame % INDEXING_SPINNER.len()];

    match status {
        IndexingStatus::NotStarted => String::new(),
        IndexingStatus::Indexing {
            files_done,
            total_files,
        } => {
            if *total_files > 0 {
                format!(
                    " {} Indexing... {}/{} files",
                    spinner, files_done, total_files
                )
            } else {
                format!(" {} Indexing...", spinner)
            }
        }
        IndexingStatus::Complete { chunks, files } => {
            format!(" ● Indexed {} chunks from {} files", chunks, files)
        }
        IndexingStatus::Error(err) => {
            let msg = if err.len() > 40 {
                format!(" ● Index error: {}...", &err[..40])
            } else {
                format!(" ● Index error: {}", err)
            };
            msg
        }
    }
}

fn render_settings_popup(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let popup_area = centered_rect(70, 70, area);

    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Gray))
        .title(" Settings ");
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(inner);

    let rows = app.settings_rows();
    let width = chunks[0].width.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::new();
    for (idx, (label, value)) in rows.iter().enumerate() {
        let is_selected = idx == app.settings_selected;
        let prefix = if is_selected { "> " } else { "  " };
        let display = format!("{:<20} {}", label, value);
        let trimmed = truncate_line(&display, width.saturating_sub(2));
        let style = if is_selected {
            Style::default().fg(Color::Black).bg(Color::Gray)
        } else {
            Style::default().fg(Color::Reset)
        };
        lines.push(Line::from(Span::styled(
            format!("{}{}", prefix, trimmed),
            style,
        )));
    }

    let list = Paragraph::new(lines).wrap(Wrap { trim: true });
    frame.render_widget(list, chunks[0]);

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Value ");
    let input = Paragraph::new(app.settings_input.as_str())
        .block(input_block)
        .wrap(Wrap { trim: false });
    frame.render_widget(input, chunks[1]);

    let hints = Paragraph::new(" Enter save │ Esc cancel │ Up/Down navigate │ Tab next ")
        .style(Style::default().fg(Color::DarkGray));
    frame.render_widget(hints, chunks[2]);

    let cursor_x = chunks[1].x + 1 + app.settings_cursor as u16;
    let cursor_y = chunks[1].y + 1;
    frame.set_cursor_position((cursor_x, cursor_y));
}

fn render_tool_group_inline(
    app: &App,
    group_id: u64,
    tool_spinner: &str,
    chat_width: usize,
    out: &mut Vec<Line<'static>>,
) {
    let tools: Vec<&ToolOutput> = app
        .tool_outputs
        .iter()
        .filter(|t| t.group_id == group_id)
        .collect();
    if tools.is_empty() {
        return;
    }

    let any_running = tools.iter().any(|t| t.status == ToolStatus::Running);
    let expanded = app
        .tool_trace_expanded
        .get(&group_id)
        .copied()
        .unwrap_or(false);
    let show_details = app
        .tool_trace_show_details
        .get(&group_id)
        .copied()
        .unwrap_or(true);

    let disclosure = if expanded { "▾" } else { "▸" };
    let status = if any_running {
        format!("{} running…", tool_spinner)
    } else {
        "● done".to_string()
    };

    out.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            format!("{} Tool Trace ({}) ", disclosure, tools.len()),
            Style::default().fg(Color::Reset).bold(),
        ),
        Span::styled(
            status,
            Style::default().fg(if any_running {
                Color::Yellow
            } else {
                Color::Green
            }),
        ),
    ]));

    for t in tools {
        render_tool_trace_item(
            app,
            t,
            expanded,
            show_details,
            tool_spinner,
            chat_width,
            out,
        );
    }

    out.push(Line::from(""));
}

fn render_tool_trace_item(
    app: &App,
    tool: &ToolOutput,
    group_expanded: bool,
    show_details: bool,
    tool_spinner: &str,
    chat_width: usize,
    out: &mut Vec<Line<'static>>,
) {
    let (status_indicator, status_color) = match tool.status {
        ToolStatus::Running => (tool_spinner, Color::Yellow),
        ToolStatus::Success => ("●", Color::Green),
        ToolStatus::Error => ("●", Color::Red),
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

    let cwd_display = tool.cwd.file_name().and_then(|s| s.to_str()).unwrap_or(".");

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
        Span::styled(
            format!("{} ", status_indicator),
            Style::default().fg(status_color),
        ),
        Span::styled(
            format!("{} ", tool.tool),
            Style::default()
                .fg(match tool.status {
                    ToolStatus::Running => Color::Yellow,
                    ToolStatus::Success => Color::Reset,
                    ToolStatus::Error => Color::Red,
                })
                .bold(),
        ),
        Span::styled(args, Style::default().fg(Color::Reset)),
        Span::styled(suffix, meta_style()),
    ]));

    let want_details = (group_expanded && show_details) || tool.status == ToolStatus::Error;
    if want_details {
        let (label, style) = if tool.sandbox.allowed {
            ("allow".to_string(), Style::default().fg(Color::Green))
        } else {
            (
                format!(
                    "deny{}",
                    tool.sandbox
                        .reason
                        .as_ref()
                        .map(|r| format!(
                            " ({})",
                            truncate_to_width(r, chat_width.saturating_sub(14))
                        ))
                        .unwrap_or_default()
                ),
                Style::default().fg(Color::Red),
            )
        };

        out.push(Line::from(vec![
            Span::raw("  └ sandbox: "),
            Span::styled(label, style),
        ]));

        // Input JSON (redacted + pretty-printed at capture time)
        const MAX_INPUT_LINES: usize = 8;
        if !tool.args_pretty_lines.is_empty() {
            out.push(Line::from(vec![
                Span::raw("  └ input: "),
                Span::styled("{", meta_style()),
            ]));
            for l in tool.args_pretty_lines.iter().take(MAX_INPUT_LINES) {
                out.push(Line::from(vec![
                    Span::raw("    "),
                    Span::styled(
                        truncate_to_width(l, chat_width.saturating_sub(4)),
                        meta_style(),
                    ),
                ]));
            }
            if tool.args_pretty_lines.len() > MAX_INPUT_LINES {
                out.push(Line::from(vec![
                    Span::raw("    "),
                    Span::styled(
                        format!(
                            "… {} more lines",
                            tool.args_pretty_lines.len() - MAX_INPUT_LINES
                        ),
                        meta_style(),
                    ),
                ]));
            }
        }
    }

    let has_output = !tool.output.is_empty() || !tool.output_lines.is_empty();

    // Collapsed-by-default: show only a single live tail line while running.
    if !group_expanded {
        if tool.status == ToolStatus::Running && has_output {
            let (tail, _remaining) = tool.tail_lines(1);
            if let Some(last) = tail.last() {
                let last = last.trim();
                if !last.is_empty() {
                    out.push(Line::from(vec![
                        Span::raw("    "),
                        Span::styled(
                            format!(
                                "… {}",
                                truncate_to_width(last, chat_width.saturating_sub(6))
                            ),
                            meta_style(),
                        ),
                    ]));
                }
            }
        }
        return;
    }

    if !has_output {
        return;
    }

    let k = if tool.status == ToolStatus::Running {
        8
    } else {
        20
    };

    let (tail, remaining) = tool.tail_lines(k);
    if tail.is_empty() {
        return;
    }

    let ext = if tool.tool == "read_file" {
        std::path::Path::new(&tool.target)
            .extension()
            .and_then(|s| s.to_str())
    } else {
        None
    };

    for (i, l) in tail.iter().enumerate() {
        let is_first = i == 0;
        let prefix = if is_first { "  └ out: " } else { "        " };

        let mut spans: Vec<Span<'static>> = Vec::new();
        spans.push(Span::styled(
            prefix,
            Style::default().fg(Color::Rgb(120, 120, 120)),
        ));

        let text = truncate_to_width(l, chat_width.saturating_sub(prefix.len()));
        if tool.tool == "list_files" {
            let style = theme::style_for_filename(&text, &app.config);
            spans.push(Span::styled(text, style));
        } else if tool.tool == "read_file" {
            spans.extend(highlight_line_for_ext(&text, ext));
        } else {
            let base_style = if tool.status == ToolStatus::Error {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::Rgb(200, 200, 200))
            };
            spans.push(Span::styled(text, base_style));
        }

        out.push(Line::from(spans));
    }

    if remaining > 0 || tool.output_is_truncated() {
        out.push(Line::from(vec![
            Span::raw("        "),
            Span::styled(format!("… {} more lines", remaining), meta_style()),
        ]));
    }
}

fn render_context_sidebar(frame: &mut Frame, app: &mut App, area: Rect) {
    let title = if app.active_pane == Pane::Context {
        " * Context "
    } else {
        "   Context "
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(title)
        .title_bottom(render_indexing_status(
            &app.indexing_status,
            app.indexing_spinner_frame,
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let width = inner.width.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::new();

    lines.push(Line::from(vec![
        Span::styled("Workspace: ", meta_style()),
        Span::raw(truncate_to_width(
            &app.workspace_root_display(),
            width.saturating_sub(11),
        )),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Model: ", meta_style()),
        Span::raw(truncate_to_width(&app.model, width.saturating_sub(7))),
    ]));

    let sandbox_enabled = app
        .config
        .sandbox
        .as_ref()
        .and_then(|s| s.enabled)
        .unwrap_or(true);
    lines.push(Line::from(vec![
        Span::styled("Sandbox: ", meta_style()),
        Span::raw(if sandbox_enabled { "on" } else { "off" }),
    ]));

    if !app.verify_suggestions.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("Verify: ", meta_style()),
            Span::raw(truncate_to_width(
                &app.verify_suggestions[0].command,
                width.saturating_sub(8),
            )),
        ]));
        lines.push(Line::from(Span::styled("Run: /verify", meta_style())));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Recent files",
        Style::default().fg(Color::Reset).bold(),
    )));
    if app.recent_files.is_empty() {
        lines.push(Line::from(Span::styled("(none yet)", meta_style())));
    } else {
        for p in app.recent_files.iter().take(10) {
            lines.push(Line::from(vec![
                Span::raw("- "),
                Span::raw(truncate_to_width(p, width.saturating_sub(2))),
            ]));
        }
    }

    let widget = Paragraph::new(lines).wrap(Wrap { trim: true });
    frame.render_widget(widget, inner);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1]);

    horizontal[1]
}

fn truncate_line(line: &str, width: usize) -> String {
    if UnicodeWidthStr::width(line) <= width {
        return line.to_string();
    }
    if width <= 1 {
        return "…".to_string();
    }

    let mut out = String::new();
    let mut used = 0;
    for ch in line.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + w + 1 > width {
            break;
        }
        out.push(ch);
        used = used.saturating_add(w);
    }
    out.push('…');
    out
}

fn highlight_line_for_ext(line: &str, ext: Option<&str>) -> Vec<Span<'static>> {
    match ext {
        Some("md") | Some("mdx") => highlight_markdown_line(line),
        _ => highlight_code_line(line, ext),
    }
}

fn highlight_markdown_line(line: &str) -> Vec<Span<'static>> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("```") {
        return vec![Span::styled(
            line.to_string(),
            Style::default().fg(Color::Magenta),
        )];
    }
    if trimmed.starts_with('#') {
        return vec![Span::styled(
            line.to_string(),
            Style::default().fg(Color::Cyan).bold(),
        )];
    }
    if trimmed.starts_with('>') {
        return vec![Span::styled(line.to_string(), meta_style())];
    }

    let mut spans = Vec::new();
    let mut buf = String::new();
    let mut in_code = false;
    for ch in line.chars() {
        if ch == '`' {
            if !buf.is_empty() {
                let style = if in_code {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::Rgb(170, 170, 170))
                };
                spans.push(Span::styled(buf.clone(), style));
                buf.clear();
            }
            in_code = !in_code;
            spans.push(Span::styled(
                "`".to_string(),
                Style::default().fg(Color::Yellow),
            ));
            continue;
        }
        buf.push(ch);
    }
    if !buf.is_empty() {
        let style = if in_code {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::Rgb(170, 170, 170))
        };
        spans.push(Span::styled(buf, style));
    }
    spans
}

fn highlight_code_line(line: &str, ext: Option<&str>) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    let chars: Vec<char> = line.chars().collect();
    let allow_hash_comment = matches!(
        ext,
        Some("py")
            | Some("rb")
            | Some("sh")
            | Some("bash")
            | Some("zsh")
            | Some("yml")
            | Some("yaml")
            | Some("toml")
    );
    let allow_backtick = matches!(ext, Some("js") | Some("ts") | Some("jsx") | Some("tsx"));

    while i < chars.len() {
        let ch = chars[i];

        // Comment start
        if allow_hash_comment && ch == '#' {
            if !buf.is_empty() {
                spans.push(Span::styled(
                    buf.clone(),
                    Style::default().fg(Color::Rgb(170, 170, 170)),
                ));
                buf.clear();
            }
            let rest: String = chars[i..].iter().collect();
            spans.push(Span::styled(
                rest,
                Style::default().fg(Color::Rgb(160, 160, 160)),
            ));
            return spans;
        }
        if ch == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            if !buf.is_empty() {
                spans.push(Span::styled(
                    buf.clone(),
                    Style::default().fg(Color::Rgb(170, 170, 170)),
                ));
                buf.clear();
            }
            let rest: String = chars[i..].iter().collect();
            spans.push(Span::styled(
                rest,
                Style::default().fg(Color::Rgb(160, 160, 160)),
            ));
            return spans;
        }

        // String start
        if ch == '"' || ch == '\'' || (allow_backtick && ch == '`') {
            if !buf.is_empty() {
                spans.push(Span::styled(
                    buf.clone(),
                    Style::default().fg(Color::Rgb(170, 170, 170)),
                ));
                buf.clear();
            }
            let quote = ch;
            let mut j = i + 1;
            while j < chars.len() {
                if chars[j] == '\\' {
                    j += 2;
                    continue;
                }
                if chars[j] == quote {
                    j += 1;
                    break;
                }
                j += 1;
            }
            let segment: String = chars[i..j.min(chars.len())].iter().collect();
            spans.push(Span::styled(
                segment,
                Style::default().fg(Color::Rgb(134, 239, 172)),
            ));
            i = j;
            continue;
        }

        buf.push(ch);
        i += 1;
    }

    if !buf.is_empty() {
        spans.push(Span::styled(
            buf,
            Style::default().fg(Color::Rgb(170, 170, 170)),
        ));
    }

    spans
}

fn truncate_to_width(text: &str, width: usize) -> String {
    if UnicodeWidthStr::width(text) <= width {
        return text.to_string();
    }

    if width <= 1 {
        return "…".to_string();
    }

    let mut out = String::new();
    let mut used = 0;
    for ch in text.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + w + 1 > width {
            break;
        }
        out.push(ch);
        used = used.saturating_add(w);
    }
    out.push('…');
    out
}
