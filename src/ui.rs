use ratatui::{
    layout::Margin,
    prelude::*,
    widgets::{
        Block, BorderType, Borders, Clear, List, ListItem, ListState, Paragraph, Scrollbar,
        ScrollbarOrientation, ScrollbarState, Wrap,
    },
};

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::app::{
    App, IndexingStatus, Pane, PlanFocus, PlanQuestionKind, Role, ToolOutput, ToolStatus,
};
use crate::markdown;
use crate::theme;

const INDEXING_SPINNER: &[&str] = &["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const TOOL_SPINNER_FRAMES: &[&str] = &["◐", "◓", "◑", "◒"];

pub fn ui(frame: &mut Frame, app: &mut App) {
    let ui_theme = theme::ui_theme(&app.config, Some(app.workspace_root_path()));
    let pal = ui_theme.palette;
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

    let splitter_style = pal.border_style();
    app.splitter_area = main_chunks[1];
    let splitter = Paragraph::new("│".repeat(main_chunks[1].height as usize)).style(splitter_style);
    frame.render_widget(splitter, main_chunks[1]);

    let chat_border_style = pal.border_style();

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
            Role::User => ("▶ ", theme::user_markdown_theme(&ui_theme)),
            Role::Agent => ("● ", ui_theme.markdown),
            Role::System => ("◆ ", ui_theme.markdown),
            Role::Tool => ("⚙ ", ui_theme.markdown),
        };

        // Add prefix line
        let prefix_style = match msg.role {
            Role::User => Style::default().fg(Color::Cyan).bold(),
            Role::Agent => Style::default().fg(pal.accent).bold(),
            Role::System => Style::default().fg(pal.warn).bold(),
            Role::Tool => Style::default().fg(pal.accent).bold(),
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
                    all_lines.push(Line::from(Span::styled(text, pal.ghost())));
                }
            }
        }

        let md_lines = markdown::render(
            &msg.content,
            theme,
            ui_theme.syntax,
            chat_width.saturating_sub(2),
        );

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
                render_tool_group_inline(
                    &*app,
                    &ui_theme,
                    group_id,
                    tool_spinner,
                    chat_width,
                    &mut all_lines,
                );
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
                    all_lines.push(Line::from(Span::styled(text, pal.ghost())));
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
                Span::styled(format!("{} ", spinner), Style::default().fg(pal.warn)),
                Span::styled(time_str, pal.meta()),
            ]));
        } else if !app.current_response.is_empty() {
            // Show streaming content with spinner
            let theme = ui_theme.markdown;
            let md_lines = markdown::render(
                &app.current_response,
                theme,
                ui_theme.syntax,
                chat_width.saturating_sub(4),
            );

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
        if app.plan_mode {
            format!("* Lorikeet · {} · PLAN ", app.model)
        } else {
            format!("* Lorikeet · {} ", app.model)
        }
    } else {
        if app.plan_mode {
            format!("  Lorikeet · {} · PLAN ", app.model)
        } else {
            format!("  Lorikeet · {} ", app.model)
        }
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

    if app.active_pane == Pane::Chat
        && !app.is_processing
        && !app.settings_open
        && !app.themes_open
        && !app.plan_popup_open
    {
        let cursor_x = left_chunks[1].x + app.cursor_pos as u16 + 1;
        let cursor_y = left_chunks[1].y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    // Status bar (minimal; keep other shortcuts discoverable via /help)
    let status_text = " ESC quit │ TAB plan │ ENTER send";
    let status = Paragraph::new(status_text).style(pal.meta());
    frame.render_widget(status, left_chunks[2]);

    // Context sidebar (right pane)
    app.context_area = main_chunks[2];
    render_context_sidebar(frame, app, main_chunks[2], &ui_theme);

    // Slash command suggestions overlay (while typing)
    if !app.settings_open && !app.themes_open && !app.plan_popup_open {
        render_command_suggestions_overlay(frame, app, left_chunks[1], pal);
    }
    if app.settings_open {
        render_settings_popup(frame, app);
    }
    if app.themes_open {
        render_themes_popup(frame, app);
    }
    if app.plan_popup_open {
        render_plan_popup(frame, app, &ui_theme);
    }
}

fn render_command_suggestions_overlay(
    frame: &mut Frame,
    app: &App,
    input_area: Rect,
    pal: theme::UiPalette,
) {
    let t = app.input.trim_start();
    // Show suggestions immediately when the user types '/', not only after a second character.
    if !t.starts_with('/') {
        return;
    }
    let suggestions = app.command_suggestions(t);
    if suggestions.is_empty() {
        return;
    }

    let max_items = suggestions.len().min(6);
    let h = (max_items as u16).saturating_add(2);
    let y = input_area.y.saturating_sub(h);
    let area = Rect {
        x: input_area.x,
        y,
        width: input_area.width,
        height: h,
    };

    frame.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(pal.border_style())
        .title(Span::styled(" Commands ", pal.meta()));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let w = inner.width.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::new();
    let selected = app
        .command_suggest_selected
        .min(max_items.saturating_sub(1));
    for (i, (cmd, desc)) in suggestions.into_iter().take(max_items).enumerate() {
        let s = format!("{:<10} {}", cmd, desc);
        let style = if i == selected {
            pal.selection()
        } else {
            Style::default().fg(pal.fg)
        };
        lines.push(Line::from(Span::styled(truncate_line(&s, w), style)));
    }

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
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
    let pal = theme::ui_palette(&app.config, Some(app.workspace_root_path()));
    let area = frame.area();
    let popup_area = centered_rect(78, 75, area);

    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(pal.border_style())
        .title(Span::styled(
            " Settings ",
            Style::default().fg(pal.accent).bold(),
        ));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(inner);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(18),
            Constraint::Length(1),
            Constraint::Min(10),
        ])
        .split(chunks[0]);

    // Categories (left)
    let cat_rows = app.settings_category_rows();
    let cat_focused = app.settings_focus_is_categories();
    let cat_width = top[0].width.saturating_sub(2) as usize;
    let mut cat_lines: Vec<Line> = Vec::new();
    for (i, label) in cat_rows.iter().enumerate() {
        let is_selected = i == app.settings_category_selected;
        let mut style = Style::default().fg(pal.fg);
        if is_selected && cat_focused {
            style = pal.selection();
        } else if is_selected && !cat_focused {
            style = pal.meta();
        }
        let glyph = if is_selected { "▶ " } else { "  " };
        let s = truncate_line(&format!("{}{}", glyph, label), cat_width);
        cat_lines.push(Line::from(Span::styled(s, style)));
    }

    let cats_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(pal.border_style())
        .title(Span::styled(" Categories ", pal.meta()));
    frame.render_widget(
        Paragraph::new(cat_lines)
            .block(cats_block)
            .wrap(Wrap { trim: true }),
        top[0],
    );

    // Divider
    frame.render_widget(
        Paragraph::new("│".repeat(top[1].height as usize)).style(pal.border_style()),
        top[1],
    );

    // Items (right)
    let rows = app.settings_rows();
    let item_focused = !cat_focused;
    let item_width = top[2].width.saturating_sub(2) as usize;
    let mut item_lines: Vec<Line> = Vec::new();
    for (idx, (label, value)) in rows.iter().enumerate() {
        let is_selected = idx == app.settings_selected;
        let mut style = Style::default().fg(pal.fg);
        if is_selected && item_focused {
            style = pal.selection();
        } else if is_selected && !item_focused {
            style = pal.meta();
        }
        let display = format!("{:<18} {}", label, value);
        item_lines.push(Line::from(Span::styled(
            truncate_line(&display, item_width),
            style,
        )));
    }

    let items_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(pal.border_style())
        .title(Span::styled(" Settings ", pal.meta()));
    frame.render_widget(
        Paragraph::new(item_lines)
            .block(items_block)
            .wrap(Wrap { trim: true }),
        top[2],
    );

    // Value editor
    let editor_title = format!(" Value — {} ", app.settings_selected_label());
    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(pal.border_style())
        .title(Span::styled(editor_title, pal.meta()));
    let input = Paragraph::new(app.settings_input.as_str())
        .block(input_block)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(pal.fg));
    frame.render_widget(input, chunks[1]);

    let hints =
        Paragraph::new(" Enter save • Esc cancel • Tab focus • ↑↓ navigate • ←→ cycle theme")
            .style(pal.meta());
    frame.render_widget(hints, chunks[2]);

    // Cursor only when editing items.
    if item_focused {
        let cursor_x = chunks[1].x + 1 + app.settings_cursor as u16;
        let cursor_y = chunks[1].y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

fn render_themes_popup(frame: &mut Frame, app: &mut App) {
    let pal = theme::ui_theme(&app.config, Some(app.workspace_root_path())).palette;
    let area = frame.area();
    let popup_area = centered_rect(72, 60, area);

    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(pal.border_style())
        .title(Span::styled(
            " Themes ",
            Style::default().fg(pal.accent).bold(),
        ));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(inner);

    let filter_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(pal.border_style())
        .title(Span::styled(" Filter ", pal.meta()));
    let filter = Paragraph::new(app.themes_query.as_str())
        .block(filter_block)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(pal.fg));
    frame.render_widget(filter, chunks[0]);
    frame.set_cursor_position((chunks[0].x + 1 + app.themes_cursor as u16, chunks[0].y + 1));

    let items = app.filtered_themes();
    let content = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(45),
            Constraint::Length(1),
            Constraint::Percentage(55),
        ])
        .split(chunks[1]);

    // Divider
    frame.render_widget(
        Paragraph::new("│".repeat(content[1].height as usize)).style(pal.border_style()),
        content[1],
    );

    // Theme list (left)
    let list_w = content[0].width.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::new();
    for (i, name) in items.iter().enumerate() {
        let is_sel = i == app.themes_selected;
        let style = if is_sel {
            pal.selection()
        } else {
            Style::default().fg(pal.fg)
        };

        let tag = theme::builtin_theme_tagline(name).unwrap_or("custom");
        let s = format!("{:<12} {}", name, tag);
        lines.push(Line::from(Span::styled(truncate_line(&s, list_w), style)));
    }
    let list_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(pal.border_style())
        .title(Span::styled(" Available ", pal.meta()));
    frame.render_widget(
        Paragraph::new(lines)
            .block(list_block)
            .wrap(Wrap { trim: true }),
        content[0],
    );

    // Preview (right)
    let sel_name = items
        .get(app.themes_selected)
        .cloned()
        .unwrap_or_else(|| "system".to_string());
    let preview_theme = theme::ui_theme_by_name(&sel_name, Some(app.workspace_root_path()));

    let preview_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .border_style(pal.border_style())
        .title(Span::styled(" Preview ", pal.meta()));
    let preview_inner = preview_block.inner(content[2]);
    frame.render_widget(preview_block, content[2]);

    const SAMPLE: &str = "# Preview\n\n- headings, lists, links\n- `inline code`\n\n```rs\nfn main() { println!(\"hi\"); }\n```";
    let w = preview_inner.width.saturating_sub(2) as usize;
    let mut preview_lines =
        markdown::render(SAMPLE, preview_theme.markdown, preview_theme.syntax, w);

    // Add a tool trace sample beneath the markdown preview.
    preview_lines.push(Line::from(""));
    preview_lines.push(Line::from(Span::styled(
        "Tool Trace",
        Style::default().fg(preview_theme.tool_trace.title).bold(),
    )));
    preview_lines.push(Line::from(vec![
        Span::styled("● ", Style::default().fg(preview_theme.palette.ok)),
        Span::styled(
            "bash ",
            Style::default()
                .fg(preview_theme.tool_trace.invocation)
                .bold(),
        ),
        Span::styled(
            "rg -n \"SandboxPolicy\" src",
            Style::default().fg(preview_theme.tool_trace.invocation),
        ),
        Span::styled(
            " [id=call_123] (cwd=repo) (12ms)",
            Style::default()
                .fg(preview_theme.tool_trace.call_id)
                .add_modifier(Modifier::DIM),
        ),
    ]));
    preview_lines.push(Line::from(vec![
        Span::styled(
            "└ sandbox: ",
            Style::default().fg(preview_theme.tool_trace.details_key),
        ),
        Span::styled(
            "allow",
            Style::default().fg(preview_theme.tool_trace.sandbox_allow),
        ),
    ]));

    frame.render_widget(
        Paragraph::new(preview_lines).wrap(Wrap { trim: true }),
        preview_inner,
    );

    let hints =
        Paragraph::new(" Enter apply • Esc close • ↑↓ select • type to filter").style(pal.meta());
    frame.render_widget(hints, chunks[2]);
}

fn render_plan_popup(frame: &mut Frame, app: &mut App, ui_theme: &theme::UiTheme) {
    let pal = ui_theme.palette;
    let area = frame.area();
    let popup_w = area.width.saturating_mul(85) / 100;
    let popup_h = area.height.saturating_mul(80) / 100;
    let popup_area = Rect {
        x: area.x + (area.width.saturating_sub(popup_w) / 2),
        y: area.y + (area.height.saturating_sub(popup_h) / 2),
        width: popup_w.max(40),
        height: popup_h.max(16),
    };

    frame.render_widget(Clear, popup_area);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(pal.border_style())
        .title(" Plan ");
    frame.render_widget(block, popup_area);

    let inner = popup_area.inner(Margin {
        vertical: 1,
        horizontal: 1,
    });

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // header / error
            Constraint::Min(1),    // main
            Constraint::Length(3), // buttons + hints
        ])
        .split(inner);

    // Header (parse error if any)
    let mut header_spans = vec![Span::styled(
        "Plan review",
        Style::default().fg(pal.accent).bold(),
    )];
    if let Some(err) = &app.plan_parse_error {
        header_spans.push(Span::raw("  "));
        header_spans.push(Span::styled(
            format!("parse error: {err}"),
            Style::default().fg(pal.err),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(header_spans)), rows[0]);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(32), Constraint::Percentage(68)])
        .split(rows[1]);

    // Questions list
    let questions = app
        .plan_draft
        .as_ref()
        .map(|d| d.questions.clone())
        .unwrap_or_default();
    let q_items: Vec<ListItem> = if questions.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "(no questions)",
            pal.meta(),
        )))]
    } else {
        questions
            .iter()
            .enumerate()
            .map(|(_i, q)| {
                let mut line = format!("{}", q.prompt);
                if let Some(ans) = app
                    .plan_draft
                    .as_ref()
                    .and_then(|d| d.answers.get(&q.id))
                    .or_else(|| q.default.as_ref())
                {
                    if !ans.trim().is_empty() {
                        line.push_str(&format!("  ({})", ans));
                    }
                }
                ListItem::new(Line::from(line))
            })
            .collect()
    };

    let mut q_state = ListState::default();
    if !q_items.is_empty() && !questions.is_empty() {
        q_state.select(Some(
            app.plan_question_selected
                .min(q_items.len().saturating_sub(1)),
        ));
    }

    let q_block = Block::default()
        .borders(Borders::ALL)
        .border_style(if matches!(app.plan_focus, PlanFocus::Questions) {
            pal.selection()
        } else {
            pal.border_style()
        })
        .title(" Questions ");
    let q_list = List::new(q_items)
        .block(q_block)
        .highlight_style(pal.selection())
        .highlight_symbol("› ");
    frame.render_stateful_widget(q_list, cols[0], &mut q_state);

    // Right side: answer + plan preview
    let right_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(1)])
        .split(cols[1]);

    let answer_block = Block::default()
        .borders(Borders::ALL)
        .border_style(if matches!(app.plan_focus, PlanFocus::Answer) {
            pal.selection()
        } else {
            pal.border_style()
        })
        .title(" Answer ");

    let kind = app.plan_selected_question_kind();
    let answer_text = match kind {
        Some(PlanQuestionKind::Select) => {
            let opts = app
                .plan_draft
                .as_ref()
                .and_then(|d| d.questions.get(app.plan_question_selected))
                .map(|q| q.options.clone())
                .unwrap_or_default();
            if opts.is_empty() {
                format!("{}\n\n(no options)", app.plan_answer_input)
            } else {
                let mut out = String::new();
                out.push_str(&format!("{}\n", app.plan_answer_input));
                out.push_str("\nOptions: ");
                for (i, o) in opts.iter().enumerate() {
                    if i > 0 {
                        out.push_str(" · ");
                    }
                    out.push_str(o);
                }
                out
            }
        }
        _ => app.plan_answer_input.clone(),
    };
    frame.render_widget(
        Paragraph::new(answer_text)
            .block(answer_block)
            .wrap(Wrap { trim: false }),
        right_rows[0],
    );

    // Cursor in answer field (only for text questions)
    if matches!(app.plan_focus, PlanFocus::Answer) && kind == Some(PlanQuestionKind::Text) {
        let x0 = right_rows[0].x + 1;
        let y0 = right_rows[0].y + 1;
        let max_x = right_rows[0].right().saturating_sub(2);
        let cursor_x = (x0 + app.plan_answer_cursor as u16).min(max_x);
        frame.set_cursor_position((cursor_x, y0));
    }

    let plan_block = Block::default()
        .borders(Borders::ALL)
        .border_style(pal.border_style())
        .title(" Plan ");

    let plan_text = app
        .plan_draft
        .as_ref()
        .map(|d| d.plan_markdown.clone())
        .unwrap_or_else(|| "(no plan)".into());
    let w = right_rows[1].width.saturating_sub(4) as usize;
    let md_lines = markdown::render(&plan_text, ui_theme.markdown, ui_theme.syntax, w.max(10));
    let plan_para = Paragraph::new(md_lines)
        .block(plan_block)
        .scroll((app.plan_preview_scroll as u16, 0));
    frame.render_widget(plan_para, right_rows[1]);

    // Buttons
    let buttons_area = rows[2];
    let buttons_inner = buttons_area.inner(Margin {
        vertical: 0,
        horizontal: 1,
    });

    let btn_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(12),
            Constraint::Length(12),
            Constraint::Min(1),
        ])
        .split(buttons_inner);

    let exec_style =
        if matches!(app.plan_focus, PlanFocus::Buttons) && app.plan_button_selected == 0 {
            pal.selection()
        } else {
            Style::default().fg(pal.accent)
        };
    let cancel_style =
        if matches!(app.plan_focus, PlanFocus::Buttons) && app.plan_button_selected == 1 {
            pal.selection()
        } else {
            Style::default().fg(pal.warn)
        };

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(" Execute ", exec_style))).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(pal.border_style()),
        ),
        btn_cols[0],
    );
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(" Cancel ", cancel_style))).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(pal.border_style()),
        ),
        btn_cols[1],
    );

    let hint = "Tab focus · ↑↓ navigate · Enter select · Esc close";
    frame.render_widget(Paragraph::new(hint).style(pal.meta()), btn_cols[2]);
}

fn render_tool_group_inline(
    app: &App,
    ui_theme: &theme::UiTheme,
    group_id: u64,
    tool_spinner: &str,
    chat_width: usize,
    out: &mut Vec<Line<'static>>,
) {
    let pal = ui_theme.palette;
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
            Style::default().fg(ui_theme.tool_trace.title).bold(),
        ),
        Span::styled(
            status,
            Style::default().fg(if any_running { pal.warn } else { pal.ok }),
        ),
    ]));

    for t in tools {
        render_tool_trace_item(
            app,
            ui_theme,
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
    ui_theme: &theme::UiTheme,
    tool: &ToolOutput,
    group_expanded: bool,
    show_details: bool,
    tool_spinner: &str,
    chat_width: usize,
    out: &mut Vec<Line<'static>>,
) {
    let pal = ui_theme.palette;
    let (status_indicator, status_color) = match tool.status {
        ToolStatus::Running => (tool_spinner, pal.warn),
        ToolStatus::Success => ("●", pal.ok),
        ToolStatus::Error => ("●", pal.err),
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
                    ToolStatus::Running => pal.warn,
                    ToolStatus::Success => ui_theme.tool_trace.invocation,
                    ToolStatus::Error => pal.err,
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

    let want_details = (group_expanded && show_details) || tool.status == ToolStatus::Error;
    if want_details {
        let (label, style) = if tool.sandbox.allowed {
            (
                "allow".to_string(),
                Style::default().fg(ui_theme.tool_trace.sandbox_allow),
            )
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
                Style::default().fg(ui_theme.tool_trace.sandbox_deny),
            )
        };

        out.push(Line::from(vec![
            Span::styled(
                "  └ sandbox: ",
                Style::default().fg(ui_theme.tool_trace.details_key),
            ),
            Span::styled(label, style),
        ]));

        // Input JSON (redacted + pretty-printed at capture time)
        const MAX_INPUT_LINES: usize = 8;
        if !tool.args_pretty_lines.is_empty() {
            out.push(Line::from(vec![
                Span::styled(
                    "  └ input: ",
                    Style::default().fg(ui_theme.tool_trace.details_key),
                ),
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
                        format!(
                            "… {} more lines",
                            tool.args_pretty_lines.len() - MAX_INPUT_LINES
                        ),
                        Style::default()
                            .fg(ui_theme.tool_trace.details_value)
                            .add_modifier(Modifier::DIM),
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
                            Style::default()
                                .fg(ui_theme.tool_trace.out_prefix)
                                .add_modifier(Modifier::DIM),
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
            Style::default()
                .fg(ui_theme.tool_trace.out_prefix)
                .add_modifier(Modifier::DIM),
        ));

        let text = truncate_to_width(l, chat_width.saturating_sub(prefix.len()));
        if tool.tool == "list_files" {
            let style = theme::style_for_filename_with_theme(&text, &app.config, ui_theme);
            spans.push(Span::styled(text, style));
        } else if tool.tool == "read_file" {
            spans.extend(highlight_line_for_ext(&text, ext, ui_theme));
        } else {
            let base_style = if tool.status == ToolStatus::Error {
                Style::default().fg(pal.err)
            } else {
                Style::default().fg(ui_theme.tool_trace.out_text)
            };
            spans.push(Span::styled(text, base_style));
        }

        out.push(Line::from(spans));
    }

    if remaining > 0 || tool.output_is_truncated() {
        out.push(Line::from(vec![
            Span::raw("        "),
            Span::styled(
                format!("… {} more lines", remaining),
                Style::default()
                    .fg(ui_theme.tool_trace.out_prefix)
                    .add_modifier(Modifier::DIM),
            ),
        ]));
    }
}

fn render_context_sidebar(frame: &mut Frame, app: &mut App, area: Rect, ui_theme: &theme::UiTheme) {
    let pal = ui_theme.palette;
    let title = if app.active_pane == Pane::Context {
        " * Context "
    } else {
        "   Context "
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(pal.border_style())
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
        Span::styled("Workspace: ", pal.meta()),
        Span::raw(truncate_to_width(
            &app.workspace_root_display(),
            width.saturating_sub(11),
        )),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Model: ", pal.meta()),
        Span::raw(truncate_to_width(&app.model, width.saturating_sub(7))),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Provider: ", pal.meta()),
        Span::raw(truncate_to_width(
            app.llm_provider_name(),
            width.saturating_sub(10),
        )),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Mode: ", pal.meta()),
        Span::raw(if app.plan_mode { "plan" } else { "auto" }),
    ]));

    let sandbox_enabled = app
        .config
        .sandbox
        .as_ref()
        .and_then(|s| s.enabled)
        .unwrap_or(true);
    lines.push(Line::from(vec![
        Span::styled("Sandbox: ", pal.meta()),
        Span::raw(if sandbox_enabled { "on" } else { "off" }),
    ]));

    if !app.verify_suggestions.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("Verify: ", pal.meta()),
            Span::raw(truncate_to_width(
                &app.verify_suggestions[0].command,
                width.saturating_sub(8),
            )),
        ]));
        lines.push(Line::from(Span::styled("Run: /verify", pal.meta())));
    }

    if let Some(cp) = &app.last_checkpoint {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("Last checkpoint: ", pal.meta()),
            Span::raw(truncate_to_width(&cp.id, width.saturating_sub(17))),
        ]));
        if let Some(name) = &cp.name {
            if !name.trim().is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("Name: ", pal.meta()),
                    Span::raw(truncate_to_width(name, width.saturating_sub(6))),
                ]));
            }
        }
        lines.push(Line::from(vec![
            Span::styled("Backend: ", pal.meta()),
            Span::raw(format!("{:?}", cp.backend).to_lowercase()),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Recent files",
        Style::default().fg(Color::Reset).bold(),
    )));
    if app.recent_files.is_empty() {
        lines.push(Line::from(Span::styled("(none yet)", pal.meta())));
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

fn highlight_line_for_ext(
    line: &str,
    ext: Option<&str>,
    ui_theme: &theme::UiTheme,
) -> Vec<Span<'static>> {
    match ext {
        Some("md") | Some("mdx") => highlight_markdown_line(line, ui_theme),
        _ => highlight_code_line(line, ext, ui_theme),
    }
}

fn highlight_markdown_line(line: &str, ui_theme: &theme::UiTheme) -> Vec<Span<'static>> {
    let md = ui_theme.markdown;
    let trimmed = line.trim_start();
    if trimmed.starts_with("```") {
        return vec![Span::styled(
            line.to_string(),
            Style::default().fg(ui_theme.syntax.keyword),
        )];
    }
    if trimmed.starts_with('#') {
        return vec![Span::styled(
            line.to_string(),
            Style::default().fg(md.heading).bold(),
        )];
    }
    if trimmed.starts_with('>') {
        return vec![Span::styled(
            line.to_string(),
            Style::default()
                .fg(md.blockquote)
                .add_modifier(Modifier::DIM | Modifier::ITALIC),
        )];
    }

    let mut spans = Vec::new();
    let mut buf = String::new();
    let mut in_code = false;
    for ch in line.chars() {
        if ch == '`' {
            if !buf.is_empty() {
                let style = if in_code {
                    Style::default().fg(md.code)
                } else {
                    Style::default().fg(md.text)
                };
                spans.push(Span::styled(buf.clone(), style));
                buf.clear();
            }
            in_code = !in_code;
            spans.push(Span::styled("`".to_string(), Style::default().fg(md.code)));
            continue;
        }
        buf.push(ch);
    }
    if !buf.is_empty() {
        let style = if in_code {
            Style::default().fg(md.code)
        } else {
            Style::default().fg(md.text)
        };
        spans.push(Span::styled(buf, style));
    }
    spans
}

fn highlight_code_line(
    line: &str,
    ext: Option<&str>,
    ui_theme: &theme::UiTheme,
) -> Vec<Span<'static>> {
    let syn = ui_theme.syntax;
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
                spans.push(Span::styled(buf.clone(), Style::default().fg(syn.ident)));
                buf.clear();
            }
            let rest: String = chars[i..].iter().collect();
            spans.push(Span::styled(rest, Style::default().fg(syn.comment)));
            return spans;
        }
        if ch == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            if !buf.is_empty() {
                spans.push(Span::styled(buf.clone(), Style::default().fg(syn.ident)));
                buf.clear();
            }
            let rest: String = chars[i..].iter().collect();
            spans.push(Span::styled(rest, Style::default().fg(syn.comment)));
            return spans;
        }

        // String start
        if ch == '"' || ch == '\'' || (allow_backtick && ch == '`') {
            if !buf.is_empty() {
                spans.push(Span::styled(buf.clone(), Style::default().fg(syn.ident)));
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
            spans.push(Span::styled(segment, Style::default().fg(syn.string)));
            i = j;
            continue;
        }

        buf.push(ch);
        i += 1;
    }

    if !buf.is_empty() {
        spans.push(Span::styled(buf, Style::default().fg(syn.ident)));
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
