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
    App, IndexingStatus, Pane, PlanFocus, PlanQuestionKind, ToolOutput, ToolStatus,
};
use crate::markdown;
use crate::theme;

const INDEXING_SPINNER: &[&str] = &["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const TOOL_SPINNER_FRAMES: &[&str] = &["◐", "◓", "◑", "◒"];

#[derive(Clone, Copy)]
struct Fill {
    style: Style,
}

impl Fill {
    fn new(style: Style) -> Self {
        Self { style }
    }
}

impl Widget for Fill {
    fn render(self, area: Rect, buf: &mut Buffer) {
        buf.set_style(area, self.style);
    }
}

pub fn ui(frame: &mut Frame, app: &mut App) {
    let ui_theme = theme::ui_theme(&app.config, Some(app.workspace_root_path()));
    let pal = ui_theme.palette;
    app.root_area = frame.area();

    // "Opacity" in TUIs is just background fill. We must explicitly paint the buffer;
    // rendering a styled Block does not reliably fill all cells.
    let bg_style = if pal.bg == Color::Reset {
        Style::default()
    } else {
        Style::default().bg(pal.bg).fg(pal.fg)
    };
    frame.render_widget(Fill::new(bg_style), frame.area());

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
    let _tool_spinner = TOOL_SPINNER_FRAMES[app.tool_spinner_frame];

    // Advance indexing spinner (used by context sidebar)
    app.indexing_spinner_frame = (app.indexing_spinner_frame + 1) % INDEXING_SPINNER.len();

    if app.is_processing {
        app.spinner_frame = (app.spinner_frame + 1) % SPINNER_FRAMES.len();
    }

    // Keep running tool spinners alive without forcing a full transcript rebuild.
    for gid in app
        .tool_outputs
        .iter()
        .filter(|t| t.status == ToolStatus::Running && t.group_id > 0)
        .map(|t| t.group_id)
    {
        app.render_store
            .mark_dirty(crate::render_store::RenderedBlockId::ToolGroup(gid));
    }

    // Cached + virtualized transcript render.
    let mut store = std::mem::take(&mut app.render_store);
    store.ensure_up_to_date(&*app, &ui_theme, chat_width);
    app.render_store = store;

    let total_lines = app.render_store.total_height();
    let visible_height = left_chunks[0].height.saturating_sub(2) as usize; // subtract borders
    let max_scroll = total_lines.saturating_sub(visible_height);

    if app.chat_follow {
        app.chat_scroll_rows = max_scroll;
    } else {
        app.chat_scroll_rows = app.chat_scroll_rows.min(max_scroll);
    }
    if app.chat_scroll_rows >= max_scroll.saturating_sub(1) {
        app.chat_follow = true;
    }

    let visible_lines = app
        .render_store
        .visible_lines(app.chat_scroll_rows, visible_height);

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
    let messages_widget = Paragraph::new(visible_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(chat_border_style)
                .title(chat_title),
        );
    frame.render_widget(messages_widget, left_chunks[0]);

    // Chat scrollbar - use max_scroll as content length so thumb reaches bottom
    let scrollbar_content = total_lines.max(1);
    let mut chat_scrollbar_state =
        ScrollbarState::new(scrollbar_content).position(app.chat_scroll_rows);
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

    // Give the input bar a stronger surface color from the selected preset so it reads
    // as an "active composer" even when the overall UI uses background inherit mode.
    let preset_name = theme::ui_theme_name(&app.config);
    let preset_theme = theme::ui_theme_by_name(&preset_name, Some(app.workspace_root_path()));
    let input_bg = if preset_theme.palette.bg == Color::Reset {
        pal.bg
    } else {
        preset_theme.palette.bg
    };
    let input_fg = if preset_theme.palette.fg == Color::Reset {
        pal.fg
    } else {
        preset_theme.palette.fg
    };

    if input_bg != Color::Reset {
        frame.render_widget(
            Fill::new(Style::default().bg(input_bg).fg(input_fg)),
            left_chunks[1],
        );
    }

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(chat_border_style)
        .title(input_title);

    let input_widget = Paragraph::new(app.input.as_str())
        .style(Style::default().fg(input_fg).bg(input_bg))
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

fn render_vsplit(frame: &mut Frame, area: Rect, style: Style) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    frame.render_widget(
        Paragraph::new("│".repeat(area.height as usize)).style(style),
        area,
    );
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
    if pal.bg != Color::Reset {
        frame.render_widget(
            Fill::new(Style::default().bg(pal.bg).fg(pal.fg)),
            area,
        );
    }
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
    // Settings should feel like a focused modal surface (opaque), even when the main UI
    // inherits terminal background. Use the selected preset's raw palette here.
    let preset = theme::ui_theme_name(&app.config);
    let modal_theme = theme::ui_theme_by_name(&preset, Some(app.workspace_root_path()));
    let pal = modal_theme.palette;
    let area = frame.area();
    let popup_area = centered_rect(74, 66, area);
    app.settings_popup_area = popup_area;

    frame.render_widget(Clear, popup_area);
    let modal_bg = if pal.bg == Color::Reset {
        Color::Black
    } else {
        pal.bg
    };
    let modal_fg = if pal.fg == Color::Reset {
        Color::White
    } else {
        pal.fg
    };
    frame.render_widget(
        Fill::new(Style::default().bg(modal_bg).fg(modal_fg)),
        popup_area,
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(pal.border_style())
        .title(Line::from(vec![
            Span::styled(" Settings ", Style::default().fg(pal.accent).bold()),
            Span::raw(" "),
            Span::styled("esc", pal.meta()),
        ]));
    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(18),
            Constraint::Length(30),
            Constraint::Min(20),
        ])
        .split(inner);

    let cat_focused = app.settings_focus_is_categories();
    let item_focused = !cat_focused;

    // Column separators (reduce boxy feel).
    render_vsplit(
        frame,
        Rect {
            x: cols[0].right().saturating_sub(1),
            y: cols[0].y,
            width: 1,
            height: cols[0].height,
        },
        pal.border_style(),
    );
    render_vsplit(
        frame,
        Rect {
            x: cols[1].right().saturating_sub(1),
            y: cols[1].y,
            width: 1,
            height: cols[1].height,
        },
        pal.border_style(),
    );

    // Categories list
    let cat_rows = app.settings_category_rows();
    let cat_items: Vec<ListItem> = cat_rows
        .iter()
        .map(|s| ListItem::new(Span::raw(s.clone())))
        .collect();
    let mut cat_state = ListState::default();
    cat_state.select(Some(
        app.settings_category_selected
            .min(cat_items.len().saturating_sub(1)),
    ));
    let cats = List::new(cat_items)
        .block(Block::default().borders(Borders::NONE))
        .highlight_style(pal.selection())
        .highlight_symbol("▶ ");
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled("Categories", pal.meta()))),
        Rect {
            x: cols[0].x + 1,
            y: cols[0].y,
            width: cols[0].width.saturating_sub(2),
            height: 1,
        },
    );
    let cats_area = cols[0].inner(Margin {
        vertical: 1,
        horizontal: 1,
    });
    app.settings_categories_area = cats_area;
    frame.render_stateful_widget(cats, cats_area, &mut cat_state);

    // Items list
    let rows = app.settings_rows();
    let item_items: Vec<ListItem> = rows
        .iter()
        .map(|(label, value)| {
            let spans: Vec<Span<'static>> = vec![
                Span::styled(format!("{label}: "), pal.meta()),
                Span::styled(value.clone(), Style::default().fg(pal.fg)),
            ];
            ListItem::new(Line::from(spans))
        })
        .collect();
    let mut item_state = ListState::default();
    item_state.select(Some(app.settings_selected.min(item_items.len().saturating_sub(1))));
    let items_list = List::new(item_items)
        .block(Block::default().borders(Borders::NONE))
        .highlight_style(pal.selection())
        .highlight_symbol("▶ ");
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled("Items", pal.meta()))),
        Rect {
            x: cols[1].x + 1,
            y: cols[1].y,
            width: cols[1].width.saturating_sub(2),
            height: 1,
        },
    );
    let items_area = cols[1].inner(Margin {
        vertical: 1,
        horizontal: 1,
    });
    app.settings_items_area = items_area;
    frame.render_stateful_widget(items_list, items_area, &mut item_state);

    // Details panel + editor
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(cols[2]);

    let mut detail_lines: Vec<Line<'static>> = Vec::new();
    detail_lines.push(Line::from(Span::styled(
        app.settings_selected_label(),
        Style::default().fg(pal.accent).bold(),
    )));
    detail_lines.push(Line::from(""));
    for l in settings_item_help(app) {
        detail_lines.push(Line::from(l));
    }

    let details = Paragraph::new(detail_lines)
        .block(Block::default().borders(Borders::NONE))
        .wrap(Wrap { trim: false });
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled("Details", pal.meta()))),
        Rect {
            x: cols[2].x + 1,
            y: cols[2].y,
            width: cols[2].width.saturating_sub(2),
            height: 1,
        },
    );
    frame.render_widget(
        details,
        right[0].inner(Margin {
            vertical: 1,
            horizontal: 1,
        }),
    );

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled("Value", pal.meta()))),
        Rect {
            x: right[1].x + 1,
            y: right[1].y,
            width: right[1].width.saturating_sub(2),
            height: 1,
        },
    );
    let editor_inner = Rect {
        x: right[1].x + 1,
        y: right[1].y + 1,
        width: right[1].width.saturating_sub(2),
        height: right[1].height.saturating_sub(1),
    };
    app.settings_editor_area = editor_inner;
    frame.render_widget(
        Paragraph::new(app.settings_input.as_str())
            .style(if item_focused {
                Style::default().fg(pal.fg)
            } else {
                pal.meta()
            })
            .wrap(Wrap { trim: false }),
        editor_inner,
    );

    if item_focused {
        let cursor_x = editor_inner.x + app.settings_cursor as u16;
        let cursor_y = editor_inner.y;
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    frame.render_widget(
        Paragraph::new("↑↓ navigate  TAB focus  ←→ cycle  click/select  wheel scroll  ENTER save  ESC cancel")
            .style(pal.meta()),
        right[2],
    );
}

fn settings_item_help(app: &App) -> Vec<String> {
    let label = app.settings_selected_label();
    match label.as_str() {
        "Provider" => vec![
            "Select which API/provider to use for LLM calls.".into(),
            "Use ←→ to cycle, then ENTER to save.".into(),
        ],
        "Model" => vec![
            "Model slug/name used by the selected provider.".into(),
            "Type to edit, then ENTER to save.".into(),
        ],
        "Theme" | "Theme preset" | "Theme Preset" => vec![
            "Controls UI colors + markdown/syntax styles.".into(),
            "Use ←→ to cycle, then ENTER to save.".into(),
        ],
        "Theme bg" | "Theme background" | "Theme Background" => vec![
            "Controls whether Lorikeet forces a solid background.".into(),
            "inherit = use terminal theme (cleaner). solid = paint the whole UI.".into(),
        ],
        "Auto index" | "Auto Index" => vec![
            "When enabled, Lorikeet indexes the workspace for semantic search.".into(),
            "Use ←→ to toggle, then ENTER to save.".into(),
        ],
        "Resume last session" | "Resume Last Session" => vec![
            "When enabled, Lorikeet resumes the last session on startup.".into(),
            "Use ←→ to toggle, then ENTER to save.".into(),
        ],
        "Sandbox enabled" | "Sandbox Enabled" => vec![
            "Policy-only sandbox for tools + file access.".into(),
            "Use ←→ to toggle, then ENTER to save.".into(),
        ],
        _ => vec![
            "Use ↑↓ to pick an item, then edit the value below.".into(),
            "ENTER saves; ESC cancels.".into(),
        ],
    }
}

fn render_themes_popup(frame: &mut Frame, app: &mut App) {
    let pal = theme::ui_theme(&app.config, Some(app.workspace_root_path())).palette;
    let area = frame.area();
    let popup_area = centered_rect(72, 60, area);

    frame.render_widget(Clear, popup_area);
    if pal.bg != Color::Reset {
        frame.render_widget(
            Fill::new(Style::default().bg(pal.bg).fg(pal.fg)),
            popup_area,
        );
    }

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
    if pal.bg != Color::Reset {
        frame.render_widget(
            Fill::new(Style::default().bg(pal.bg).fg(pal.fg)),
            popup_area,
        );
    }
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
