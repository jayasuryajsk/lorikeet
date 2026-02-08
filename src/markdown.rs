use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::prelude::*;

use crate::theme::{MarkdownTheme, SyntaxTheme};

#[derive(Default, Clone, Copy)]
struct StyleState {
    bold: bool,
    italic: bool,
    strikethrough: bool,
    code: bool,
    link: bool,
}

fn style_for_state(theme: MarkdownTheme, state: StyleState) -> Style {
    let mut style = Style::default().fg(theme.text);

    if state.code {
        // Prefer high-contrast foreground; background stays terminal default.
        style = style
            .fg(theme.code)
            .bg(theme.code_bg)
            .add_modifier(Modifier::BOLD);
    }

    if state.link {
        style = style.fg(theme.link).underlined();
    }

    if state.bold {
        style = style.fg(theme.bold).bold();
    }

    if state.italic {
        style = style.fg(theme.italic).italic();
    }

    if state.strikethrough {
        style = style
            .fg(theme.strikethrough)
            .add_modifier(Modifier::CROSSED_OUT);
    }

    style
}

// Syntax highlighting colors for code blocks
fn syntax_color(token: &str, lang: &str, syn: SyntaxTheme) -> Color {
    let keywords = [
        "fn", "let", "mut", "const", "if", "else", "match", "for", "while", "loop", "return",
        "break", "continue", "struct", "enum", "impl", "trait", "pub", "use", "mod", "async",
        "await", "self", "Self", "true", "false", "None", "Some", "Ok", "Err", "function", "var",
        "const", "class", "import", "export", "from", "default", "new", "def", "class", "import",
        "from", "as", "try", "except", "finally", "with", "lambda", "int", "str", "bool", "float",
        "void", "char", "String", "Vec", "Option", "Result",
    ];

    let types = [
        "i32", "i64", "u32", "u64", "f32", "f64", "usize", "isize", "bool", "char", "String",
        "str", "Vec", "Option", "Result", "Box", "Rc", "Arc", "HashMap", "HashSet",
    ];

    if token.starts_with("//") || (token.starts_with('#') && lang != "markdown") {
        syn.comment
    } else if keywords.contains(&token) {
        syn.keyword
    } else if types.contains(&token) {
        syn.ty
    } else if token.starts_with('"') || token.starts_with('\'') || token.starts_with('`') {
        syn.string
    } else if token.chars().all(|c| c.is_ascii_digit() || c == '.') && !token.is_empty() {
        syn.number
    } else {
        syn.ident
    }
}

fn highlight_code(code: &str, lang: &str, theme: MarkdownTheme, syn: SyntaxTheme) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut string_char = '"';
    let mut in_comment = false;

    for c in code.chars() {
        if in_comment {
            current.push(c);
            continue;
        }

        if !in_string && (c == '"' || c == '\'' || c == '`') {
            if !current.is_empty() {
                let color = syntax_color(&current, lang, syn);
                spans.push(Span::styled(
                    std::mem::take(&mut current),
                    Style::default().fg(color).bg(theme.code_bg),
                ));
            }
            in_string = true;
            string_char = c;
            current.push(c);
        } else if in_string && c == string_char {
            current.push(c);
            spans.push(Span::styled(
                std::mem::take(&mut current),
                Style::default().fg(Color::Green).bg(theme.code_bg),
            ));
            in_string = false;
        } else if in_string {
            current.push(c);
        } else if c == '/' && current.ends_with('/') {
            current.push(c);
            in_comment = true;
        } else if c.is_alphanumeric() || c == '_' {
            current.push(c);
        } else {
            if !current.is_empty() {
                let color = syntax_color(&current, lang, syn);
                spans.push(Span::styled(
                    std::mem::take(&mut current),
                    Style::default().fg(color).bg(theme.code_bg),
                ));
            }
            spans.push(Span::styled(
                c.to_string(),
                // Avoid hard-coded white, which becomes unreadable on light terminals.
                Style::default().fg(syn.punct).bg(theme.code_bg),
            ));
        }
    }

    if !current.is_empty() {
        let color = if in_comment {
            syn.comment
        } else if in_string {
            syn.string
        } else {
            syntax_color(&current, lang, syn)
        };
        spans.push(Span::styled(
            current,
            Style::default().fg(color).bg(theme.code_bg),
        ));
    }

    if spans.is_empty() {
        spans.push(Span::styled(
            code.to_string(),
            Style::default().fg(syn.ident).bg(theme.code_bg),
        ));
    }

    spans
}

pub fn render(text: &str, theme: MarkdownTheme, syn: SyntaxTheme, width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(text, options);

    let mut style_state = StyleState::default();
    let mut heading_level: Option<HeadingLevel> = None;
    let mut in_blockquote = false;

    let mut list_stack: Vec<(bool, usize)> = Vec::new(); // (ordered, counter)
    let mut in_list_item = false;
    let mut list_item_segments: Vec<Vec<Span<'static>>> = Vec::new();
    let mut current_segment: Vec<Span<'static>> = Vec::new();

    let mut paragraph_segments: Vec<Vec<Span<'static>>> = Vec::new();
    let mut in_paragraph = false;

    let mut in_code_block = false;
    let mut code_lang = String::new();
    let mut code_block_content: Vec<String> = Vec::new();

    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut in_table = false;

    let push_text = |text: &str, state: StyleState, target: &mut Vec<Span<'static>>| {
        if text.is_empty() {
            return;
        }
        target.push(Span::styled(
            text.to_string(),
            style_for_state(theme, state),
        ));
    };

    let flush_paragraph = |segments: &mut Vec<Vec<Span<'static>>>,
                           out: &mut Vec<Line<'static>>,
                           heading: Option<HeadingLevel>,
                           theme: MarkdownTheme,
                           width: usize,
                           in_blockquote: bool| {
        let total = segments.len();
        for (idx, seg) in segments.drain(..).enumerate() {
            if seg.is_empty() {
                continue;
            }

            let mut prefix: Option<String> = None;
            if in_blockquote {
                prefix = Some("┃ ".to_string());
            }

            let wrapped = wrap_spans(seg, prefix.as_deref(), theme, width);
            let mut final_lines = Vec::new();

            for line in wrapped {
                if let Some(level) = heading {
                    let styled = style_heading_line(line, level, theme);
                    final_lines.push(styled);
                } else if in_blockquote {
                    final_lines.push(style_blockquote_line(line, theme));
                } else {
                    final_lines.push(line);
                }
            }

            out.extend(final_lines);
            if idx + 1 == total {
                out.push(Line::from(""));
            }
        }
    };

    let flush_list_item = |segments: &mut Vec<Vec<Span<'static>>>,
                           out: &mut Vec<Line<'static>>,
                           list_stack: &mut Vec<(bool, usize)>,
                           theme: MarkdownTheme,
                           width: usize,
                           in_blockquote: bool| {
        if list_stack.is_empty() {
            return;
        }

        let (ordered, counter) = list_stack.last().copied().unwrap_or((false, 0));
        let indent = (list_stack.len().saturating_sub(1)) * 2;
        let prefix = if ordered {
            format!("{}{}. ", " ".repeat(indent), counter)
        } else {
            format!("{}• ", " ".repeat(indent))
        };

        let total = segments.len();
        for (idx, seg) in segments.drain(..).enumerate() {
            if seg.is_empty() {
                continue;
            }
            let wrapped = wrap_spans(seg, Some(&prefix), theme, width);
            for line in wrapped {
                if in_blockquote {
                    out.push(style_blockquote_line(line, theme));
                } else {
                    out.push(line);
                }
            }
            if idx + 1 == total {
                out.push(Line::from(""));
            }
        }
    };

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Paragraph => {
                    in_paragraph = true;
                    paragraph_segments.clear();
                    current_segment.clear();
                }
                Tag::Heading { level, .. } => {
                    heading_level = Some(level);
                    in_paragraph = true;
                    paragraph_segments.clear();
                    current_segment.clear();
                }
                Tag::BlockQuote(_) => {
                    in_blockquote = true;
                }
                Tag::List(start) => {
                    let ordered = start.is_some();
                    let counter = start.unwrap_or(1) as usize;
                    list_stack.push((ordered, counter));
                }
                Tag::Item => {
                    in_list_item = true;
                    list_item_segments.clear();
                    current_segment.clear();
                }
                Tag::Emphasis => style_state.italic = true,
                Tag::Strong => style_state.bold = true,
                Tag::Strikethrough => style_state.strikethrough = true,
                Tag::Link { .. } => style_state.link = true,
                Tag::CodeBlock(kind) => {
                    in_code_block = true;
                    code_block_content.clear();
                    code_lang = match kind {
                        CodeBlockKind::Fenced(lang) => lang.to_string(),
                        CodeBlockKind::Indented => String::new(),
                    };
                }
                Tag::Table(_) => {
                    in_table = true;
                    table_rows.clear();
                }
                Tag::TableRow => {
                    table_rows.push(Vec::new());
                }
                Tag::TableCell => {}
                _ => {}
            },
            Event::End(tag_end) => match tag_end {
                TagEnd::Paragraph => {
                    if !current_segment.is_empty() {
                        paragraph_segments.push(std::mem::take(&mut current_segment));
                    }
                    flush_paragraph(
                        &mut paragraph_segments,
                        &mut lines,
                        heading_level.take(),
                        theme,
                        width,
                        in_blockquote,
                    );
                    in_paragraph = false;
                }
                TagEnd::Heading(_) => {
                    if !current_segment.is_empty() {
                        paragraph_segments.push(std::mem::take(&mut current_segment));
                    }
                    flush_paragraph(
                        &mut paragraph_segments,
                        &mut lines,
                        heading_level.take(),
                        theme,
                        width,
                        in_blockquote,
                    );
                    in_paragraph = false;
                }
                TagEnd::BlockQuote(_) => {
                    in_blockquote = false;
                }
                TagEnd::List(_) => {
                    list_stack.pop();
                    lines.push(Line::from(""));
                }
                TagEnd::Item => {
                    if !current_segment.is_empty() {
                        list_item_segments.push(std::mem::take(&mut current_segment));
                    }
                    flush_list_item(
                        &mut list_item_segments,
                        &mut lines,
                        &mut list_stack,
                        theme,
                        width,
                        in_blockquote,
                    );
                    in_list_item = false;
                }
                TagEnd::Emphasis => style_state.italic = false,
                TagEnd::Strong => style_state.bold = false,
                TagEnd::Strikethrough => style_state.strikethrough = false,
                TagEnd::Link => style_state.link = false,
                TagEnd::CodeBlock => {
                    for code_line in &code_block_content {
                        let mut spans =
                            vec![Span::styled("  ", Style::default().bg(theme.code_bg))];
                        spans.extend(highlight_code(code_line, &code_lang, theme, syn));
                        let current_len: usize =
                            spans.iter().map(|s| s.content.chars().count()).sum();
                        if current_len < width.saturating_sub(2) {
                            spans.push(Span::styled(
                                " ".repeat(width.saturating_sub(current_len + 2)),
                                Style::default().bg(theme.code_bg),
                            ));
                        }
                        let line = Line::from(spans);
                        lines.push(if in_blockquote {
                            style_blockquote_line(line, theme)
                        } else {
                            line
                        });
                    }
                    code_block_content.clear();
                    code_lang.clear();
                    in_code_block = false;
                    lines.push(Line::from(""));
                }
                TagEnd::Table => {
                    render_table(&table_rows, theme, width, &mut lines);
                    table_rows.clear();
                    in_table = false;
                    lines.push(Line::from(""));
                }
                _ => {}
            },
            Event::Text(text) => {
                if in_code_block {
                    code_block_content.extend(text.lines().map(|l| l.to_string()));
                } else if in_table {
                    if let Some(row) = table_rows.last_mut() {
                        if row.is_empty() {
                            row.push(text.to_string());
                        } else {
                            row.last_mut().unwrap().push_str(&text);
                        }
                    }
                } else if in_list_item || in_paragraph {
                    push_text(&text, style_state, &mut current_segment);
                }
            }
            Event::Code(code) => {
                let style = Style::default().fg(theme.code).bg(theme.code_bg);
                current_segment.push(Span::styled(code.to_string(), style));
            }
            Event::SoftBreak => {
                push_text(" ", style_state, &mut current_segment);
            }
            Event::HardBreak => {
                if in_list_item {
                    list_item_segments.push(std::mem::take(&mut current_segment));
                } else if in_paragraph {
                    paragraph_segments.push(std::mem::take(&mut current_segment));
                }
            }
            Event::Rule => {
                let hr_width = width.saturating_sub(4);
                lines.push(Line::from(Span::styled(
                    "─".repeat(hr_width),
                    Style::default().fg(theme.hr),
                )));
                lines.push(Line::from(""));
            }
            Event::Html(_) => {}
            Event::InlineHtml(_) => {}
            Event::FootnoteReference(_) => {}
            Event::InlineMath(_) => {}
            Event::DisplayMath(_) => {}
            Event::TaskListMarker(checked) => {
                let checkbox = if checked { "☑ " } else { "☐ " };
                let mut span =
                    Span::styled(checkbox.to_string(), Style::default().fg(theme.checkbox));
                if checked {
                    span.style = span.style.add_modifier(Modifier::CROSSED_OUT);
                }
                current_segment.push(span);
            }
        }
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            text.to_string(),
            Style::default().fg(theme.text),
        )));
    }

    lines
}

fn style_heading_line(line: Line<'static>, level: HeadingLevel, theme: MarkdownTheme) -> Line<'static> {
    let style = match level {
        HeadingLevel::H1 => Style::default().fg(theme.heading).bold().underlined(),
        HeadingLevel::H2 => Style::default().fg(theme.heading).bold().underlined(),
        HeadingLevel::H3 => Style::default().fg(theme.heading2).bold(),
        HeadingLevel::H4 => Style::default().fg(theme.heading2).bold(),
        _ => Style::default().fg(theme.heading3),
    };

    let spans = line
        .spans
        .into_iter()
        .map(|s| Span::styled(s.content.to_string(), style))
        .collect::<Vec<_>>();

    Line::from(spans)
}

fn style_blockquote_line(mut line: Line<'static>, theme: MarkdownTheme) -> Line<'static> {
    let mut spans = vec![Span::styled(
        "┃ ",
        Style::default().fg(theme.blockquote_bar),
    )];
    spans.extend(line.spans.into_iter().map(|s| {
        let style = s.style.fg(theme.blockquote).italic();
        Span::styled(s.content.to_string(), style)
    }));
    line.spans = spans;
    line
}

fn render_table(
    rows: &[Vec<String>],
    theme: MarkdownTheme,
    width: usize,
    lines: &mut Vec<Line<'static>>,
) {
    if rows.is_empty() {
        return;
    }

    // Calculate column widths
    let num_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut col_widths: Vec<usize> = vec![0; num_cols];

    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < num_cols {
                col_widths[i] = col_widths[i].max(cell.chars().count());
            }
        }
    }

    // Limit total width
    let total_width: usize = col_widths.iter().sum::<usize>() + num_cols * 3 + 1;
    if total_width > width.saturating_sub(4) {
        let scale = (width.saturating_sub(4)) as f64 / total_width as f64;
        for w in &mut col_widths {
            *w = ((*w as f64) * scale).max(3.0) as usize;
        }
    }

    // Top border
    let mut top = String::from("┌");
    for (i, w) in col_widths.iter().enumerate() {
        top.push_str(&"─".repeat(*w + 2));
        if i < num_cols - 1 {
            top.push('┬');
        }
    }
    top.push('┐');
    lines.push(Line::from(Span::styled(
        top,
        Style::default().fg(theme.table_border),
    )));

    // Rows
    for (row_idx, row) in rows.iter().enumerate() {
        let mut spans = vec![Span::styled("│", Style::default().fg(theme.table_border))];

        for (i, w) in col_widths.iter().enumerate() {
            let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
            let cell_display: String = cell.chars().take(*w).collect();
            let padding = w.saturating_sub(cell_display.chars().count());

            spans.push(Span::raw(" "));
            if row_idx == 0 {
                // Header row
                spans.push(Span::styled(
                    cell_display,
                    Style::default().fg(theme.table_header).bold(),
                ));
            } else {
                spans.push(Span::styled(cell_display, Style::default().fg(theme.text)));
            }
            spans.push(Span::raw(" ".repeat(padding + 1)));
            spans.push(Span::styled("│", Style::default().fg(theme.table_border)));
        }

        lines.push(Line::from(spans));

        // Header separator
        if row_idx == 0 && rows.len() > 1 {
            let mut sep = String::from("├");
            for (i, w) in col_widths.iter().enumerate() {
                sep.push_str(&"─".repeat(*w + 2));
                if i < num_cols - 1 {
                    sep.push('┼');
                }
            }
            sep.push('┤');
            lines.push(Line::from(Span::styled(
                sep,
                Style::default().fg(theme.table_border),
            )));
        }
    }

    // Bottom border
    let mut bottom = String::from("└");
    for (i, w) in col_widths.iter().enumerate() {
        bottom.push_str(&"─".repeat(*w + 2));
        if i < num_cols - 1 {
            bottom.push('┴');
        }
    }
    bottom.push('┘');
    lines.push(Line::from(Span::styled(
        bottom,
        Style::default().fg(theme.table_border),
    )));
}

fn wrap_spans(
    spans: Vec<Span<'static>>,
    prefix: Option<&str>,
    theme: MarkdownTheme,
    width: usize,
) -> Vec<Line<'static>> {
    if width == 0 {
        return vec![Line::from(spans)];
    }

    let mut lines = Vec::new();
    let mut current_line: Vec<Span<'static>> = Vec::new();
    let mut current_width = 0;
    let prefix_width = prefix.as_ref().map(|p| p.chars().count()).unwrap_or(0);
    let effective_width = width.saturating_sub(prefix_width + 2);

    if let Some(p) = prefix {
        current_line.push(Span::styled(
            p.to_string(),
            Style::default().fg(theme.list_marker),
        ));
        current_width = prefix_width;
    }

    for span in spans {
        let text = span.content.to_string();
        let style = span.style;

        for word in text.split_inclusive(' ') {
            let word_width = word.chars().count();

            if current_width + word_width > effective_width && current_width > prefix_width {
                lines.push(Line::from(std::mem::take(&mut current_line)));
                let indent = " ".repeat(prefix_width);
                current_line.push(Span::styled(indent, Style::default()));
                current_width = prefix_width;
            }

            current_line.push(Span::styled(word.to_string(), style));
            current_width += word_width;
        }
    }

    if !current_line.is_empty() {
        lines.push(Line::from(current_line));
    }

    if lines.is_empty() {
        lines.push(Line::from(""));
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_render_respects_heading_color() {
        let md = MarkdownTheme {
            text: Color::Reset,
            bold: Color::Blue,
            italic: Color::DarkGray,
            code: Color::Green,
            code_bg: Color::Reset,
            heading: Color::Red,
            heading2: Color::Yellow,
            heading3: Color::Magenta,
            list_marker: Color::Cyan,
            link: Color::Blue,
            blockquote: Color::DarkGray,
            blockquote_bar: Color::DarkGray,
            hr: Color::DarkGray,
            table_border: Color::DarkGray,
            table_header: Color::Red,
            strikethrough: Color::DarkGray,
            checkbox: Color::Green,
        };
        let syn = SyntaxTheme {
            keyword: Color::Magenta,
            ty: Color::Cyan,
            string: Color::Green,
            number: Color::Yellow,
            comment: Color::DarkGray,
            punct: Color::Reset,
            ident: Color::Reset,
        };

        let lines = render("# Title", md, syn, 80);
        // First non-empty line should be the heading.
        let first = lines.iter().find(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
                .trim()
                .len()
                > 0
        });
        let Some(first) = first else {
            panic!("no lines rendered");
        };
        assert!(
            first.spans.iter().any(|s| s.style.fg == Some(Color::Red)),
            "expected heading fg Color::Red"
        );
    }
}
