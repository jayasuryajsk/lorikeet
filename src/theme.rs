use std::collections::HashSet;

use ratatui::style::{Color, Modifier, Style};

use crate::config::AppConfig;

pub fn style_for_filename(name: &str, config: &AppConfig) -> Style {
    if name.ends_with('/') {
        return style_for_category("directory", config);
    }
    if name.starts_with('.') {
        return style_for_category("hidden", config);
    }

    let ext = name.rsplit('.').next().unwrap_or("");
    if ext != name {
        if let Some(style) = style_from_extension(ext, config) {
            return style;
        }
        if let Some(category) = default_extension_category(ext) {
            return style_for_category(category, config);
        }
    }

    style_for_category("default", config)
}

fn style_from_extension(ext: &str, config: &AppConfig) -> Option<Style> {
    let theme = config.theme.as_ref()?;
    let map = theme.file_extensions.as_ref()?;
    let value = map.get(ext)?;

    if let Some(style) = category_style(value, config) {
        return Some(style);
    }
    parse_style_spec(value)
}

fn category_style(category: &str, config: &AppConfig) -> Option<Style> {
    let cat = normalize_category(category);
    let mut known = default_category_keys();
    if let Some(theme) = &config.theme {
        if let Some(overrides) = &theme.file_categories {
            for key in overrides.keys() {
                known.insert(normalize_category(key));
            }
        }
    }
    if known.contains(&cat) {
        Some(style_for_category(&cat, config))
    } else {
        None
    }
}

fn style_for_category(category: &str, config: &AppConfig) -> Style {
    let cat = normalize_category(category);
    if let Some(theme) = &config.theme {
        if let Some(overrides) = &theme.file_categories {
            if let Some(spec) = overrides.get(&cat) {
                if let Some(style) = parse_style_spec(spec) {
                    return style;
                }
            }
        }
    }

    default_category_style(&cat)
}

fn normalize_category(category: &str) -> String {
    match category {
        "dir" => "directory".to_string(),
        "doc" => "docs".to_string(),
        other => other.to_string(),
    }
}

fn default_extension_category(ext: &str) -> Option<&'static str> {
    match ext {
        "md" | "mdx" | "rst" | "txt" => Some("docs"),
        "toml" | "json" | "yaml" | "yml" | "ini" | "conf" | "env" | "lock" => Some("config"),
        "py" | "rb" | "sh" | "bash" | "zsh" | "fish" | "ps1" => Some("script"),
        "js" | "ts" | "jsx" | "tsx" | "html" | "css" => Some("web"),
        "csv" | "tsv" | "parquet" => Some("data"),
        "rs" | "go" | "c" | "h" | "cpp" | "hpp" | "java" | "kt" | "cs" | "swift" => Some("code"),
        _ => None,
    }
}

fn default_category_style(category: &str) -> Style {
    match category {
        "directory" => Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        "hidden" => Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        "docs" => Style::default().fg(Color::Green).add_modifier(Modifier::ITALIC),
        "config" => Style::default().fg(Color::Magenta),
        "script" => Style::default().fg(Color::Rgb(190, 242, 100)),
        "web" => Style::default().fg(Color::Yellow),
        "data" => Style::default().fg(Color::Rgb(125, 211, 252)),
        "code" => Style::default().fg(Color::Rgb(200, 200, 200)),
        _ => Style::default().fg(Color::Rgb(200, 200, 200)),
    }
}

fn default_category_keys() -> HashSet<String> {
    [
        "directory",
        "hidden",
        "docs",
        "config",
        "script",
        "web",
        "data",
        "code",
        "default",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn parse_style_spec(spec: &str) -> Option<Style> {
    let mut parts = spec.split_whitespace();
    let color_part = parts.next()?;
    let mut style = Style::default();

    if let Some(color) = parse_color(color_part) {
        style = style.fg(color);
    } else {
        return None;
    }

    for token in parts {
        style = match token {
            "bold" => style.add_modifier(Modifier::BOLD),
            "italic" => style.add_modifier(Modifier::ITALIC),
            "dim" => style.add_modifier(Modifier::DIM),
            "underline" => style.add_modifier(Modifier::UNDERLINED),
            _ => style,
        };
    }

    Some(style)
}

fn parse_color(token: &str) -> Option<Color> {
    let lower = token.to_lowercase();
    if lower.starts_with('#') && lower.len() == 7 {
        let r = u8::from_str_radix(&lower[1..3], 16).ok()?;
        let g = u8::from_str_radix(&lower[3..5], 16).ok()?;
        let b = u8::from_str_radix(&lower[5..7], 16).ok()?;
        return Some(Color::Rgb(r, g, b));
    }
    if let Some((r, g, b)) = parse_rgb_triplet(&lower) {
        return Some(Color::Rgb(r, g, b));
    }

    match lower.as_str() {
        "black" => Some(Color::Black),
        "red" => Some(Color::Red),
        "green" => Some(Color::Green),
        "yellow" => Some(Color::Yellow),
        "blue" => Some(Color::Blue),
        "magenta" => Some(Color::Magenta),
        "cyan" => Some(Color::Cyan),
        "gray" | "grey" => Some(Color::Gray),
        "darkgray" | "darkgrey" => Some(Color::DarkGray),
        "lightred" => Some(Color::LightRed),
        "lightgreen" => Some(Color::LightGreen),
        "lightyellow" => Some(Color::LightYellow),
        "lightblue" => Some(Color::LightBlue),
        "lightmagenta" => Some(Color::LightMagenta),
        "lightcyan" => Some(Color::LightCyan),
        "white" => Some(Color::White),
        _ => None,
    }
}

fn parse_rgb_triplet(token: &str) -> Option<(u8, u8, u8)> {
    let mut parts = token.split(',');
    let r = parts.next()?.trim().parse::<u8>().ok()?;
    let g = parts.next()?.trim().parse::<u8>().ok()?;
    let b = parts.next()?.trim().parse::<u8>().ok()?;
    Some((r, g, b))
}
