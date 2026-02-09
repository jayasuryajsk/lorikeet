use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ratatui::style::{Color, Modifier, Style};

use crate::config::AppConfig;

#[derive(Debug, Clone, Copy)]
pub struct UiPalette {
    pub fg: Color,
    pub fg_dim: Color,
    pub bg: Color,
    pub border: Color,
    pub accent: Color,
    pub ok: Color,
    pub warn: Color,
    pub err: Color,
}

impl UiPalette {
    pub fn selection(&self) -> Style {
        // REVERSED is terminal-theme-safe in both dark and light terminals.
        Style::default().add_modifier(Modifier::REVERSED)
    }

    pub fn ghost(&self) -> Style {
        // OpenCode "system" behavior: avoid fixed colors; let terminal defaults show through.
        // For fixed themes, fg_dim can still be set.
        let mut s = Style::default().add_modifier(Modifier::ITALIC | Modifier::DIM);
        if self.fg_dim != Color::Reset {
            s = s.fg(self.fg_dim);
        }
        s
    }

    pub fn meta(&self) -> Style {
        let mut s = Style::default().add_modifier(Modifier::DIM);
        if self.fg_dim != Color::Reset {
            s = s.fg(self.fg_dim);
        }
        s
    }

    pub fn border_style(&self) -> Style {
        let mut s = Style::default().add_modifier(Modifier::DIM);
        if self.border != Color::Reset {
            s = s.fg(self.border);
        }
        s
    }
}

pub fn ui_theme_name(config: &AppConfig) -> String {
    if let Ok(v) = std::env::var("LORIKEET_THEME") {
        return v.trim().to_string();
    }

    config
        .theme
        .as_ref()
        .and_then(|t| t.preset.clone())
        .unwrap_or_else(|| "system".to_string())
}

#[derive(Debug, Clone, Copy)]
pub struct MarkdownTheme {
    pub text: Color,
    pub bold: Color,
    pub italic: Color,
    pub code: Color,
    pub code_bg: Color,
    pub heading: Color,
    pub heading2: Color,
    pub heading3: Color,
    pub list_marker: Color,
    pub link: Color,
    pub blockquote: Color,
    pub blockquote_bar: Color,
    pub hr: Color,
    pub table_border: Color,
    pub table_header: Color,
    pub strikethrough: Color,
    pub checkbox: Color,
}

#[derive(Debug, Clone, Copy)]
pub struct SyntaxTheme {
    pub keyword: Color,
    pub ty: Color,
    pub string: Color,
    pub number: Color,
    pub comment: Color,
    pub punct: Color,
    pub ident: Color,
}

#[derive(Debug, Clone, Copy)]
pub struct ToolTraceTheme {
    pub title: Color,
    pub invocation: Color,
    pub details_key: Color,
    pub details_value: Color,
    pub out_prefix: Color,
    pub out_text: Color,
    pub sandbox_allow: Color,
    pub sandbox_deny: Color,
    pub call_id: Color,
    pub cwd: Color,
    pub duration: Color,
}

#[derive(Debug, Clone)]
pub struct FileTheme {
    pub categories: HashMap<String, String>,
    pub extensions: HashMap<String, String>,
}

impl Default for FileTheme {
    fn default() -> Self {
        Self {
            categories: HashMap::new(),
            extensions: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UiTheme {
    pub palette: UiPalette,
    pub markdown: MarkdownTheme,
    pub syntax: SyntaxTheme,
    pub tool_trace: ToolTraceTheme,
    pub files: FileTheme,
}

pub fn ui_theme(config: &AppConfig, workspace_root: Option<&Path>) -> UiTheme {
    let name = normalize_theme_name(&ui_theme_name(config));
    load_ui_theme(&name, workspace_root).unwrap_or_else(|| theme_from_palette(system_palette()))
}

pub fn ui_theme_by_name(name: &str, workspace_root: Option<&Path>) -> UiTheme {
    let name = normalize_theme_name(name);
    load_ui_theme(&name, workspace_root).unwrap_or_else(|| theme_from_palette(system_palette()))
}

pub fn user_markdown_theme(base: &UiTheme) -> MarkdownTheme {
    // Keep user messages distinct and easy to scan regardless of the selected theme.
    // This is intentionally conservative (ANSI colors) and terminal-theme-safe.
    let mut md = base.markdown;
    md.text = Color::Cyan;
    md.bold = Color::LightCyan;
    md.italic = Color::Cyan;
    md.heading = Color::LightCyan;
    md.heading2 = Color::Cyan;
    md.link = Color::Blue;
    md.list_marker = Color::DarkGray;
    md.checkbox = Color::Yellow;
    md
}

pub fn ui_palette(config: &AppConfig, workspace_root: Option<&Path>) -> UiPalette {
    ui_theme(config, workspace_root).palette
}

pub fn builtin_theme_tagline(name: &str) -> Option<&'static str> {
    match normalize_theme_name(name).as_str() {
        "system" => Some("terminal-default safe"),
        "opencode" => Some("cool gray + blue accent"),
        "tokyonight" => Some("soft neon blues"),
        "everforest" => Some("muted green"),
        "ayu" => Some("bright + modern"),
        "catppuccin" => Some("pastel"),
        "gruvbox" => Some("warm"),
        "kanagawa" => Some("inky"),
        "nord" => Some("cold"),
        "matrix" => Some("green phosphor"),
        "one-dark" => Some("classic"),
        _ => None,
    }
}

fn normalize_theme_name(name: &str) -> String {
    let n = name.trim().to_lowercase();
    match n.as_str() {
        "" => "system".to_string(),
        // Back-compat with earlier Lorikeet presets.
        "auto" | "light" => "system".to_string(),
        "dark" => "opencode".to_string(),
        other => other.to_string(),
    }
}

fn system_palette() -> UiPalette {
    UiPalette {
        fg: Color::Reset,
        // Use a real dim color even for system theme; many terminals ignore DIM on light themes.
        // DarkGray stays legible on both light and dark backgrounds when combined with modifiers.
        fg_dim: Color::DarkGray,
        bg: Color::Reset,
        border: Color::DarkGray,
        // ANSI color accents (respect terminal palette).
        accent: Color::Cyan,
        ok: Color::Green,
        warn: Color::Yellow,
        err: Color::Red,
    }
}

fn rgb(hex: &str) -> Color {
    let h = hex.trim().trim_start_matches('#');
    if h.len() != 6 {
        return Color::Reset;
    }
    let r = u8::from_str_radix(&h[0..2], 16).unwrap_or(0);
    let g = u8::from_str_radix(&h[2..4], 16).unwrap_or(0);
    let b = u8::from_str_radix(&h[4..6], 16).unwrap_or(0);
    Color::Rgb(r, g, b)
}

fn builtin_palettes() -> HashMap<&'static str, UiPalette> {
    // These are approximations of the named themes (truecolor where possible).
    // The important part is semantic consistency across the UI.
    HashMap::from([
        ("system", system_palette()),
        (
            "opencode",
            UiPalette {
                fg: Color::Reset,
                fg_dim: rgb("#6b7280"),
                bg: rgb("#0b1220"),
                border: rgb("#4b5563"),
                accent: rgb("#60a5fa"),
                ok: rgb("#22c55e"),
                warn: rgb("#f59e0b"),
                err: rgb("#ef4444"),
            },
        ),
        (
            "tokyonight",
            UiPalette {
                fg: Color::Reset,
                fg_dim: rgb("#565f89"),
                bg: rgb("#1a1b26"),
                border: rgb("#414868"),
                accent: rgb("#7aa2f7"),
                ok: rgb("#9ece6a"),
                warn: rgb("#e0af68"),
                err: rgb("#f7768e"),
            },
        ),
        (
            "everforest",
            UiPalette {
                fg: Color::Reset,
                fg_dim: rgb("#859289"),
                bg: rgb("#2b3339"),
                border: rgb("#4f585e"),
                accent: rgb("#a7c080"),
                ok: rgb("#a7c080"),
                warn: rgb("#dbbc7f"),
                err: rgb("#e67e80"),
            },
        ),
        (
            "ayu",
            UiPalette {
                fg: Color::Reset,
                fg_dim: rgb("#707a8c"),
                bg: rgb("#0f1419"),
                border: rgb("#3d424d"),
                accent: rgb("#39bae6"),
                ok: rgb("#b8cc52"),
                warn: rgb("#ffb454"),
                err: rgb("#ff3333"),
            },
        ),
        (
            "catppuccin",
            UiPalette {
                fg: Color::Reset,
                fg_dim: rgb("#6c7086"),
                bg: rgb("#1e1e2e"),
                border: rgb("#45475a"),
                accent: rgb("#89b4fa"),
                ok: rgb("#a6e3a1"),
                warn: rgb("#f9e2af"),
                err: rgb("#f38ba8"),
            },
        ),
        (
            "gruvbox",
            UiPalette {
                fg: Color::Reset,
                fg_dim: rgb("#928374"),
                bg: rgb("#282828"),
                border: rgb("#665c54"),
                accent: rgb("#83a598"),
                ok: rgb("#b8bb26"),
                warn: rgb("#fabd2f"),
                err: rgb("#fb4934"),
            },
        ),
        (
            "kanagawa",
            UiPalette {
                fg: Color::Reset,
                fg_dim: rgb("#727169"),
                bg: rgb("#1f1f28"),
                border: rgb("#54546d"),
                accent: rgb("#7e9cd8"),
                ok: rgb("#98bb6c"),
                warn: rgb("#e6c384"),
                err: rgb("#c34043"),
            },
        ),
        (
            "nord",
            UiPalette {
                fg: Color::Reset,
                fg_dim: rgb("#616e88"),
                bg: rgb("#2e3440"),
                border: rgb("#4c566a"),
                accent: rgb("#88c0d0"),
                ok: rgb("#a3be8c"),
                warn: rgb("#ebcb8b"),
                err: rgb("#bf616a"),
            },
        ),
        (
            "matrix",
            UiPalette {
                fg: Color::Reset,
                fg_dim: rgb("#00aa3f"),
                bg: rgb("#00110a"),
                border: rgb("#007a2c"),
                accent: rgb("#00ff5f"),
                ok: rgb("#00ff5f"),
                warn: rgb("#a3ff12"),
                err: rgb("#ff2a2a"),
            },
        ),
        (
            "one-dark",
            UiPalette {
                fg: Color::Reset,
                fg_dim: rgb("#5c6370"),
                bg: rgb("#282c34"),
                border: rgb("#4b5263"),
                accent: rgb("#61afef"),
                ok: rgb("#98c379"),
                warn: rgb("#e5c07b"),
                err: rgb("#e06c75"),
            },
        ),
    ])
}

fn theme_from_palette(p: UiPalette) -> UiTheme {
    // Stronger defaults than previous Lorikeet themeing: markdown + tool trace are themed
    // using semantic palette colors. Backgrounds default to terminal-default (Reset).
    let md = MarkdownTheme {
        text: p.fg,
        bold: p.accent,
        italic: p.fg_dim,
        code: p.accent,
        code_bg: Color::Reset,
        heading: p.accent,
        heading2: p.accent,
        heading3: p.fg_dim,
        list_marker: p.accent,
        link: p.accent,
        blockquote: p.fg_dim,
        blockquote_bar: p.border,
        hr: p.border,
        table_border: p.border,
        table_header: p.accent,
        strikethrough: p.fg_dim,
        checkbox: p.accent,
    };
    let syn = SyntaxTheme {
        keyword: p.accent,
        ty: Color::Cyan,
        string: Color::Green,
        number: p.warn,
        comment: p.fg_dim,
        punct: p.fg,
        ident: p.fg,
    };
    let tt = ToolTraceTheme {
        title: p.fg,
        invocation: p.fg,
        details_key: p.fg_dim,
        details_value: p.fg,
        out_prefix: p.fg_dim,
        out_text: p.fg,
        sandbox_allow: p.ok,
        sandbox_deny: p.err,
        call_id: p.fg_dim,
        cwd: p.fg_dim,
        duration: p.fg_dim,
    };
    UiTheme {
        palette: p,
        markdown: md,
        syntax: syn,
        tool_trace: tt,
        files: FileTheme::default(),
    }
}

fn builtin_themes() -> HashMap<&'static str, UiTheme> {
    builtin_palettes()
        .into_iter()
        .map(|(k, p)| (k, theme_from_palette(p)))
        .collect()
}

fn load_ui_theme(name: &str, workspace_root: Option<&Path>) -> Option<UiTheme> {
    let builtins = builtin_themes();
    if let Some(t) = builtins.get(name) {
        return Some(t.clone());
    }
    load_custom_theme(name, workspace_root).ok().flatten()
}

fn load_ui_palette(name: &str, workspace_root: Option<&Path>) -> Option<UiPalette> {
    let builtins = builtin_palettes();
    if let Some(p) = builtins.get(name) {
        return Some(*p);
    }
    load_custom_palette(name, workspace_root).ok().flatten()
}

fn load_custom_theme(name: &str, workspace_root: Option<&Path>) -> Result<Option<UiTheme>, ()> {
    for dir in ui_theme_paths(workspace_root) {
        let path = dir.join(format!("{}.json", name));
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        if let Some(t) = theme_from_json(&v) {
            return Ok(Some(t));
        }
        // Back-compat: accept minimal palette-only JSON.
        if let Some(p) = palette_from_json(&v) {
            return Ok(Some(theme_from_palette(p)));
        }
    }
    Ok(None)
}

fn load_custom_palette(name: &str, workspace_root: Option<&Path>) -> Result<Option<UiPalette>, ()> {
    for dir in ui_theme_paths(workspace_root) {
        let path = dir.join(format!("{}.json", name));
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        if let Some(p) = palette_from_json(&v) {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

fn palette_from_json(v: &serde_json::Value) -> Option<UiPalette> {
    // Minimal, maintainable subset:
    // - Accept {"palette": {...}} or {"theme": {...}}.
    // - Values can be "#RRGGBB", "none", or 0..255 indexed.
    let obj = v.as_object()?;
    let pal = obj
        .get("palette")
        .or_else(|| obj.get("theme"))
        .and_then(|x| x.as_object())?;

    let get = |k: &str| pal.get(k).and_then(parse_color_value);

    Some(UiPalette {
        fg: get("fg").unwrap_or(Color::Reset),
        fg_dim: get("fg_dim").unwrap_or(Color::Reset),
        bg: get("bg").unwrap_or(Color::Reset),
        border: get("border").unwrap_or(Color::Reset),
        accent: get("accent").unwrap_or(Color::Cyan),
        ok: get("ok").unwrap_or(Color::Green),
        warn: get("warn").unwrap_or(Color::Yellow),
        err: get("err").unwrap_or(Color::Red),
    })
}

fn theme_from_json(v: &serde_json::Value) -> Option<UiTheme> {
    let obj = v.as_object()?;

    // defs: {"primary": "#RRGGBB", ...}
    let defs = obj
        .get("defs")
        .and_then(|d| d.as_object())
        .cloned()
        .unwrap_or_default();

    // theme: {...} or back-compat minimal object
    let theme_obj = obj.get("theme").and_then(|t| t.as_object())?;

    // If this looks like a minimal palette, accept it.
    if theme_obj.contains_key("fg") || theme_obj.contains_key("accent") {
        let p = palette_from_json(&serde_json::Value::Object(obj.clone()))?;
        return Some(theme_from_palette(p));
    }

    let ui_section = theme_obj.get("ui").and_then(|t| t.as_object());
    let p = parse_ui_palette(ui_section, &defs).unwrap_or_else(system_palette);

    let base = theme_from_palette(p);

    let md = theme_obj
        .get("markdown")
        .and_then(|m| parse_markdown_theme(m, &defs, base.markdown));
    let syn = theme_obj
        .get("syntax")
        .and_then(|s| parse_syntax_theme(s, &defs, base.syntax));
    let tt = theme_obj
        .get("tool_trace")
        .and_then(|t| parse_tool_trace_theme(t, &defs, base.tool_trace));
    let files = theme_obj
        .get("files")
        .and_then(|f| parse_file_theme(f))
        .unwrap_or_else(|| base.files.clone());

    Some(UiTheme {
        palette: p,
        markdown: md.unwrap_or(base.markdown),
        syntax: syn.unwrap_or(base.syntax),
        tool_trace: tt.unwrap_or(base.tool_trace),
        files,
    })
}

fn resolve_color(
    v: &serde_json::Value,
    defs: &serde_json::Map<String, serde_json::Value>,
    visited: &mut HashSet<String>,
    depth: usize,
) -> Option<Color> {
    if depth > 16 {
        return None;
    }
    match v {
        serde_json::Value::String(s) => {
            let t = s.trim();
            if t.starts_with('$') {
                let key = t.trim_start_matches('$').to_string();
                if visited.contains(&key) {
                    return None;
                }
                let Some(next) = defs.get(&key) else {
                    return None;
                };
                visited.insert(key);
                return resolve_color(next, defs, visited, depth + 1);
            }
            parse_color_value(v)
        }
        _ => parse_color_value(v),
    }
}

fn parse_ui_palette(
    ui: Option<&serde_json::Map<String, serde_json::Value>>,
    defs: &serde_json::Map<String, serde_json::Value>,
) -> Option<UiPalette> {
    let ui = ui?;
    let get = |k: &str| {
        ui.get(k)
            .and_then(|v| resolve_color(v, defs, &mut HashSet::new(), 0))
    };

    Some(UiPalette {
        fg: get("fg").unwrap_or(Color::Reset),
        fg_dim: get("fg_dim").unwrap_or(Color::Reset),
        bg: get("bg").unwrap_or(Color::Reset),
        border: get("border").unwrap_or(Color::Reset),
        accent: get("accent").unwrap_or(Color::Cyan),
        ok: get("ok").unwrap_or(Color::Green),
        warn: get("warn").unwrap_or(Color::Yellow),
        err: get("err").unwrap_or(Color::Red),
    })
}

fn parse_markdown_theme(
    md: &serde_json::Value,
    defs: &serde_json::Map<String, serde_json::Value>,
    base: MarkdownTheme,
) -> Option<MarkdownTheme> {
    let md = md.as_object()?;
    let get = |k: &str| {
        md.get(k)
            .and_then(|v| resolve_color(v, defs, &mut HashSet::new(), 0))
    };
    Some(MarkdownTheme {
        text: get("text").unwrap_or(base.text),
        bold: get("bold").unwrap_or(base.bold),
        italic: get("italic").unwrap_or(base.italic),
        code: get("code").unwrap_or(base.code),
        code_bg: get("code_bg").unwrap_or(base.code_bg),
        heading: get("heading").unwrap_or(base.heading),
        heading2: get("heading2").unwrap_or(base.heading2),
        heading3: get("heading3").unwrap_or(base.heading3),
        list_marker: get("list_marker").unwrap_or(base.list_marker),
        link: get("link").unwrap_or(base.link),
        blockquote: get("blockquote").unwrap_or(base.blockquote),
        blockquote_bar: get("blockquote_bar").unwrap_or(base.blockquote_bar),
        hr: get("hr").unwrap_or(base.hr),
        table_border: get("table_border").unwrap_or(base.table_border),
        table_header: get("table_header").unwrap_or(base.table_header),
        strikethrough: get("strikethrough").unwrap_or(base.strikethrough),
        checkbox: get("checkbox").unwrap_or(base.checkbox),
    })
}

fn parse_syntax_theme(
    syn: &serde_json::Value,
    defs: &serde_json::Map<String, serde_json::Value>,
    base: SyntaxTheme,
) -> Option<SyntaxTheme> {
    let syn = syn.as_object()?;
    let get = |k: &str| {
        syn.get(k)
            .and_then(|v| resolve_color(v, defs, &mut HashSet::new(), 0))
    };
    Some(SyntaxTheme {
        keyword: get("keyword").unwrap_or(base.keyword),
        ty: get("type").or_else(|| get("ty")).unwrap_or(base.ty),
        string: get("string").unwrap_or(base.string),
        number: get("number").unwrap_or(base.number),
        comment: get("comment").unwrap_or(base.comment),
        punct: get("punct").unwrap_or(base.punct),
        ident: get("ident").unwrap_or(base.ident),
    })
}

fn parse_tool_trace_theme(
    tt: &serde_json::Value,
    defs: &serde_json::Map<String, serde_json::Value>,
    base: ToolTraceTheme,
) -> Option<ToolTraceTheme> {
    let tt = tt.as_object()?;
    let get = |k: &str| {
        tt.get(k)
            .and_then(|v| resolve_color(v, defs, &mut HashSet::new(), 0))
    };
    Some(ToolTraceTheme {
        title: get("title").unwrap_or(base.title),
        invocation: get("invocation").unwrap_or(base.invocation),
        details_key: get("details_key").unwrap_or(base.details_key),
        details_value: get("details_value").unwrap_or(base.details_value),
        out_prefix: get("out_prefix").unwrap_or(base.out_prefix),
        out_text: get("out_text").unwrap_or(base.out_text),
        sandbox_allow: get("sandbox_allow").unwrap_or(base.sandbox_allow),
        sandbox_deny: get("sandbox_deny").unwrap_or(base.sandbox_deny),
        call_id: get("call_id").unwrap_or(base.call_id),
        cwd: get("cwd").unwrap_or(base.cwd),
        duration: get("duration").unwrap_or(base.duration),
    })
}

fn parse_file_theme(v: &serde_json::Value) -> Option<FileTheme> {
    let obj = v.as_object()?;
    let mut out = FileTheme::default();
    if let Some(cats) = obj.get("categories").and_then(|c| c.as_object()) {
        for (k, v) in cats {
            if let Some(s) = v.as_str() {
                out.categories.insert(k.clone(), s.to_string());
            }
        }
    }
    if let Some(exts) = obj.get("extensions").and_then(|c| c.as_object()) {
        for (k, v) in exts {
            if let Some(s) = v.as_str() {
                out.extensions.insert(k.clone(), s.to_string());
            }
        }
    }
    Some(out)
}

fn parse_color_value(v: &serde_json::Value) -> Option<Color> {
    match v {
        serde_json::Value::Null => Some(Color::Reset),
        serde_json::Value::Number(n) => n.as_u64().and_then(|u| {
            if u <= 255 {
                Some(Color::Indexed(u as u8))
            } else {
                None
            }
        }),
        serde_json::Value::String(s) => {
            let t = s.trim();
            if t.is_empty() || t.eq_ignore_ascii_case("none") {
                return Some(Color::Reset);
            }
            if t.starts_with('#') && t.len() == 7 {
                return Some(rgb(t));
            }
            None
        }
        _ => None,
    }
}

pub fn ui_theme_paths(workspace_root: Option<&Path>) -> Vec<PathBuf> {
    let mut out = Vec::new();

    // Workspace-scoped themes: <root>/.lorikeet/themes/<name>.json
    if let Some(root) = workspace_root {
        out.push(root.join(".lorikeet").join("themes"));
    }

    // User-scoped themes: ~/.lorikeet/themes/<name>.json
    if let Some(home) = dirs::home_dir() {
        out.push(home.join(".lorikeet").join("themes"));
    }

    out
}

pub fn list_ui_themes(workspace_root: Option<&Path>) -> Vec<String> {
    let mut names = builtin_themes()
        .keys()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    for dir in ui_theme_paths(workspace_root) {
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for ent in rd.flatten() {
                let p = ent.path();
                if p.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }
                if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                    if !names.iter().any(|n| n == stem) {
                        names.push(stem.to_string());
                    }
                }
            }
        }
    }

    names.sort();
    names
}

pub fn style_for_filename(name: &str, config: &AppConfig) -> Style {
    if name.ends_with('/') {
        return style_for_category("directory", config, None);
    }
    if name.starts_with('.') {
        return style_for_category("hidden", config, None);
    }

    let ext = name.rsplit('.').next().unwrap_or("");
    if ext != name {
        if let Some(style) = style_from_extension(ext, config, None) {
            return style;
        }
        if let Some(category) = default_extension_category(ext) {
            return style_for_category(category, config, None);
        }
    }

    style_for_category("default", config, None)
}

pub fn style_for_filename_with_theme(name: &str, config: &AppConfig, theme: &UiTheme) -> Style {
    if name.ends_with('/') {
        return style_for_category("directory", config, Some(theme));
    }
    if name.starts_with('.') {
        return style_for_category("hidden", config, Some(theme));
    }

    let ext = name.rsplit('.').next().unwrap_or("");
    if ext != name {
        if let Some(style) = style_from_extension(ext, config, Some(theme)) {
            return style;
        }
        if let Some(category) = default_extension_category(ext) {
            return style_for_category(category, config, Some(theme));
        }
    }

    style_for_category("default", config, Some(theme))
}

fn style_from_extension(ext: &str, config: &AppConfig, theme: Option<&UiTheme>) -> Option<Style> {
    // Config overrides win.
    if let Some(cfg) = config
        .theme
        .as_ref()
        .and_then(|t| t.file_extensions.as_ref())
    {
        if let Some(value) = cfg.get(ext) {
            if let Some(style) = category_style(value, config, theme) {
                return Some(style);
            }
            return parse_style_spec(value);
        }
    }

    // Theme JSON/base theme.
    if let Some(theme) = theme {
        if let Some(value) = theme.files.extensions.get(ext) {
            if let Some(style) = category_style(value, config, Some(theme)) {
                return Some(style);
            }
            return parse_style_spec(value);
        }
    }

    None
}

fn category_style(category: &str, config: &AppConfig, theme: Option<&UiTheme>) -> Option<Style> {
    let cat = normalize_category(category);
    let mut known = default_category_keys();
    if let Some(theme) = &config.theme {
        if let Some(overrides) = &theme.file_categories {
            for key in overrides.keys() {
                known.insert(normalize_category(key));
            }
        }
    }
    if let Some(t) = theme {
        for key in t.files.categories.keys() {
            known.insert(normalize_category(key));
        }
    }
    if known.contains(&cat) {
        Some(style_for_category(&cat, config, theme))
    } else {
        None
    }
}

fn style_for_category(category: &str, config: &AppConfig, theme: Option<&UiTheme>) -> Style {
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

    if let Some(t) = theme {
        if let Some(spec) = t.files.categories.get(&cat) {
            if let Some(style) = parse_style_spec(spec) {
                return style;
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
        "directory" => Style::default()
            .fg(Color::Blue)
            .add_modifier(Modifier::BOLD),
        "hidden" => Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
        "docs" => Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::ITALIC),
        "config" => Style::default().fg(Color::Yellow),
        "script" => Style::default().fg(Color::Green),
        "web" => Style::default().fg(Color::Cyan),
        "data" => Style::default().fg(Color::Blue),
        // Use terminal default for code/default so it's readable in both dark + light themes.
        "code" => Style::default().fg(Color::Reset),
        _ => Style::default().fg(Color::Reset),
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_theme_defs_and_refs() {
        let v = json!({
            "defs": {
                "primary": "#112233",
                "dim": "$primary"
            },
            "theme": {
                "ui": {
                    "accent": "$primary",
                    "fg_dim": "$dim",
                    "border": "$primary"
                },
                "markdown": {
                    "heading": "$primary",
                    "italic": "$dim"
                },
                "syntax": {
                    "keyword": "$primary",
                    "comment": "$dim"
                },
                "tool_trace": {
                    "sandbox_allow": "$primary",
                    "sandbox_deny": "$primary"
                }
            }
        });

        let t = theme_from_json(&v).expect("theme parses");
        assert_eq!(t.palette.accent, rgb("#112233"));
        assert_eq!(t.palette.fg_dim, rgb("#112233"));
        assert_eq!(t.markdown.heading, rgb("#112233"));
        assert_eq!(t.syntax.keyword, rgb("#112233"));
        assert_eq!(t.tool_trace.sandbox_allow, rgb("#112233"));
    }

    #[test]
    fn none_and_null_inherit() {
        let v = json!({
            "theme": {
                "ui": { "fg": null, "accent": "none", "bg": "none" }
            }
        });
        let t = theme_from_json(&v).expect("theme parses");
        assert_eq!(t.palette.fg, Color::Reset);
        assert_eq!(t.palette.accent, Color::Reset);
        assert_eq!(t.palette.bg, Color::Reset);
    }

    #[test]
    fn back_compat_palette_json() {
        let v = json!({
            "palette": {
                "accent": "#010203",
                "fg_dim": "none"
            }
        });
        let p = palette_from_json(&v).expect("palette parses");
        assert_eq!(p.accent, rgb("#010203"));
        let t = theme_from_palette(p);
        assert_eq!(t.palette.accent, rgb("#010203"));
    }
}
