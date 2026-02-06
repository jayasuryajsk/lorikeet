use std::path::{Path, PathBuf};

use serde_json::Value;

#[derive(Debug, Clone)]
pub struct VerifySuggestion {
    pub label: String,
    pub command: String,
    pub confidence: f32,
}

pub fn detect_suggestions(root: &Path) -> Vec<VerifySuggestion> {
    let mut out: Vec<VerifySuggestion> = Vec::new();

    // Rust
    if root.join("Cargo.toml").exists() {
        out.push(s("Run tests", "cargo test", 0.95));
        out.push(s("Format check", "cargo fmt --all -- --check", 0.8));
        out.push(s(
            "Clippy",
            "cargo clippy --all-targets --all-features -D warnings",
            0.7,
        ));
        return out;
    }

    // Node
    if root.join("package.json").exists() {
        return detect_node_suggestions(root);
    }

    // Python
    if root.join("pyproject.toml").exists() || root.join("requirements.txt").exists() {
        out.push(s("Run tests", "pytest", 0.7));
        out.push(s("Lint", "ruff check .", 0.55));
        out.push(s("Format check", "ruff format --check .", 0.45));
        return out;
    }

    // Go
    if root.join("go.mod").exists() {
        out.push(s("Run tests", "go test ./...", 0.9));
        out.push(s("Format", "gofmt -w .", 0.4));
        return out;
    }

    out
}

fn detect_node_suggestions(root: &Path) -> Vec<VerifySuggestion> {
    let pm = node_package_manager(root);
    let scripts = read_package_json_scripts(&root.join("package.json"));

    // Prefer scripts that actually exist.
    let mut out: Vec<VerifySuggestion> = Vec::new();

    // test
    if scripts.contains_key("test") {
        out.push(s("Run tests", &format!("{} test", pm), 0.9));
    } else {
        // still useful: some repos rely on default npm test behavior or CI-only tests
        out.push(s("Run tests (maybe)", &format!("{} test", pm), 0.5));
    }

    // typecheck / lint / build
    out.extend(node_script_suggestions(&pm, &scripts));

    // If none of the core scripts exist, offer "install" as a first step.
    if !scripts.contains_key("test")
        && !scripts.contains_key("lint")
        && !scripts.contains_key("typecheck")
        && !scripts.contains_key("build")
    {
        out.push(s("Install deps (maybe)", &format!("{} install", pm), 0.35));
    }

    // Sort by confidence descending, keep at most 6.
    out.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(6);
    out
}

fn node_script_suggestions(
    pm: &str,
    scripts: &std::collections::HashMap<String, String>,
) -> Vec<VerifySuggestion> {
    let mut out = Vec::new();

    if scripts.contains_key("typecheck") {
        out.push(s("Typecheck", &format!("{} run typecheck", pm), 0.85));
    } else if scripts.contains_key("tsc") {
        out.push(s("Typecheck", &format!("{} run tsc", pm), 0.65));
    }

    if scripts.contains_key("lint") {
        out.push(s("Lint", &format!("{} run lint", pm), 0.85));
    }

    if scripts.contains_key("build") {
        out.push(s("Build", &format!("{} run build", pm), 0.75));
    }

    // common alternates
    if scripts.contains_key("check") {
        out.push(s("Check", &format!("{} run check", pm), 0.65));
    }

    if scripts.contains_key("ci") {
        out.push(s("CI", &format!("{} run ci", pm), 0.6));
    }

    out
}

fn read_package_json_scripts(path: &Path) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        return out;
    };
    let Some(scripts) = v.get("scripts") else {
        return out;
    };
    let Some(obj) = scripts.as_object() else {
        return out;
    };
    for (k, val) in obj {
        if let Some(cmd) = val.as_str() {
            out.insert(k.to_string(), cmd.to_string());
        }
    }
    out
}

fn s(label: &str, command: &str, confidence: f32) -> VerifySuggestion {
    VerifySuggestion {
        label: label.to_string(),
        command: command.to_string(),
        confidence,
    }
}

fn node_package_manager(root: &Path) -> String {
    if root.join("pnpm-lock.yaml").exists() {
        return "pnpm".to_string();
    }
    if root.join("yarn.lock").exists() {
        return "yarn".to_string();
    }
    if root.join("bun.lockb").exists() {
        return "bun".to_string();
    }
    "npm".to_string()
}

#[allow(dead_code)]
pub fn workspace_root_from_cwd() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}
