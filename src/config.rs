use std::path::PathBuf;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AppConfig {
    pub general: Option<GeneralConfig>,
    pub sandbox: Option<SandboxConfig>,
    pub theme: Option<ThemeConfig>,
    pub memory: Option<MemoryConfig>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct GeneralConfig {
    pub model: Option<String>,
    pub split_ratio: Option<u16>,
    pub auto_index: Option<bool>,
    pub resume_last: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SandboxConfig {
    pub enabled: Option<bool>,
    pub root: Option<PathBuf>,
    pub allow_paths: Option<Vec<PathBuf>>,
    pub deny_paths: Option<Vec<PathBuf>>,
    pub allow_commands: Option<Vec<String>>,
    pub allow_tools: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ThemeConfig {
    pub file_categories: Option<HashMap<String, String>>,
    pub file_extensions: Option<HashMap<String, String>>,
}


#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MemoryConfig {
    pub enabled: Option<bool>,
    /// Inject recalled memories as an ephemeral system message each turn.
    pub auto_inject: Option<bool>,
    /// Learn from tool failures automatically.
    pub auto_learn_failures: Option<bool>,
    /// Learn from high-signal user preferences/corrections automatically.
    pub auto_learn_user: Option<bool>,
    /// Use an LLM to extract durable memories at the end of a turn.
    pub auto_extract: Option<bool>,
    /// Optional model override for extraction.
    pub extraction_model: Option<String>,
}

impl AppConfig {
    pub fn load() -> Self {
        let Some(path) = default_config_path() else {
            return Self::default();
        };

        let Ok(contents) = std::fs::read_to_string(&path) else {
            return Self::default();
        };

        toml::from_str(&contents).unwrap_or_default()
    }

    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = default_config_path() else {
            return Ok(());
        };

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        std::fs::write(path, contents)
    }
}

fn default_config_path() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".lorikeet").join("config.toml"))
}
