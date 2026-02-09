use std::io;
use std::path::PathBuf;
use std::time::Duration;

use color_eyre::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::CrosstermBackend;
use ratatui::Terminal;
use tokio::sync::mpsc;

mod app;
mod checkpoints;
mod codex_oauth;
mod config;
mod events;
mod llm;
mod lsp;
mod markdown;
mod memory;
mod sandbox;
mod semantic_search;
mod session;
mod theme;
mod tools;
mod types;
mod ui;
mod verify;

use app::App;
use config::AppConfig;
use events::AppEvent;
use memory::MemoryManager;
use sandbox::SandboxPolicy;
use semantic_search::{index_dir_for_workspace, SearchConfig, SemanticSearch};
use tools::TOOL_NAMES;
use ui::ui;

use llm::LlmProvider;

async fn load_llm_credentials() -> Result<(LlmProvider, String), String> {
    // Try project-local .env first.
    let _ = dotenvy::dotenv();

    // Then try user-level ~/.lorikeet/.env for global installs.
    if let Some(home) = dirs::home_dir() {
        let user_env = home.join(".lorikeet").join(".env");
        if user_env.exists() {
            let _ = dotenvy::from_path(user_env);
        }
    }

    let preferred = std::env::var("LORIKEET_PROVIDER")
        .ok()
        .map(|s| s.to_lowercase());

    let try_openrouter = || -> Option<(LlmProvider, String)> {
        std::env::var("OPENROUTER_API_KEY")
            .ok()
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
            .map(|k| (LlmProvider::OpenRouter, k))
    };

    let try_openai = || -> Option<(LlmProvider, String)> {
        std::env::var("OPENAI_API_KEY")
            .ok()
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
            .map(|k| (LlmProvider::OpenAI, k))
    };

    let try_codex = || async {
        let k = codex_oauth::openai_api_key_from_codex_oauth().await?;
        if k.trim().is_empty() {
            return Err("Codex OAuth present but returned an empty API key".to_string());
        }
        Ok((LlmProvider::OpenAI, k))
    };

    match preferred.as_deref() {
        Some("openrouter") => {
            if let Some(v) = try_openrouter() {
                return Ok(v);
            }
            return Err("LORIKEET_PROVIDER=openrouter but OPENROUTER_API_KEY is not set".into());
        }
        Some("openai") => {
            if let Some(v) = try_openai() {
                return Ok(v);
            }
            return Err("LORIKEET_PROVIDER=openai but OPENAI_API_KEY is not set".into());
        }
        Some("codex") | Some("codex_oauth") => return try_codex().await,
        _ => {}
    }

    if let Some(v) = try_openrouter() {
        return Ok(v);
    }
    if let Some(v) = try_openai() {
        return Ok(v);
    }
    match try_codex().await {
        Ok(v) => Ok(v),
        Err(e) => Err(format!(
            "No API key found.\n\nSet OPENROUTER_API_KEY or OPENAI_API_KEY, or sign in via Codex CLI.\n\nOptional: set LORIKEET_PROVIDER=openrouter|openai|codex\n\nDetails: {}",
            e
        )),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    // Check for CLI subcommands
    let args: Vec<String> = std::env::args().collect();
    let mut resume_override: Option<bool> = None;
    if args.len() > 1 {
        match args[1].as_str() {
            "continue" => {
                // Start the TUI and resume the latest session for this workspace (if any).
                resume_override = Some(true);
            }
            "index" => {
                return run_index_command(&args[2..]).await;
            }
            "help" | "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {
                eprintln!("Unknown command: {}", args[1]);
                print_help();
                std::process::exit(1);
            }
        }
    }
    let (provider, api_key) = match load_llm_credentials().await {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("{}", msg);
            eprintln!();
            eprintln!("Examples:");
            eprintln!("  export OPENROUTER_API_KEY=... ");
            eprintln!("  echo 'OPENROUTER_API_KEY=...' > ~/.lorikeet/.env");
            eprintln!();
            eprintln!("Codex OAuth:");
            eprintln!("  codex login   # then run lorikeet (will reuse ~/.codex/auth.json)");
            std::process::exit(1);
        }
    };

    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    io::stdout().execute(EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AppEvent>();

    let input_tx = event_tx.clone();
    tokio::spawn(async move {
        loop {
            if event::poll(Duration::from_millis(50)).unwrap_or(false) {
                match event::read() {
                    Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                        let _ = input_tx.send(AppEvent::Input(key));
                    }
                    Ok(Event::Mouse(mouse)) => {
                        let _ = input_tx.send(AppEvent::Mouse(mouse));
                    }
                    _ => {}
                }
            }
        }
    });

    let workspace_root = std::env::current_dir()?;
    let config = AppConfig::load();
    let sandbox_policy = std::sync::Arc::new(SandboxPolicy::from_config(
        config.clone(),
        workspace_root.clone(),
        TOOL_NAMES,
    ));

    let memory = std::sync::Arc::new(
        MemoryManager::init(&workspace_root)
            .await
            .map_err(|e| color_eyre::eyre::eyre!(e.to_string()))?,
    );

    let mut app = App::new(
        event_tx,
        provider,
        api_key,
        sandbox_policy,
        config.clone(),
        workspace_root.clone(),
        memory,
    );

    let resume = resume_override.unwrap_or_else(|| {
        config
            .general
            .as_ref()
            .and_then(|g| g.resume_last)
            .unwrap_or(false)
    });
    app.init_session(resume);

    if config
        .general
        .as_ref()
        .and_then(|g| g.auto_index)
        .unwrap_or(true)
        && !index_file_exists(&workspace_root)
    {
        // Start background indexing for semantic search
        app.start_background_indexing();
    }

    loop {
        terminal.draw(|frame| ui(frame, &mut app))?;

        match tokio::time::timeout(Duration::from_millis(16), event_rx.recv()).await {
            Ok(Some(event)) => app.handle_event(event),
            Ok(None) => break,
            Err(_) => {}
        }

        if app.should_quit {
            break;
        }
    }

    disable_raw_mode()?;
    io::stdout().execute(DisableMouseCapture)?;
    io::stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}

fn index_file_exists(workspace_root: &std::path::Path) -> bool {
    let index_dir = index_dir_for_workspace(workspace_root);
    let index_path = index_dir.join("index.bin");
    std::fs::metadata(index_path)
        .map(|m| m.len() > 0)
        .unwrap_or(false)
}

fn print_help() {
    println!("Lorikeet - An autonomous coding agent");
    println!();
    println!("USAGE:");
    println!("    lorikeet              Start a new interactive TUI session");
    println!("    lorikeet continue     Resume the latest session for this workspace");
    println!("    lorikeet index [DIR]  Index a directory for semantic search");
    println!("    lorikeet help         Show this help message");
    println!();
    println!("ENVIRONMENT:");
    println!(
        "    OPENROUTER_API_KEY    API key for OpenRouter (preferred)
    OPENAI_API_KEY         Fallback env var (if set)

NOTES:
    If installed globally, you can also store OPENROUTER_API_KEY in ~/.lorikeet/.env"
    );
}

async fn run_index_command(args: &[String]) -> Result<()> {
    let dir = if args.is_empty() {
        std::env::current_dir()?
    } else {
        PathBuf::from(&args[0])
    };

    if !dir.exists() {
        eprintln!("Error: Directory does not exist: {}", dir.display());
        std::process::exit(1);
    }

    if !dir.is_dir() {
        eprintln!("Error: Not a directory: {}", dir.display());
        std::process::exit(1);
    }

    let config = AppConfig::load();
    let sandbox_policy = SandboxPolicy::from_config(config, std::env::current_dir()?, TOOL_NAMES);

    let checked_dir = sandbox_policy
        .check_path_allowed(&dir)
        .map_err(|e| color_eyre::eyre::eyre!(e.to_string()))?;

    println!("Indexing {}...", checked_dir.display());
    println!("(This will download the embedding model on first run, ~22MB)");
    println!();

    // Create semantic search and index (workspace-specific index dir)
    let cfg = SearchConfig::for_workspace(&checked_dir);
    let search = SemanticSearch::new(cfg)
        .map_err(|e| color_eyre::eyre::eyre!("Failed to initialize semantic search: {}", e))?;

    match search.index_directory(&checked_dir) {
        Ok(stats) => {
            println!("Indexing complete!");
            println!();
            println!("Statistics:");
            println!("  Chunks indexed: {}", stats.total_chunks);
            println!("  Files indexed:  {}", stats.total_files);
            println!("  Index size:     {} bytes", stats.index_size_bytes);
            println!();
            println!("Languages:");
            for (lang, count) in &stats.languages {
                println!("  {:?}: {} chunks", lang, count);
            }
            println!();
            println!("Index stored at: ~/.lorikeet/index/");
        }
        Err(e) => {
            eprintln!("Error indexing: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

#[cfg(test)]
mod verify_tests;
