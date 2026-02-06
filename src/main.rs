use std::io;
use std::path::PathBuf;
use std::time::Duration;

use color_eyre::Result;
use crossterm::{
    event::{self, Event, KeyEventKind, EnableMouseCapture, DisableMouseCapture},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::prelude::CrosstermBackend;
use ratatui::Terminal;
use tokio::sync::mpsc;

mod app;
mod config;
mod events;
mod llm;
mod markdown;
mod memory;
mod semantic_search;
mod session;
mod sandbox;
mod theme;
mod tools;
mod verify;
mod types;
mod ui;

use app::App;
use config::AppConfig;
use events::AppEvent;
use semantic_search::{SearchConfig, SemanticSearch};
use memory::MemoryManager;
use sandbox::SandboxPolicy;
use tools::TOOL_NAMES;
use ui::ui;

fn load_api_key() -> Result<String, String> {
    // Try project-local .env first.
    let _ = dotenvy::dotenv();

    // Then try user-level ~/.lorikeet/.env for global installs.
    if let Some(home) = dirs::home_dir() {
        let user_env = home.join(".lorikeet").join(".env");
        if user_env.exists() {
            let _ = dotenvy::from_path(user_env);
        }
    }

    std::env::var("OPENROUTER_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .map_err(|_| {
            "OPENROUTER_API_KEY not set. Set it in your shell env, or create ~/.lorikeet/.env with OPENROUTER_API_KEY=...".to_string()
        })
}


#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    // Check for CLI subcommands
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
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
    let api_key = match load_api_key() {
        Ok(k) => k,
        Err(msg) => {
            eprintln!("{}", msg);
            eprintln!();
            eprintln!("Examples:");
            eprintln!("  export OPENROUTER_API_KEY=... ");
            eprintln!("  echo 'OPENROUTER_API_KEY=...' > ~/.lorikeet/.env");
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
        api_key,
        sandbox_policy,
        config.clone(),
        workspace_root,
        memory,
    );

    let resume = config
        .general
        .as_ref()
        .and_then(|g| g.resume_last)
        .unwrap_or(true);
    app.init_session(resume);

    if config
        .general
        .as_ref()
        .and_then(|g| g.auto_index)
        .unwrap_or(true)
        && !index_file_exists()
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

fn index_file_exists() -> bool {
    let index_dir = SearchConfig::default().index_dir;
    let index_path = index_dir.join("index.bin");
    std::fs::metadata(index_path).map(|m| m.len() > 0).unwrap_or(false)
}

fn print_help() {
    println!("Lorikeet - An autonomous coding agent");
    println!();
    println!("USAGE:");
    println!("    lorikeet              Start the interactive TUI");
    println!("    lorikeet index [DIR]  Index a directory for semantic search");
    println!("    lorikeet help         Show this help message");
    println!();
    println!("ENVIRONMENT:");
    println!("    OPENROUTER_API_KEY    API key for OpenRouter (preferred)
    OPENAI_API_KEY         Fallback env var (if set)

NOTES:
    If installed globally, you can also store OPENROUTER_API_KEY in ~/.lorikeet/.env");
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
    let sandbox_policy = SandboxPolicy::from_config(
        config,
        std::env::current_dir()?,
        TOOL_NAMES,
    );

    let checked_dir = sandbox_policy
        .check_path_allowed(&dir)
        .map_err(|e| color_eyre::eyre::eyre!(e.to_string()))?;

    println!("Indexing {}...", checked_dir.display());
    println!("(This will download the embedding model on first run, ~22MB)");
    println!();

    // Create semantic search and index
    let search = SemanticSearch::with_defaults()
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
