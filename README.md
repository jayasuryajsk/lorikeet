# Lorikeet

Lorikeet is a fast, terminal-native coding agent built in Rust (Ratatui). It can read/search/edit your codebase, run safe tool calls, and keep project-scoped memory.

Status: **alpha** (expect rough edges).

## Highlights

- **Inline Tool Trace (audit-first):** see what the model asked, what was executed (cwd + sandbox allow/deny), and streaming output.
- **Plan mode (tool-gated):** toggle PLAN with `TAB` to force “plan-only” responses; then execute via the plan modal.
- **Semantic search + `rg` + smart search:** fast exact search and embeddings-backed search (workspace-indexed).
- **Policy-only sandbox:** path allow/deny + command allowlist (no container/VM; blocks obvious footguns).
- **Project memory:** stored under `<repo>/.lorikeet/memory/` (SQLite); learns preferences and tool-failure “mistakes”.
- **Checkpoints + restore:** snapshot/rewind files + session timeline.

## Install

### From source (recommended for now)

```bash
git clone https://github.com/jayasuryajsk/lorikeet.git
cd lorikeet
cargo install --path . --force
```

### One-shot install script

```bash
./install.sh
```

## API Key Setup

Lorikeet reads keys in this order:

1. Project-local `.env` (via `dotenvy`)
2. `~/.lorikeet/.env` (recommended for global installs)
3. Shell env vars
4. **Codex OAuth cache** (`~/.codex/auth.json`) if present (derives an OpenAI API key via token exchange)

Create `~/.lorikeet/.env`:

```bash
mkdir -p ~/.lorikeet
echo 'OPENROUTER_API_KEY=...' > ~/.lorikeet/.env
```

Env vars:

- `OPENROUTER_API_KEY` (preferred)
- `OPENAI_API_KEY` (fallback)
- `LORIKEET_PROVIDER=openrouter|openai|codex` (optional override)

### Codex OAuth (ChatGPT login)

If you’ve logged in to the official Codex CLI using “sign in with ChatGPT”, Lorikeet can reuse that login to run with your Codex/ChatGPT subscription:

```bash
codex login
LORIKEET_PROVIDER=codex lorikeet
```

## Usage

Start Lorikeet in a repo:

```bash
cd /path/to/repo
lorikeet
```

Resume the last session for this workspace:

```bash
lorikeet continue
```

Index for semantic search (first run downloads the embedding model, ~22MB):

```bash
lorikeet index .
```

## Keybinds

- `ESC` quit
- `ENTER` send
- `TAB` toggle Plan mode (PLAN)
- `Shift+TAB` switch pane (Chat ↔ Context)
- `PgUp/PgDn` scroll chat

## Slash Commands (local, not sent to the model)

- `/settings` (alias: `/s`) open settings
- `/themes` (alias: `/t`) theme picker
- `/verify` run the suggested verify command for the workspace (or provide one)
- `/plan` enable Plan mode
- `/auto` disable Plan mode
- `/go` execute once with tools enabled (mostly superseded by the Plan modal)
- `/checkpoint [name...]` create a checkpoint
- `/checkpoints` list checkpoints
- `/checkpoint-diff <id|latest>` show checkpoint diff summary
- `/restore <id|latest>` restore checkpoint + rewind session
- `/new` start a new session
- `/resume` resume latest session for this workspace
- `/sessions` show the sessions directory
- `/help` show commands

## Configuration

Config file: `~/.lorikeet/config.toml`

Example:

```toml
[general]
model = "openai/gpt-5.2"
auto_index = true
resume_last = false

[sandbox]
enabled = true
allow_commands = ["rg","ls","cat","pwd","sed","awk","find","wc","head","tail","git"]
```

Notes:

- Sandbox is **policy-only** (no OS/container isolation). It’s meant to prevent accidental access to `~/.ssh`, `/etc`, etc.
- Semantic search indexes are cached per-workspace under `~/.lorikeet/index/<project_id>/`.

## Storage Layout

- Sessions: `~/.lorikeet/sessions/<project_id>/*.jsonl`
- Semantic index: `~/.lorikeet/index/<project_id>/`
- Checkpoints: `~/.lorikeet/checkpoints/<project_id>/<checkpoint_id>/`
- Project memory DB: `<repo>/.lorikeet/memory/memories.db`

## Dependencies / System Requirements

- Rust toolchain (stable)
- `rg` (ripgrep) available on `PATH` for exact search
- Optional (for LSP tool):
  - `rust-analyzer`
  - `typescript-language-server` (and `typescript`)

## Contributing

PRs welcome. If you’re changing agent behavior, include:

- a short acceptance checklist
- at least one unit test when it’s not purely UI

## License

No license file yet. Add one before promoting beyond alpha.
