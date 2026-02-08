use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use flate2::read::GzDecoder;
use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::oneshot;

use crate::sandbox::SandboxPolicy;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LspLanguage {
    Rust,
    TypeScript,
}

impl LspLanguage {
    pub fn from_user(s: &str, path: &Path) -> Option<Self> {
        let t = s.trim().to_lowercase();
        if t.is_empty() || t == "auto" {
            return Self::from_path(path);
        }
        match t.as_str() {
            "rs" | "rust" => Some(Self::Rust),
            "ts" | "tsx" | "typescript" | "js" | "jsx" | "javascript" => Some(Self::TypeScript),
            _ => None,
        }
    }

    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        match ext.as_str() {
            "rs" => Some(Self::Rust),
            "ts" | "tsx" | "js" | "jsx" => Some(Self::TypeScript),
            _ => None,
        }
    }

    pub fn server_exe(&self) -> &'static str {
        match self {
            Self::Rust => "rust-analyzer",
            Self::TypeScript => "typescript-language-server",
        }
    }

    pub fn server_args(&self) -> &'static [&'static str] {
        match self {
            Self::Rust => &[],
            Self::TypeScript => &["--stdio"],
        }
    }

    pub fn resolve_executable(&self, workspace_root: &Path) -> PathBuf {
        match self {
            Self::Rust => {
                if let Some(p) = rust_analyzer_cached_path().filter(|p| p.exists()) {
                    return p;
                }
                PathBuf::from(self.server_exe())
            }
            Self::TypeScript => {
                let local = workspace_root
                    .join("node_modules")
                    .join(".bin")
                    .join("typescript-language-server");
                if local.exists() {
                    local
                } else if let Some(p) = ts_lsp_cached_path().filter(|p| p.exists()) {
                    p
                } else {
                    PathBuf::from(self.server_exe())
                }
            }
        }
    }
}

fn downloads_disabled() -> bool {
    matches!(
        std::env::var("LORIKEET_DISABLE_LSP_DOWNLOAD").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn lsp_cache_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".lorikeet").join("lsp"))
}

fn rust_analyzer_cached_path() -> Option<PathBuf> {
    lsp_cache_dir().map(|d| d.join("rust").join("rust-analyzer"))
}

fn ts_lsp_cache_dir() -> Option<PathBuf> {
    lsp_cache_dir().map(|d| d.join("ts"))
}

fn ts_lsp_cached_path() -> Option<PathBuf> {
    ts_lsp_cache_dir().map(|d| {
        d.join("node_modules")
            .join(".bin")
            .join("typescript-language-server")
    })
}

async fn ensure_rust_analyzer_installed() -> Result<PathBuf, String> {
    let Some(dst) = rust_analyzer_cached_path() else {
        return Err("Error: cannot determine home directory for LSP cache".to_string());
    };
    if dst.exists() {
        return Ok(dst);
    }

    if downloads_disabled() {
        return Err("Error: rust-analyzer not found and LSP downloads are disabled (set LORIKEET_DISABLE_LSP_DOWNLOAD=0).".to_string());
    }

    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let target = match (os, arch) {
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        ("linux", "x86_64") => "x86_64-unknown-linux-gnu",
        ("linux", "aarch64") => "aarch64-unknown-linux-gnu",
        _ => {
            return Err(format!(
                "Error: unsupported platform for managed rust-analyzer download: {}-{}",
                os, arch
            ))
        }
    };

    let url = format!(
        "https://github.com/rust-lang/rust-analyzer/releases/latest/download/rust-analyzer-{}.gz",
        target
    );

    let bytes = reqwest::get(url)
        .await
        .map_err(|e| format!("Error: download rust-analyzer: {}", e))?
        .bytes()
        .await
        .map_err(|e| format!("Error: download rust-analyzer body: {}", e))?;

    let parent = dst
        .parent()
        .ok_or_else(|| "Error: invalid rust-analyzer path".to_string())?;
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|e| format!("Error: create LSP cache dir: {}", e))?;

    let mut decoder = GzDecoder::new(bytes.as_ref());
    let mut out: Vec<u8> = Vec::new();
    std::io::Read::read_to_end(&mut decoder, &mut out)
        .map_err(|e| format!("Error: decompress rust-analyzer: {}", e))?;

    tokio::fs::write(&dst, out)
        .await
        .map_err(|e| format!("Error: write rust-analyzer: {}", e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = tokio::fs::metadata(&dst)
            .await
            .map_err(|e| format!("Error: stat rust-analyzer: {}", e))?
            .permissions();
        perms.set_mode(0o755);
        tokio::fs::set_permissions(&dst, perms)
            .await
            .map_err(|e| format!("Error: chmod rust-analyzer: {}", e))?;
    }

    Ok(dst)
}

async fn ensure_ts_language_server_installed() -> Result<PathBuf, String> {
    let Some(cache_dir) = ts_lsp_cache_dir() else {
        return Err("Error: cannot determine home directory for LSP cache".to_string());
    };
    let Some(bin) = ts_lsp_cached_path() else {
        return Err("Error: cannot determine TS LSP cache path".to_string());
    };
    if bin.exists() {
        return Ok(bin);
    }

    if downloads_disabled() {
        return Err("Error: typescript-language-server not found and LSP downloads are disabled (set LORIKEET_DISABLE_LSP_DOWNLOAD=0).".to_string());
    }

    tokio::fs::create_dir_all(&cache_dir)
        .await
        .map_err(|e| format!("Error: create TS LSP cache dir: {}", e))?;

    let pkg = cache_dir.join("package.json");
    if tokio::fs::metadata(&pkg).await.is_err() {
        tokio::fs::write(
            &pkg,
            r#"{"name":"lorikeet-ts-lsp","private":true,"version":"0.0.0"}"#,
        )
        .await
        .map_err(|e| format!("Error: write TS LSP package.json: {}", e))?;
    }

    let mut cmd = Command::new("npm");
    cmd.current_dir(&cache_dir)
        .arg("install")
        .arg("--silent")
        .arg("--no-progress")
        .arg("typescript")
        .arg("typescript-language-server")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let out = cmd
        .output()
        .await
        .map_err(|e| format!("Error: failed to run npm to install TS LSP: {}", e))?;

    if !out.status.success() {
        let mut msg = String::new();
        msg.push_str("Error: npm install failed while installing TypeScript LSP.\n");
        if !out.stderr.is_empty() {
            msg.push_str(&String::from_utf8_lossy(&out.stderr));
        } else if !out.stdout.is_empty() {
            msg.push_str(&String::from_utf8_lossy(&out.stdout));
        }
        return Err(msg);
    }

    if !bin.exists() {
        return Err(
            "Error: TS LSP install completed but binary was not found in cache.".to_string(),
        );
    }
    Ok(bin)
}

fn file_uri(path: &Path) -> Result<String, String> {
    url::Url::from_file_path(path)
        .map(|u| u.to_string())
        .map_err(|_| {
            format!(
                "Error: unable to convert path to file URI: {}",
                path.display()
            )
        })
}

fn lsp_language_id(lang: LspLanguage, path: &Path) -> &'static str {
    match lang {
        LspLanguage::Rust => "rust",
        LspLanguage::TypeScript => match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
            "js" | "jsx" => "javascript",
            _ => "typescript",
        },
    }
}

#[derive(Debug)]
pub(crate) struct LspClient {
    _child: Child,
    stdin: tokio::sync::Mutex<ChildStdin>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    diagnostics: Arc<Mutex<HashMap<String, Value>>>, // uri -> publishDiagnostics params
    next_id: AtomicU64,
    root: PathBuf,
    lang: LspLanguage,
    opened: Mutex<HashMap<String, i64>>, // uri -> version
}

impl LspClient {
    async fn start(
        lang: LspLanguage,
        root: PathBuf,
        policy: &SandboxPolicy,
    ) -> Result<Self, String> {
        let mut exe = lang.resolve_executable(&root);

        // Tie LSP server execution to the same allowlist as `bash` to keep policy consistent.
        // SandboxPolicy compares by basename, so local `node_modules/.bin/typescript-language-server`
        // is allowed if `typescript-language-server` is allowlisted.
        policy
            .check_command_allowed(exe.to_string_lossy().as_ref())
            .map_err(|e| e.to_string())?;

        let spawn_child = |exe: &PathBuf| -> Result<Child, std::io::Error> {
            let mut cmd = Command::new(exe);
            for a in lang.server_args() {
                cmd.arg(a);
            }
            cmd.stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
        };

        let mut child = match spawn_child(&exe) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                exe = match lang {
                    LspLanguage::Rust => ensure_rust_analyzer_installed().await?,
                    LspLanguage::TypeScript => ensure_ts_language_server_installed().await?,
                };
                policy
                    .check_command_allowed(exe.to_string_lossy().as_ref())
                    .map_err(|e| e.to_string())?;
                spawn_child(&exe).map_err(|e| {
                    format!(
                        "Error: failed to start LSP server `{}` after install: {}",
                        exe.display(),
                        e
                    )
                })?
            }
            Err(e) => {
                return Err(format!(
                    "Error: failed to start LSP server `{}`: {}",
                    exe.display(),
                    e
                ))
            }
        };

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Error: failed to open lsp stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Error: failed to open lsp stdout".to_string())?;

        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let diagnostics: Arc<Mutex<HashMap<String, Value>>> = Arc::new(Mutex::new(HashMap::new()));
        let pending_bg = pending.clone();
        let diagnostics_bg = diagnostics.clone();

        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            loop {
                match read_lsp_message(&mut reader).await {
                    Ok(Some(msg)) => {
                        if let Some(id) = msg.get("id").and_then(|v| v.as_u64()) {
                            if let Some(tx) = pending_bg.lock().remove(&id) {
                                let _ = tx.send(msg);
                            }
                            continue;
                        }
                        if let Some(method) = msg.get("method").and_then(|v| v.as_str()) {
                            if method == "textDocument/publishDiagnostics" {
                                if let Some(params) = msg.get("params") {
                                    if let Some(uri) = params.get("uri").and_then(|v| v.as_str()) {
                                        diagnostics_bg
                                            .lock()
                                            .insert(uri.to_string(), params.clone());
                                    }
                                }
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(_) => break,
                }
            }
        });

        let client = Self {
            _child: child,
            stdin: tokio::sync::Mutex::new(stdin),
            pending,
            diagnostics,
            next_id: AtomicU64::new(1),
            root,
            lang,
            opened: Mutex::new(HashMap::new()),
        };

        client.initialize().await?;
        Ok(client)
    }

    async fn initialize(&self) -> Result<(), String> {
        let root_uri = file_uri(&self.root)?;
        let params = json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "definition": {},
                    "references": {},
                    "rename": {},
                    "publishDiagnostics": {},
                }
            },
            "workspaceFolders": [{
                "uri": file_uri(&self.root)?,
                "name": self.root.file_name().and_then(|s| s.to_str()).unwrap_or("workspace")
            }]
        });

        let _ = self.request("initialize", params).await?;
        self.notify("initialized", json!({})).await?;
        Ok(())
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), String> {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        let mut w = self.stdin.lock().await;
        write_lsp_message(&mut *w, &msg).await
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });

        let (tx, rx) = oneshot::channel();
        self.pending.lock().insert(id, tx);
        let mut w = self.stdin.lock().await;
        write_lsp_message(&mut *w, &msg).await?;

        let resp = tokio::time::timeout(Duration::from_secs(10), rx)
            .await
            .map_err(|_| format!("Error: LSP request timeout: {}", method))?
            .map_err(|_| format!("Error: LSP response channel closed: {}", method))?;

        if let Some(err) = resp.get("error") {
            return Err(format!("Error: LSP {} failed: {}", method, err));
        }
        Ok(resp.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn ensure_open(&self, lang: LspLanguage, path: &Path) -> Result<String, String> {
        let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let uri = file_uri(&canon)?;
        if self.opened.lock().contains_key(&uri) {
            return Ok(uri);
        }

        let text = tokio::fs::read_to_string(&canon)
            .await
            .map_err(|e| format!("Error: read file for lsp: {}", e))?;

        let language_id = lsp_language_id(lang, path);
        let params = json!({
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": 1,
                "text": text
            }
        });
        self.notify("textDocument/didOpen", params).await?;
        self.opened.lock().insert(uri.clone(), 1);
        Ok(uri)
    }
}

async fn write_lsp_message(stdin: &mut ChildStdin, msg: &Value) -> Result<(), String> {
    let body = serde_json::to_vec(msg).map_err(|e| format!("Error: LSP json encode: {}", e))?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());

    stdin
        .write_all(header.as_bytes())
        .await
        .map_err(|e| format!("Error: LSP write header: {}", e))?;
    stdin
        .write_all(&body)
        .await
        .map_err(|e| format!("Error: LSP write body: {}", e))?;
    stdin
        .flush()
        .await
        .map_err(|e| format!("Error: LSP flush: {}", e))?;
    Ok(())
}

async fn read_lsp_message<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
) -> Result<Option<Value>, String> {
    // Read headers
    let mut content_length: Option<usize> = None;
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .await
            .map_err(|e| format!("Error: LSP read header: {}", e))?;
        if n == 0 {
            return Ok(None);
        }
        let l = line.trim_end_matches(&['\r', '\n'][..]);
        if l.is_empty() {
            break;
        }
        if let Some(v) = l.strip_prefix("Content-Length:") {
            content_length = v.trim().parse::<usize>().ok();
        }
    }

    let len = content_length.ok_or_else(|| "Error: LSP missing Content-Length".to_string())?;
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .await
        .map_err(|e| format!("Error: LSP read body: {}", e))?;

    serde_json::from_slice::<Value>(&buf)
        .map(Some)
        .map_err(|e| format!("Error: LSP json decode: {}", e))
}

#[derive(Default)]
pub struct LspManager {
    // (root, language) -> client
    clients: Mutex<HashMap<(PathBuf, LspLanguage), Arc<tokio::sync::Mutex<LspClient>>>>,
}

impl LspManager {
    pub async fn get_or_start(
        &self,
        lang: LspLanguage,
        root: PathBuf,
        policy: &SandboxPolicy,
    ) -> Result<Arc<tokio::sync::Mutex<LspClient>>, String> {
        let key = (root.clone(), lang);
        if let Some(c) = self.clients.lock().get(&key) {
            return Ok(c.clone());
        }

        let client = LspClient::start(lang, root.clone(), policy).await?;
        let arc = Arc::new(tokio::sync::Mutex::new(client));
        self.clients.lock().insert(key, arc.clone());
        Ok(arc)
    }
}

static LSP_MANAGER: std::sync::OnceLock<LspManager> = std::sync::OnceLock::new();

pub fn lsp_manager() -> &'static LspManager {
    LSP_MANAGER.get_or_init(LspManager::default)
}

#[derive(Debug, Clone)]
pub struct LspLocation {
    pub path: PathBuf,
    pub line1: usize,
    pub col1: usize,
}

fn parse_locations(v: &Value) -> Vec<LspLocation> {
    fn loc_from_location(obj: &Value) -> Option<LspLocation> {
        let uri = obj.get("uri")?.as_str()?;
        let path = url::Url::parse(uri).ok()?.to_file_path().ok()?;
        let start = obj.get("range")?.get("start")?;
        let line0 = start.get("line")?.as_u64()? as usize;
        let col0 = start.get("character")?.as_u64()? as usize;
        Some(LspLocation {
            path,
            line1: line0 + 1,
            col1: col0 + 1,
        })
    }

    fn loc_from_location_link(obj: &Value) -> Option<LspLocation> {
        let uri = obj.get("targetUri")?.as_str()?;
        let path = url::Url::parse(uri).ok()?.to_file_path().ok()?;
        let start = obj
            .get("targetSelectionRange")
            .or_else(|| obj.get("targetRange"))?
            .get("start")?;
        let line0 = start.get("line")?.as_u64()? as usize;
        let col0 = start.get("character")?.as_u64()? as usize;
        Some(LspLocation {
            path,
            line1: line0 + 1,
            col1: col0 + 1,
        })
    }

    match v {
        Value::Array(arr) => arr
            .iter()
            .filter_map(|x| {
                if x.get("uri").is_some() {
                    loc_from_location(x)
                } else if x.get("targetUri").is_some() {
                    loc_from_location_link(x)
                } else {
                    None
                }
            })
            .collect(),
        Value::Object(_) => {
            if v.get("uri").is_some() {
                loc_from_location(v).into_iter().collect()
            } else if v.get("targetUri").is_some() {
                loc_from_location_link(v).into_iter().collect()
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

pub async fn lsp_definition(
    policy: &SandboxPolicy,
    lang: LspLanguage,
    path: &Path,
    line1: usize,
    col1: usize,
) -> Result<Vec<LspLocation>, String> {
    let checked = policy.check_path_allowed(path).map_err(|e| e.to_string())?;
    let client = lsp_manager()
        .get_or_start(lang, policy.root.clone(), policy)
        .await?;
    let c = client.lock().await;
    let uri = c.ensure_open(lang, &checked).await?;

    let params = json!({
        "textDocument": {"uri": uri},
        "position": {"line": line1.saturating_sub(1), "character": col1.saturating_sub(1)}
    });
    let result = c.request("textDocument/definition", params).await?;
    Ok(parse_locations(&result))
}

pub async fn lsp_references(
    policy: &SandboxPolicy,
    lang: LspLanguage,
    path: &Path,
    line1: usize,
    col1: usize,
    include_declaration: bool,
) -> Result<Vec<LspLocation>, String> {
    let checked = policy.check_path_allowed(path).map_err(|e| e.to_string())?;
    let client = lsp_manager()
        .get_or_start(lang, policy.root.clone(), policy)
        .await?;
    let c = client.lock().await;
    let uri = c.ensure_open(lang, &checked).await?;

    let params = json!({
        "textDocument": {"uri": uri},
        "position": {"line": line1.saturating_sub(1), "character": col1.saturating_sub(1)},
        "context": {"includeDeclaration": include_declaration}
    });
    let result = c.request("textDocument/references", params).await?;
    Ok(parse_locations(&result))
}

pub async fn lsp_rename(
    policy: &SandboxPolicy,
    lang: LspLanguage,
    path: &Path,
    line1: usize,
    col1: usize,
    new_name: &str,
) -> Result<Value, String> {
    let checked = policy.check_path_allowed(path).map_err(|e| e.to_string())?;
    let client = lsp_manager()
        .get_or_start(lang, policy.root.clone(), policy)
        .await?;
    let c = client.lock().await;
    let uri = c.ensure_open(lang, &checked).await?;

    let params = json!({
        "textDocument": {"uri": uri},
        "position": {"line": line1.saturating_sub(1), "character": col1.saturating_sub(1)},
        "newName": new_name
    });
    c.request("textDocument/rename", params).await
}

pub async fn lsp_diagnostics(
    policy: &SandboxPolicy,
    lang: LspLanguage,
    path: &Path,
) -> Result<Value, String> {
    let checked = policy.check_path_allowed(path).map_err(|e| e.to_string())?;
    let client = lsp_manager()
        .get_or_start(lang, policy.root.clone(), policy)
        .await?;
    let c = client.lock().await;
    let uri = c.ensure_open(lang, &checked).await?;

    // Wait briefly for publishDiagnostics to arrive.
    let diag_map = c.diagnostics.clone();
    drop(c);

    let mut tries = 0;
    loop {
        if let Some(v) = diag_map.lock().get(&uri).cloned() {
            return Ok(v);
        }
        tries += 1;
        if tries > 20 {
            return Ok(json!({"uri": uri, "diagnostics": []}));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

pub async fn format_locations_with_snippets(
    policy: &SandboxPolicy,
    locs: &[LspLocation],
    limit: usize,
) -> String {
    let mut out = String::new();
    let mut n = 0usize;

    for loc in locs.iter().take(limit) {
        if policy.check_path_allowed(&loc.path).is_err() {
            continue;
        }

        let rel = loc
            .path
            .strip_prefix(&policy.root)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| loc.path.display().to_string());

        let line_text = match tokio::fs::read_to_string(&loc.path).await {
            Ok(s) => s
                .lines()
                .nth(loc.line1.saturating_sub(1))
                .unwrap_or("")
                .trim()
                .to_string(),
            Err(_) => String::new(),
        };

        out.push_str(&format!(
            "{}:{}:{}  {}\n",
            rel,
            loc.line1,
            loc.col1,
            truncate_single_line(&line_text, 180)
        ));

        n += 1;
        if n >= limit {
            break;
        }
    }

    if out.trim().is_empty() {
        "No results.".to_string()
    } else {
        out
    }
}

fn truncate_single_line(s: &str, max: usize) -> String {
    let t = s.replace('\t', " ").trim().to_string();
    if t.len() <= max {
        return t;
    }
    let mut out = t;
    out.truncate(max.saturating_sub(3));
    out.push_str("...");
    out
}
