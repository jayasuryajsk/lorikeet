use std::path::PathBuf;

use base64::Engine;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
struct CodexAuthFile {
    #[serde(default)]
    pub tokens: CodexTokens,
    #[serde(default)]
    pub last_refresh: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
struct CodexTokens {
    pub access_token: String,
    pub id_token: String,
    pub refresh_token: String,
    #[serde(default)]
    pub account_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    id_token: Option<String>,
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
}

const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

fn codex_auth_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".codex").join("auth.json"))
}

fn read_codex_auth() -> Option<CodexAuthFile> {
    let path = codex_auth_path()?;
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

async fn refresh_tokens(refresh_token: &str) -> Result<TokenResponse, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(TOKEN_URL)
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", CODEX_CLIENT_ID),
            ("refresh_token", refresh_token),
        ])
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Codex OAuth refresh failed: HTTP {status}: {body}"));
    }

    resp.json::<TokenResponse>()
        .await
        .map_err(|e| format!("Codex OAuth refresh parse error: {e}"))
}

fn jwt_exp(jwt: &str) -> Option<i64> {
    let mut parts = jwt.split('.');
    let (_h, payload_b64, _s) = match (parts.next(), parts.next(), parts.next()) {
        (Some(h), Some(p), Some(s)) if !h.is_empty() && !p.is_empty() && !s.is_empty() => (h, p, s),
        _ => return None,
    };
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload_b64)
        .ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    v.get("exp").and_then(|x| x.as_i64())
}

fn token_needs_refresh(access_token: &str) -> bool {
    // Refresh if token expires within the next 5 minutes (or exp missing).
    let Some(exp) = jwt_exp(access_token) else {
        return true;
    };
    let now = time::OffsetDateTime::now_utc().unix_timestamp();
    exp <= now.saturating_add(300)
}

async fn load_fresh_tokens() -> Result<CodexTokens, String> {
    let mut auth = read_codex_auth().ok_or_else(|| {
        "Codex OAuth not found. Run `codex login` (sign in with ChatGPT) first.".to_string()
    })?;

    if auth.tokens.access_token.trim().is_empty() || auth.tokens.refresh_token.trim().is_empty() {
        return Err("Codex OAuth file is missing tokens".into());
    }

    if token_needs_refresh(&auth.tokens.access_token) {
        let refreshed = refresh_tokens(&auth.tokens.refresh_token).await?;
        if let Some(at) = refreshed.access_token {
            auth.tokens.access_token = at;
        }
        if let Some(it) = refreshed.id_token {
            auth.tokens.id_token = it;
        }
        if let Some(rt) = refreshed.refresh_token {
            auth.tokens.refresh_token = rt;
        }
    }

    Ok(auth.tokens)
}

/// Return a Bearer token suitable for calling the Codex ChatGPT backend.
///
/// Lorikeet reads `~/.codex/auth.json` created by `codex login`, refreshes if needed,
/// and returns the `access_token`. We do **not** write back to `auth.json`.
pub async fn codex_chatgpt_access_token() -> Result<String, String> {
    let tokens = load_fresh_tokens().await?;
    if tokens.access_token.trim().is_empty() {
        return Err("Codex OAuth access_token is empty".into());
    }
    Ok(tokens.access_token)
}
