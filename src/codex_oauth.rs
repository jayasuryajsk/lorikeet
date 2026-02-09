use std::path::PathBuf;

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

async fn exchange_id_token_for_api_key(id_token: &str) -> Result<String, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(TOKEN_URL)
        .form(&[
            (
                "grant_type",
                "urn:ietf:params:oauth:grant-type:token-exchange",
            ),
            ("client_id", CODEX_CLIENT_ID),
            ("requested_token", "openai-api-key"),
            (
                "subject_token_type",
                "urn:ietf:params:oauth:token-type:id_token",
            ),
            ("subject_token", id_token),
        ])
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!(
            "Codex OAuth token exchange failed: HTTP {status}: {body}"
        ));
    }

    let v: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Codex OAuth token exchange parse error: {e}"))?;

    let key = v
        .get("access_token")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if key.is_empty() {
        return Err("Codex OAuth token exchange did not return an access_token".into());
    }
    Ok(key)
}

fn should_refresh(last_refresh: Option<&str>) -> bool {
    // auth.json uses RFC3339 timestamps like "2026-02-07T10:08:14.853330Z".
    // If parsing fails, refresh pessimistically.
    let Some(ts) = last_refresh else {
        return true;
    };
    let Ok(dt) = time::OffsetDateTime::parse(ts, &time::format_description::well_known::Rfc3339)
    else {
        return true;
    };
    let age = time::OffsetDateTime::now_utc() - dt;
    // Access tokens are typically short-lived; refresh if older than 45 minutes.
    age > time::Duration::minutes(45)
}

/// Best-effort: derive an OpenAI API key using the Codex "Sign in with ChatGPT" OAuth cache.
///
/// This reads `~/.codex/auth.json`, refreshes tokens if they're stale, then performs a token
/// exchange to obtain an API key string.
pub async fn openai_api_key_from_codex_oauth() -> Result<String, String> {
    let mut auth = read_codex_auth().ok_or_else(|| {
        "Codex OAuth not found. Run `codex login` (or sign in with ChatGPT in Codex CLI) first."
            .to_string()
    })?;

    if should_refresh(auth.last_refresh.as_deref()) {
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
        auth.last_refresh = Some(
            time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "unknown".into()),
        );

        // Write back so Codex and Lorikeet stay in sync.
        if let Some(path) = codex_auth_path() {
            let _ = std::fs::write(
                &path,
                serde_json::to_string_pretty(&auth).unwrap_or_default(),
            );
        }
    }

    exchange_id_token_for_api_key(&auth.tokens.id_token).await
}
