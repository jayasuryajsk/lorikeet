use regex::Regex;

/// Best-effort redaction / safety filter for content stored in memory.
///
/// This is intentionally conservative: it avoids storing obvious secrets
/// and trims very large content.
pub struct Redactor {
    patterns: Vec<Regex>,
}

impl Redactor {
    pub fn new() -> Self {
        let patterns = vec![
            // OpenAI-style keys, OpenRouter keys, generic API key patterns.
            Regex::new(r"(?i)\bsk-[a-z0-9_\-]{16,}\b").unwrap(),
            Regex::new(r"(?i)\bapi[_-]?key\b\s*[:=]\s*[^\s\n]{8,}").unwrap(),
            Regex::new(r"(?i)\bsecret\b\s*[:=]\s*[^\s\n]{8,}").unwrap(),
            Regex::new(r"(?i)\btoken\b\s*[:=]\s*[^\s\n]{8,}").unwrap(),
            // Private keys.
            Regex::new(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----").unwrap(),
        ];

        Self { patterns }
    }

    pub fn redact(&self, input: &str) -> String {
        // Hard cap to keep memories small.
        const MAX: usize = 2000;
        let mut out = input.to_string();
        if out.len() > MAX {
            out.truncate(MAX);
            out.push_str("…");
        }

        for re in &self.patterns {
            out = re.replace_all(&out, "[REDACTED]").to_string();
        }

        out
    }

    pub fn looks_sensitive(&self, input: &str) -> bool {
        self.patterns.iter().any(|re| re.is_match(input))
    }
}
