use axum::http::{HeaderMap, StatusCode};

#[derive(Clone, Debug)]
pub struct AuthConfig {
    pub required: bool,
    pub keys: Vec<String>,
}

impl AuthConfig {
    pub fn from_env() -> Self {
        let keys = std::env::var("FLOWGRID_SERVE_API_KEYS")
            .ok()
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Self {
            required: !keys.is_empty(),
            keys,
        }
    }
}

pub fn authorize(headers: &HeaderMap, cfg: &AuthConfig) -> Result<(), StatusCode> {
    if !cfg.required {
        return Ok(());
    }

    let mut authorized = false;

    if let Some(auth) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
    {
        let token = auth.strip_prefix("Bearer ").unwrap_or("");
        authorized |= key_matches_any(token, &cfg.keys);
    }

    const API_KEY: &str = "api-key";
    const X_API_KEY: &str = "x-api-key";
    if let Some(raw) = headers.get(API_KEY).and_then(|h| h.to_str().ok()) {
        authorized |= key_matches_any(raw.trim(), &cfg.keys);
    }
    if let Some(raw) = headers.get(X_API_KEY).and_then(|h| h.to_str().ok()) {
        authorized |= key_matches_any(raw.trim(), &cfg.keys);
    }

    if authorized {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

fn key_matches_any(candidate: &str, keys: &[String]) -> bool {
    let candidate_bytes = candidate.as_bytes();
    keys.iter()
        .any(|k| constant_time_eq(candidate_bytes, k.as_bytes()))
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}
