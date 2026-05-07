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
    let auth = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default();
    let ok = cfg.keys.iter().any(|k| auth == format!("Bearer {k}"));
    if ok {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}
