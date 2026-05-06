//! Webhook verification helpers (feature `webhooks`).
//!
//! This follows the common `t=<unix>,v1=<hex(hmac)>` scheme used by OpenAI webhook examples.

use crate::internal::client::oai::OpenAI;
use crate::internal::error::oai::{Error, Result};
use hmac::{Hmac, Mac};
use reqwest::header::HeaderMap;
use serde_json::Value;
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};
use subtle::{Choice, ConstantTimeEq};

type HmacSha256 = Hmac<Sha256>;

/// Webhook utilities (`client.webhooks` in Node).
pub struct WebhooksClient<'a> {
    secret: Option<String>,
    _marker: std::marker::PhantomData<&'a OpenAI>,
}

impl<'a> WebhooksClient<'a> {
    pub(crate) fn new(client: &'a OpenAI) -> Self {
        Self {
            secret: client.transport.config.webhook_secret.clone(),
            _marker: std::marker::PhantomData,
        }
    }

    fn secret(&self) -> Result<&str> {
        self.secret
            .as_deref()
            .ok_or_else(|| Error::Webhook("webhook_secret not configured".to_string()))
    }

    /// Verify signature headers, then parse JSON (raw body string; do not pre-parse JSON).
    pub fn unwrap(&self, body: &str, headers: &HeaderMap) -> Result<Value> {
        self.verify_signature(body, headers)?;
        Ok(serde_json::from_str(body)?)
    }

    /// Verify `openai-signature` only.
    pub fn verify_signature(&self, body: &str, headers: &HeaderMap) -> Result<()> {
        let secret = self.secret()?;
        let sig_header = headers
            .get("openai-signature")
            .or_else(|| headers.get("OpenAI-Signature"))
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| Error::Webhook("missing openai-signature header".to_string()))?;

        let mut ts: Option<i64> = None;
        let mut v1: Option<Vec<u8>> = None;
        for part in sig_header.split(',') {
            let mut it = part.trim().splitn(2, '=');
            let k = it.next().unwrap_or_default().trim();
            let v = it.next().unwrap_or_default().trim();
            match k {
                "t" => ts = v.parse().ok(),
                "v1" => v1 = hex::decode(v).ok(),
                _ => {}
            }
        }
        let ts = ts.ok_or_else(|| Error::Webhook("missing timestamp in signature".to_string()))?;
        let expected = v1.ok_or_else(|| Error::Webhook("missing v1 signature".to_string()))?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| Error::Webhook(e.to_string()))?
            .as_secs() as i64;
        if (now - ts).abs() > 300 {
            return Err(Error::Webhook(
                "signature timestamp skew too large".to_string(),
            ));
        }

        let payload = format!("{ts}.{body}");
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .map_err(|e| Error::Webhook(e.to_string()))?;
        mac.update(payload.as_bytes());
        let result = mac.finalize().into_bytes();
        if result.len() != expected.len() {
            return Err(Error::Webhook("invalid webhook signature".to_string()));
        }
        let mut ok = Choice::from(1u8);
        for (x, y) in result.iter().zip(expected.iter()) {
            ok &= x.ct_eq(y);
        }
        if !bool::from(ok) {
            return Err(Error::Webhook("invalid webhook signature".to_string()));
        }
        Ok(())
    }
}
