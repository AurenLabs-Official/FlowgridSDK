//! Optional typed parsing for Anthropic **Messages** streaming JSON lines (`data:` payloads).
//!
//! Requires Cargo features **`anthropic`** and **`stream-types`**.

use serde::Deserialize;

/// `delta` object when `type` is `text_delta` (streaming text).
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTextDelta {
    #[serde(rename = "type")]
    pub delta_type: Option<String>,
    pub text: Option<String>,
}

/// `content_block_delta` SSE event (common case: incremental text).
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicContentBlockDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: Option<String>,
    pub index: Option<u32>,
    pub delta: Option<AnthropicTextDelta>,
}

/// Parse a single `data:` line from an Anthropic SSE stream.
///
/// Returns `Ok(None)` for empty or whitespace-only payloads. Unknown JSON shapes yield
/// `Ok(Some(AnthropicStreamLine::Raw(value)))`.
pub fn parse_anthropic_message_stream_json(
    data: &str,
) -> Result<Option<AnthropicStreamLine>, serde_json::Error> {
    let t = data.trim();
    if t.is_empty() {
        return Ok(None);
    }
    let v: serde_json::Value = serde_json::from_str(t)?;
    let typ = v.get("type").and_then(|x| x.as_str()).unwrap_or_default();
    if typ == "content_block_delta" {
        return serde_json::from_value(v)
            .map(AnthropicStreamLine::ContentBlockDelta)
            .map(Some);
    }
    if typ == "message_start" || typ == "message_delta" || typ == "message_stop" || typ == "ping" {
        return Ok(Some(AnthropicStreamLine::Raw(v)));
    }
    Ok(Some(AnthropicStreamLine::Raw(v)))
}

/// Parsed streaming line: structured delta or raw JSON for other event types.
#[derive(Debug, Clone)]
pub enum AnthropicStreamLine {
    ContentBlockDelta(AnthropicContentBlockDeltaEvent),
    Raw(serde_json::Value),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_content_block_text_delta() {
        let raw =
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
        let l = parse_anthropic_message_stream_json(raw).unwrap().unwrap();
        match l {
            AnthropicStreamLine::ContentBlockDelta(e) => {
                assert_eq!(e.index, Some(0));
                assert_eq!(e.delta.as_ref().and_then(|d| d.text.as_deref()), Some("Hi"));
            }
            _ => panic!("expected ContentBlockDelta"),
        }
    }

    #[test]
    fn parse_content_block_start_as_raw() {
        let raw = r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        match parse_anthropic_message_stream_json(raw).unwrap().unwrap() {
            AnthropicStreamLine::Raw(v) => assert_eq!(v["type"], "content_block_start"),
            _ => panic!("expected Raw"),
        }
    }
}
