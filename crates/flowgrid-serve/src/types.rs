use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct ChatReq {
    pub model: String,
    pub stream: Option<bool>,
    pub messages: Option<Vec<ChatMessage>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesReq {
    pub model: String,
    pub stream: Option<bool>,
    pub input: Value,
}
