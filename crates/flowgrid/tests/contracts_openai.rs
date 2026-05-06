//! Contract tests: stable JSON shapes for public deserialize types (offline).

#![cfg(feature = "openai")]

use flowgrid::ChatCompletion;

#[test]
fn contract_openai_chat_completion_deserializes() {
    let raw = include_str!("fixtures/contracts/openai_chat_completion_v1.json");
    let v: ChatCompletion = serde_json::from_str(raw).unwrap();
    assert_eq!(v.id, "contract_chat_1");
    assert_eq!(v.message_content().as_deref(), Some("hello"));
}
