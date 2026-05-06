//! Contract tests: Anthropic message JSON (offline).

#![cfg(feature = "anthropic")]

use flowgrid::Message;

#[test]
fn contract_anthropic_message_deserializes() {
    let raw = include_str!("fixtures/contracts/anthropic_message_v1.json");
    let v: Message = serde_json::from_str(raw).unwrap();
    assert_eq!(v.id, "contract_msg_1");
    assert_eq!(v.text_concat().as_deref(), Some("Hi"));
}
