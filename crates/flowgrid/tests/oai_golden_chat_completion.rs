#![cfg(feature = "openai")]

use flowgrid::ChatCompletion;

#[test]
fn golden_chat_completion_deserializes() {
    let raw = include_str!("fixtures/chat_completion.json");
    let c: ChatCompletion = serde_json::from_str(raw).unwrap();
    assert_eq!(c.id, "chatcmpl_fixture_1");
    assert_eq!(c.message_content().as_deref(), Some("Hello from fixture"));
}
