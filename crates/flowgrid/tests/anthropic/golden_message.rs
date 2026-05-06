use flowgrid_anthropic::Message;

#[test]
fn golden_message_deserializes() {
    let raw = include_str!("fixtures/message.json");
    let m: Message = serde_json::from_str(raw).unwrap();
    assert_eq!(m.id, "msg_fixture_1");
    assert_eq!(m.text_concat().as_deref(), Some("Hello from fixture"));
}
