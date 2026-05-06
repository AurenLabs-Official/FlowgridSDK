#![cfg(feature = "openai")]

use flowgrid::ResponseObject;

#[test]
fn golden_response_object_deserializes() {
    let raw = include_str!("fixtures/response_object.json");
    let r: ResponseObject = serde_json::from_str(raw).unwrap();
    assert_eq!(r.id, "resp_fixture_1");
    assert!(r.output.is_empty());
}
