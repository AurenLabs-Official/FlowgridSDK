#![cfg(all(feature = "openai", feature = "webhooks"))]

#[test]
fn webhook_signature_roundtrip() {
    use flowgrid::ClientBuilder;
    use hmac::{Hmac, Mac};
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
    use sha2::Sha256;
    use std::time::{SystemTime, UNIX_EPOCH};

    let secret = "whsec_test";
    let client = ClientBuilder::default()
        .api_key("k")
        .webhook_secret(secret)
        .build()
        .unwrap();
    let wh = client.webhooks();
    let body = r#"{"type":"test","data":{}}"#;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
    mac.update(format!("{ts}.{body}").as_bytes());
    let sig = hex::encode(mac.finalize().into_bytes());
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("openai-signature"),
        HeaderValue::from_str(&format!("t={ts},v1={sig}")).unwrap(),
    );
    wh.verify_signature(body, &headers).unwrap();
    let v = wh.unwrap(body, &headers).unwrap();
    assert_eq!(v["type"], "test");
}
