#![cfg(feature = "openai")]

use serde_json::json;
use std::env;

#[cfg(feature = "azure")]
#[tokio::test]
#[ignore]
async fn live_azure_chat_smoke() {
    use flowgrid::{AzureClientBuilder, CreateChatCompletionRequest};
    env::var("AZURE_OPENAI_KEY").expect("AZURE_OPENAI_KEY");
    env::var("AZURE_OPENAI_ENDPOINT").expect("AZURE_OPENAI_ENDPOINT");
    let client = AzureClientBuilder::new()
        .api_key(env::var("AZURE_OPENAI_KEY").unwrap())
        .endpoint(env::var("AZURE_OPENAI_ENDPOINT").unwrap())
        .build()
        .unwrap();
    let dep = env::var("AZURE_OPENAI_DEPLOYMENT").expect("AZURE_OPENAI_DEPLOYMENT");
    let req = CreateChatCompletionRequest {
        model: dep,
        messages: vec![json!({"role":"user","content":"Say hi in one word."})],
        stream: Some(false),
        extra: serde_json::Map::new(),
    };
    let _ = client.chat().completions().create(&req).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn live_openai_embeddings_smoke() {
    use flowgrid::ClientBuilder;
    use flowgrid::CreateEmbeddingRequest;
    env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY");
    let client = ClientBuilder::default()
        .api_key(env::var("OPENAI_API_KEY").unwrap())
        .build()
        .unwrap();
    let req = CreateEmbeddingRequest {
        model: "text-embedding-3-small".into(),
        input: json!("hello world"),
        extra: serde_json::Map::new(),
    };
    let resp = client.embeddings().create(&req).await.unwrap();
    assert!(!resp.data.is_empty());
}
