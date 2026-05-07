#![cfg(all(feature = "openai", feature = "assistants"))]

use flowgrid::{AssistantsListParams, ClientBuilder, ListPage, ThreadRun, ThreadRunStep};

#[tokio::test]
async fn wiremock_lists_run_steps() {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(
            "/v1/threads/thread_w_mock/runs/run_w_mock/steps",
        ))
        .and(header("authorization", "Bearer k"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [{
                "id": "step_wm_1",
                "object": "thread.run.step",
                "run_id": "run_w_mock",
                "thread_id": "thread_w_mock",
                "type": "tool_calls",
                "status": "completed"
            }],
            "has_more": false
        })))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/threads/thread_w_mock/runs/run_w_mock/cancel"))
        .and(header("authorization", "Bearer k"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "run_w_mock",
            "object": "thread.run",
            "status": "cancelled"
        })))
        .mount(&server)
        .await;

    let client = ClientBuilder::new()
        .api_key("k")
        .base_url(format!("{}/v1", server.uri()))
        .build()
        .unwrap();
    let page: ListPage<ThreadRunStep> = client
        .threads()
        .thread("thread_w_mock")
        .runs()
        .list_steps_typed("run_w_mock", &AssistantsListParams::default())
        .await
        .unwrap();
    assert_eq!(page.data.len(), 1);
    assert_eq!(page.data[0].id, "step_wm_1");
    assert_eq!(page.data[0].step_type.as_deref(), Some("tool_calls"));

    let run: ThreadRun = client
        .threads()
        .thread("thread_w_mock")
        .runs()
        .cancel_typed("run_w_mock")
        .await
        .unwrap();
    assert_eq!(run.status.as_deref(), Some("cancelled"));
}
