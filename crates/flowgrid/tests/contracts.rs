//! Offline contract tests: versioned JSON fixtures for public deserialize types.
//!
//! Naming: `<provider>_<resource>_v<api_hint>_<scenario>.json` under `tests/fixtures/contracts/`.
//! Use `tools/import_contract.ps1` / `tools/import_contract.sh` from the repo root
//! to copy and redact captures into this directory.

macro_rules! contract_fixture {
    ($path:literal, $ty:ty, |$v:ident| $body:block) => {
        let raw = include_str!($path);
        let $v: $ty = serde_json::from_str(raw).unwrap_or_else(|e| {
            panic!("deserialize {}: {e}", $path);
        });
        $body
    };
}

#[cfg(feature = "openai")]
mod openai {
    use flowgrid::{ChatCompletion, Completion, CreateEmbeddingResponse, ResponseObject};

    #[test]
    fn openai_chat_completion_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/openai_chat_completion_v1_deserialize.json",
            ChatCompletion,
            |v| {
                assert_eq!(v.id, "contract_chat_1");
                assert_eq!(v.message_content().as_deref(), Some("hello"));
            }
        );
    }

    #[test]
    fn openai_embedding_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/openai_embedding_v1_deserialize.json",
            CreateEmbeddingResponse,
            |v| {
                assert_eq!(v.data.len(), 1);
                let u = v.usage.as_ref().expect("usage");
                assert_eq!(u.prompt_tokens, Some(2));
                assert_eq!(u.total_tokens, Some(2));
            }
        );
    }

    #[test]
    fn openai_completion_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/openai_completion_v1_deserialize.json",
            Completion,
            |v| {
                assert_eq!(v.id.as_deref(), Some("cmpl_contract_1"));
                let u = v.usage.as_ref().expect("usage");
                assert_eq!(u.completion_tokens, Some(2));
            }
        );
    }

    #[test]
    fn openai_response_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/openai_response_v1_deserialize.json",
            ResponseObject,
            |v| {
                assert_eq!(v.id, "resp_contract_1");
                let u = v.usage.as_ref().expect("usage");
                assert_eq!(u.input_tokens, Some(10));
                assert_eq!(u.output_tokens, Some(5));
            }
        );
    }
}

#[cfg(all(feature = "openai", feature = "assistants"))]
mod openai_assistants {
    use flowgrid::{
        Assistant, ListPage, Thread, ThreadMessage, ThreadRun, ThreadRunStep,
    };

    #[test]
    fn openai_assistant_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/openai_assistant_v1_deserialize.json",
            Assistant,
            |v| {
                assert_eq!(v.id, "asst_contract_1");
                assert_eq!(v.model.as_deref(), Some("gpt-4o-mini"));
            }
        );
    }

    #[test]
    fn openai_thread_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/openai_thread_v1_deserialize.json",
            Thread,
            |v| {
                assert_eq!(v.id, "thread_contract_1");
            }
        );
    }

    #[test]
    fn openai_thread_message_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/openai_thread_message_v1_deserialize.json",
            ThreadMessage,
            |v| {
                assert_eq!(v.id, "msg_contract_1");
                assert_eq!(v.thread_id.as_deref(), Some("thread_contract_1"));
            }
        );
    }

    #[test]
    fn openai_thread_run_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/openai_thread_run_v1_deserialize.json",
            ThreadRun,
            |v| {
                assert_eq!(v.id, "run_contract_1");
                assert_eq!(v.status.as_deref(), Some("completed"));
            }
        );
    }

    #[test]
    fn openai_assistants_list_page_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/openai_assistants_list_v1_deserialize.json",
            ListPage<Assistant>,
            |v| {
                assert_eq!(v.data.len(), 1);
                assert_eq!(v.data[0].id, "asst_contract_1");
            }
        );
    }

    #[test]
    fn openai_thread_run_steps_list_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/openai_thread_run_steps_list_v1_deserialize.json",
            ListPage<ThreadRunStep>,
            |v| {
                assert_eq!(v.data.len(), 1);
                assert_eq!(v.data[0].id, "step_contract_1");
                assert_eq!(v.data[0].step_type.as_deref(), Some("message_creation"));
            }
        );
    }
}

#[cfg(feature = "anthropic")]
mod anthropic {
    use flowgrid::Message;

    #[test]
    fn anthropic_message_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/anthropic_message_v1_deserialize.json",
            Message,
            |v| {
                assert_eq!(v.id, "contract_msg_1");
                assert_eq!(v.text_concat().as_deref(), Some("Hi"));
            }
        );
    }
}

#[cfg(all(feature = "anthropic", feature = "beta"))]
mod anthropic_beta {
    use flowgrid::BetaModelsListResponse;

    #[test]
    fn anthropic_beta_models_v1_deserialize() {
        contract_fixture!(
            "fixtures/contracts/anthropic_beta_models_v1_deserialize.json",
            BetaModelsListResponse,
            |v| {
                assert_eq!(v.data.len(), 1);
                assert_eq!(v.data[0].id, "claude-contract-1");
                assert_eq!(v.data[0].display_name.as_deref(), Some("Contract Model"));
                assert_eq!(v.data[0].kind.as_deref(), Some("model"));
            }
        );
    }
}
