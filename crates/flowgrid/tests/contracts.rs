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
    use flowgrid::{
        ChatCompletion, Completion, CreateEmbeddingResponse, ResponseObject,
    };

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
