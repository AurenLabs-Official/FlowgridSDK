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
    use flowgrid::ChatCompletion;

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
