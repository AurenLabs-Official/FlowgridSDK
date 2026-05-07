use std::collections::HashMap;

use flowgrid_model::hf_loader::{
    infer_arch_from_config_text, validate_expected_keys, HfArch,
};

#[test]
fn infer_arch_gpt2() {
    let cfg = r#"{"architectures":["GPT2LMHeadModel"]}"#;
    let arch = infer_arch_from_config_text(cfg).expect("arch");
    assert_eq!(arch, HfArch::Gpt2);
}

#[test]
fn infer_arch_llama() {
    let cfg = r#"{"architectures":["LlamaForCausalLM"]}"#;
    let arch = infer_arch_from_config_text(cfg).expect("arch");
    assert_eq!(arch, HfArch::Llama);
}

#[test]
fn infer_arch_mistral_and_qwen() {
    let mistral = r#"{"architectures":["MistralForCausalLM"]}"#;
    let qwen = r#"{"architectures":["Qwen2ForCausalLM"]}"#;
    assert_eq!(
        infer_arch_from_config_text(mistral).expect("mistral arch"),
        HfArch::Mistral
    );
    assert_eq!(
        infer_arch_from_config_text(qwen).expect("qwen arch"),
        HfArch::Qwen2
    );
}

#[test]
fn validate_keys_missing_errors() {
    let tensors: HashMap<String, Vec<u8>> = HashMap::new();
    let err = validate_expected_keys(&tensors, HfArch::Gpt2).expect_err("must fail");
    assert!(err.to_string().contains("missing HF tensor keys"));
    assert!(err.to_string().contains("expected="));
    assert!(err.to_string().contains("found="));
}
