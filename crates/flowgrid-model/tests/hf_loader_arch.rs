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
fn validate_keys_missing_errors() {
    let tensors: HashMap<String, Vec<u8>> = HashMap::new();
    let err = validate_expected_keys(&tensors, HfArch::Gpt2).expect_err("must fail");
    assert!(err.to_string().contains("missing HF tensor key"));
}
