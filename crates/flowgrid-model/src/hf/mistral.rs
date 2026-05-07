pub fn expected_keys() -> Vec<&'static str> {
    vec![
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
}
