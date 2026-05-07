pub fn expected_keys() -> Vec<&'static str> {
    vec![
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "transformer.ln_f.weight",
        "lm_head.weight",
    ]
}
