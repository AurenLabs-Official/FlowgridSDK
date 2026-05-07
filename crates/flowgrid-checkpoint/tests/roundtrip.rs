use burn::backend::NdArray;
use burn::tensor::Int;
use burn::tensor::Tensor;
use flowgrid_checkpoint::{load_manifest, load_nano_gpt_checkpoint, save_nano_gpt_checkpoint};
use flowgrid_model::NanoGptConfig;
use tempfile::tempdir;

type B = NdArray<f32>;

#[test]
fn save_load_forward_matches() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let cfg = NanoGptConfig {
        vocab_size: 48,
        block_size: 12,
        n_layer: 2,
        n_head: 4,
        n_embd: 24,
        dropout: 0.0,
        use_rope: true,
        rope_theta: 10_000.0,
    };
    let model = cfg.init::<B>(&device);
    let dir = tempdir().unwrap();
    save_nano_gpt_checkpoint(&model, dir.path(), &cfg, None, None).unwrap();
    let m = load_manifest(dir.path()).unwrap();
    assert_eq!(m.manifest_version, 1);
    assert!(m.fingerprint.starts_with("b3:"));
    let (loaded, _) = load_nano_gpt_checkpoint::<B>(dir.path(), &device).unwrap();

    let ids: Vec<i32> = vec![3, 7, 9, 11];
    let inp = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([1, ids.len()]);
    let a = model
        .forward(inp.clone())
        .into_data()
        .convert::<f32>()
        .value;
    let b = loaded.forward(inp).into_data().convert::<f32>().value;
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "{x} vs {y}");
    }
}

#[test]
fn save_with_lora_sidecar_sets_manifest_fields() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let cfg = NanoGptConfig {
        vocab_size: 16,
        block_size: 8,
        n_layer: 1,
        n_head: 2,
        n_embd: 8,
        dropout: 0.0,
        use_rope: true,
        rope_theta: 10_000.0,
    };
    let model = cfg.init::<B>(&device);
    let dir = tempdir().unwrap();
    save_nano_gpt_checkpoint(&model, dir.path(), &cfg, None, Some("lora.json")).unwrap();
    let m = load_manifest(dir.path()).unwrap();
    assert_eq!(m.lora.as_deref(), Some("lora.json"));
    assert_eq!(m.lora_schema_version, Some(1));
}
