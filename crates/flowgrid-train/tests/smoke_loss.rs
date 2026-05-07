use burn::backend::{Autodiff, NdArray};
use flowgrid_data::write_token_blob;
use flowgrid_model::{NanoGptConfig, NanoGpt};
use flowgrid_train::loop_train::debug_loss_file;
use std::io::Write;
use tempfile::NamedTempFile;

type B = Autodiff<NdArray<f32>>;

#[test]
fn debug_loss_runs_on_tiny_bin() {
    let mut f = NamedTempFile::new().unwrap();
    let ids: Vec<u32> = (0u32..64).map(|i| (i % 32) as u32).collect();
    write_token_blob(f.path(), &ids).unwrap();
    f.flush().unwrap();

    let device = burn_ndarray::NdArrayDevice::Cpu;
    let cfg = NanoGptConfig {
        vocab_size: 32,
        block_size: 8,
        n_layer: 1,
        n_head: 4,
        n_embd: 16,
        dropout: 0.0,
    };
    let model: NanoGpt<B> = cfg.init(&device);
    let l = debug_loss_file(&model, f.path(), &cfg, &device);
    assert!(l.is_finite());
}
