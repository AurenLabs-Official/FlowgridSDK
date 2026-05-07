use anyhow::{Context, Result};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use flowgrid_model::lora::LoraSpec;
use flowgrid_model::{NanoGpt, NanoGptConfig};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub arch: String,
    pub dtype: String,
    pub vocab_size: usize,
    pub block_size: usize,
    pub hidden: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: Option<usize>,
    pub tokenizer_path: Option<String>,
    pub lora: Option<String>,
    pub fingerprint: String,
    #[serde(default = "default_use_rope")]
    pub use_rope: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_use_rope() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    10_000.0
}

impl Manifest {
    pub fn from_nano_gpt(cfg: &NanoGptConfig, tokenizer_path: Option<String>) -> Self {
        Self {
            arch: "nanogpt".into(),
            dtype: "f32".into(),
            vocab_size: cfg.vocab_size,
            block_size: cfg.block_size,
            hidden: cfg.n_embd,
            n_layer: cfg.n_layer,
            n_head: cfg.n_head,
            n_kv_head: None,
            tokenizer_path,
            lora: None,
            fingerprint: format!(
                "nanogpt-v{}-b{}-l{}-h{}",
                cfg.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_embd
            ),
            use_rope: cfg.use_rope,
            rope_theta: cfg.rope_theta,
        }
    }

    pub fn to_nano_gpt_config(&self) -> NanoGptConfig {
        NanoGptConfig {
            vocab_size: self.vocab_size,
            block_size: self.block_size,
            n_layer: self.n_layer,
            n_head: self.n_head.max(1),
            n_embd: self.hidden,
            dropout: 0.0,
            use_rope: self.use_rope,
            rope_theta: self.rope_theta,
        }
    }
}

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("manifest.json")
}

fn model_path(dir: &Path) -> PathBuf {
    dir.join("model.bin")
}

fn lora_spec_path(dir: &Path) -> PathBuf {
    dir.join("lora.json")
}

pub fn save_manifest(dir: impl AsRef<Path>, manifest: &Manifest) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let path = manifest_path(dir);
    let body = serde_json::to_vec_pretty(manifest).context("serialize manifest")?;
    std::fs::write(&path, body).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

pub fn load_manifest(dir: impl AsRef<Path>) -> Result<Manifest> {
    let path = manifest_path(dir.as_ref());
    let body = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&body).context("parse manifest")
}

/// Save [`Manifest`] plus a Burn binary record of [`NanoGpt` weights] to `model.bin`.
pub fn save_nano_gpt_checkpoint<B: Backend>(
    model: &NanoGpt<B>,
    dir: impl AsRef<Path>,
    cfg: &NanoGptConfig,
    tokenizer_path: Option<String>,
) -> Result<()> {
    let dir = dir.as_ref();
    save_manifest(dir, &Manifest::from_nano_gpt(cfg, tokenizer_path))?;
    let path = model_path(dir);
    let record = model.clone().into_record();
    BinFileRecorder::<FullPrecisionSettings>::default()
        .record(record, path.clone())
        .with_context(|| format!("write burn record {}", path.display()))?;
    Ok(())
}

/// Load manifest and model weights (CPU/`NdArray` backend `B`).
pub fn load_nano_gpt_checkpoint<B: Backend>(
    dir: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(NanoGpt<B>, Manifest)> {
    let dir = dir.as_ref();
    let manifest = load_manifest(dir)?;
    let cfg = manifest.to_nano_gpt_config();
    let path = model_path(dir);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder
        .load(path.clone(), device)
        .with_context(|| format!("read burn record {}", path.display()))?;
    let model = cfg.init(device).load_record(record);
    Ok((model, manifest))
}

pub fn load_nano_gpt_config(dir: impl AsRef<Path>) -> Result<NanoGptConfig> {
    let m = load_manifest(dir)?;
    Ok(m.to_nano_gpt_config())
}

pub fn resolve_tokenizer_path(dir: impl AsRef<Path>) -> Result<Option<PathBuf>> {
    let dir = dir.as_ref();
    let manifest = load_manifest(dir)?;
    let path = manifest.tokenizer_path.map(PathBuf::from);
    Ok(path.map(|p| if p.is_absolute() { p } else { dir.join(p) }))
}

pub fn save_lora_spec(dir: impl AsRef<Path>, spec: &LoraSpec) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let path = lora_spec_path(dir);
    let body = serde_json::to_vec_pretty(spec).context("serialize lora spec")?;
    std::fs::write(&path, body).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

pub fn load_lora_spec(dir: impl AsRef<Path>) -> Result<LoraSpec> {
    let path = lora_spec_path(dir.as_ref());
    let body = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&body).context("parse lora spec")
}
