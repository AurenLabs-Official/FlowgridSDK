use anyhow::{Context, Result};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use flowgrid_model::lora::LoraSpec;
use flowgrid_model::{NanoGpt, NanoGptConfig};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const MANIFEST_FORMAT_V1: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version for `manifest.json` (increment on breaking layout changes).
    #[serde(default = "default_manifest_version")]
    pub manifest_version: u32,
    pub arch: String,
    pub dtype: String,
    pub vocab_size: usize,
    pub block_size: usize,
    pub hidden: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: Option<usize>,
    pub tokenizer_path: Option<String>,
    /// Relative path under checkpoint dir for LoRA metadata (e.g. `lora.json`) when using adapter training.
    pub lora: Option<String>,
    /// Schema version for sidecar referenced by [`Self::lora`] (e.g. `1` = `lora.json` + weights in `model.bin`).
    #[serde(default)]
    pub lora_schema_version: Option<u32>,
    /// Stable identity: BLAKE3 over config basis plus `model.bin` digest (see `save_nano_gpt_checkpoint`).
    /// Older manifests may use a human-readable `nanogpt-v…` string instead of `b3:…`.
    pub fingerprint: String,
    #[serde(default = "default_use_rope")]
    pub use_rope: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_manifest_version() -> u32 {
    MANIFEST_FORMAT_V1
}

fn default_use_rope() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn config_basis(cfg: &NanoGptConfig, tokenizer_path: Option<&str>) -> String {
    format!(
        "arch=nanogpt\ndtype=f32\nvocab={}\nblock={}\nn_layer={}\nn_head={}\nn_kv_head={}\nn_embd={}\nuse_rope={}\nrope_theta={}\ntokenizer={}\n",
        cfg.vocab_size,
        cfg.block_size,
        cfg.n_layer,
        cfg.n_head,
        cfg.resolved_n_kv_head(),
        cfg.n_embd,
        cfg.use_rope,
        cfg.rope_theta,
        tokenizer_path.unwrap_or(""),
    )
}

/// BLAKE3 over config only (no weights). Prefixed `b3:` for new manifests.
pub fn fingerprint_config_only(cfg: &NanoGptConfig, tokenizer_path: Option<&str>) -> String {
    let mut h = blake3::Hasher::new();
    h.update(b"flowgrid-manifest-config-v1\0");
    h.update(config_basis(cfg, tokenizer_path).as_bytes());
    format!("b3:{}", h.finalize().to_hex())
}

fn fingerprint_checkpoint(
    cfg: &NanoGptConfig,
    tokenizer_path: Option<&str>,
    model_file_blake3: &[u8; 32],
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(b"flowgrid-checkpoint-v1\0");
    h.update(config_basis(cfg, tokenizer_path).as_bytes());
    h.update(model_file_blake3);
    format!("b3:{}", h.finalize().to_hex())
}

impl Manifest {
    /// Build manifest metadata from training config only. `fingerprint` is config-scoped (no `model.bin` yet).
    pub fn from_nano_gpt(cfg: &NanoGptConfig, tokenizer_path: Option<String>) -> Self {
        let fingerprint = fingerprint_config_only(cfg, tokenizer_path.as_deref());
        Self {
            manifest_version: MANIFEST_FORMAT_V1,
            arch: "nanogpt".into(),
            dtype: "f32".into(),
            vocab_size: cfg.vocab_size,
            block_size: cfg.block_size,
            hidden: cfg.n_embd,
            n_layer: cfg.n_layer,
            n_head: cfg.n_head,
            n_kv_head: Some(cfg.resolved_n_kv_head()),
            tokenizer_path,
            lora: None,
            lora_schema_version: None,
            fingerprint,
            use_rope: cfg.use_rope,
            rope_theta: cfg.rope_theta,
        }
    }

    fn from_nano_gpt_with_weights(
        cfg: &NanoGptConfig,
        tokenizer_path: Option<String>,
        model_file_blake3: &[u8; 32],
    ) -> Self {
        let mut m = Self::from_nano_gpt(cfg, tokenizer_path);
        m.fingerprint = fingerprint_checkpoint(cfg, m.tokenizer_path.as_deref(), model_file_blake3);
        m
    }

    pub fn to_nano_gpt_config(&self) -> NanoGptConfig {
        NanoGptConfig {
            vocab_size: self.vocab_size,
            block_size: self.block_size,
            n_layer: self.n_layer,
            n_head: self.n_head.max(1),
            n_kv_head: self.n_kv_head.unwrap_or(0),
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

fn hash_model_file_blake3(path: &Path) -> Result<[u8; 32]> {
    let mut f = BufReader::new(
        File::open(path).with_context(|| format!("open {}", path.display()))?,
    );
    let mut h = blake3::Hasher::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = f.read(&mut buf).with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(*h.finalize().as_bytes())
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
    let v: serde_json::Value = serde_json::from_slice(&body).context("parse manifest json")?;
    let has_manifest_version = v.get("manifest_version").is_some();
    let m: Manifest = serde_json::from_value(v).context("decode manifest fields")?;
    if !has_manifest_version {
        tracing::warn!(
            path = %path.display(),
            "checkpoint manifest has no manifest_version field; treating as legacy. Re-save with current flowgrid-cli to write manifest_version and a b3 fingerprint."
        );
    }
    if !m.fingerprint.starts_with("b3:") {
        tracing::warn!(
            path = %path.display(),
            fingerprint = %m.fingerprint,
            "checkpoint fingerprint is not b3-prefixed (content hash); legacy identity string. Re-save the checkpoint for a weight-inclusive BLAKE3 fingerprint."
        );
    }
    Ok(m)
}

/// Writes `model.bin` first, then `manifest.json` with a fingerprint that includes the on-disk weights digest.
/// `lora_sidecar` sets [`Manifest::lora`] and [`Manifest::lora_schema_version`] when non-empty (e.g. `"lora.json"`).
pub fn save_nano_gpt_checkpoint<B: Backend>(
    model: &NanoGpt<B>,
    dir: impl AsRef<Path>,
    cfg: &NanoGptConfig,
    tokenizer_path: Option<String>,
    lora_sidecar: Option<&str>,
) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let path = model_path(dir);
    let record = model.clone().into_record();
    BinFileRecorder::<FullPrecisionSettings>::default()
        .record(record, path.clone())
        .with_context(|| format!("write burn record {}", path.display()))?;

    let digest = hash_model_file_blake3(&path)?;
    let mut manifest = Manifest::from_nano_gpt_with_weights(cfg, tokenizer_path, &digest);
    if let Some(rel) = lora_sidecar {
        manifest.lora = Some(rel.to_string());
        manifest.lora_schema_version = Some(1);
    }
    save_manifest(dir, &manifest)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_manifest_warns_but_loads_legacy_file() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{
            "arch": "nanogpt",
            "dtype": "f32",
            "vocab_size": 10,
            "block_size": 8,
            "hidden": 16,
            "n_layer": 1,
            "n_head": 2,
            "n_kv_head": null,
            "tokenizer_path": null,
            "lora": null,
            "fingerprint": "legacy-no-b3",
            "use_rope": true,
            "rope_theta": 10000.0
        }"#;
        let path = dir.path().join("manifest.json");
        std::fs::write(&path, json).unwrap();
        let m = load_manifest(dir.path()).unwrap();
        assert_eq!(m.manifest_version, 1);
        assert_eq!(m.fingerprint, "legacy-no-b3");
    }

    #[test]
    fn manifest_deserializes_without_manifest_version() {
        let json = r#"{
            "arch": "nanogpt",
            "dtype": "f32",
            "vocab_size": 10,
            "block_size": 8,
            "hidden": 16,
            "n_layer": 1,
            "n_head": 2,
            "n_kv_head": null,
            "tokenizer_path": null,
            "lora": null,
            "fingerprint": "legacy",
            "use_rope": true,
            "rope_theta": 10000.0
        }"#;
        let m: Manifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.manifest_version, 1);
        assert_eq!(m.fingerprint, "legacy");
    }

    #[test]
    fn checkpoint_fingerprint_changes_with_weights() {
        let cfg = NanoGptConfig {
            vocab_size: 16,
            block_size: 8,
            n_layer: 1,
            n_head: 2,
            n_kv_head: 0,
            n_embd: 8,
            dropout: 0.0,
            use_rope: true,
            rope_theta: 10_000.0,
        };
        let a = fingerprint_checkpoint(&cfg, None, &[1u8; 32]);
        let b = fingerprint_checkpoint(&cfg, None, &[2u8; 32]);
        assert_ne!(a, b);
        assert!(a.starts_with("b3:"));
    }
}
