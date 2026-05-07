//! Hugging Face safetensors loader helpers for GPT-2 / Llama / Mistral / Qwen.

use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::{FgError, FgResult};
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::hf;

/// Parsed HF `config.json` subset for GPT-2 style models.
#[derive(Debug, Deserialize)]
pub struct Gpt2StyleConfigJson {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfArch {
    Gpt2,
    Llama,
    Mistral,
    Qwen2,
}

#[derive(Debug, Deserialize)]
pub struct HfConfigJson {
    pub architectures: Option<Vec<String>>,
}

pub fn infer_arch_from_config_text(text: &str) -> FgResult<HfArch> {
    let cfg: HfConfigJson =
        serde_json::from_str(text).map_err(|e| FgError::config(format!("config parse: {e}")))?;
    let arch = cfg
        .architectures
        .as_ref()
        .and_then(|v| v.first())
        .map(|s| s.to_lowercase())
        .ok_or_else(|| FgError::config("config.architectures[0] missing"))?;
    if arch.contains("gpt2") {
        Ok(HfArch::Gpt2)
    } else if arch.contains("mistral") {
        Ok(HfArch::Mistral)
    } else if arch.contains("qwen") {
        Ok(HfArch::Qwen2)
    } else if arch.contains("llama") {
        Ok(HfArch::Llama)
    } else {
        Err(FgError::config(format!("unsupported architecture: {arch}")))
    }
}

pub fn expected_keys_for_arch(arch: HfArch) -> Vec<&'static str> {
    match arch {
        HfArch::Gpt2 => hf::gpt2::expected_keys(),
        HfArch::Llama => hf::llama::expected_keys(),
        HfArch::Mistral => hf::mistral::expected_keys(),
        HfArch::Qwen2 => hf::qwen::expected_keys(),
    }
}

pub fn validate_expected_keys(tensors: &HashMap<String, Vec<u8>>, arch: HfArch) -> FgResult<()> {
    let expected = expected_keys_for_arch(arch);
    let missing: Vec<&str> = expected
        .iter()
        .copied()
        .filter(|k| !tensors.contains_key(*k))
        .collect();
    if !missing.is_empty() {
        let sample_available = tensors
            .keys()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        return Err(FgError::shape(format!(
            "missing HF tensor keys [{}] (expected={}, found={}); sample available: {}",
            missing.join(", "),
            expected.len(),
            tensors.len(),
            sample_available
        )));
    }
    Ok(())
}

fn safetensors_dtype_tag(tv: &TensorView<'_>) -> String {
    use safetensors::Dtype;
    match tv.dtype() {
        Dtype::F32 => "F32".to_string(),
        Dtype::BF16 => "BF16".to_string(),
        Dtype::F16 => "F16".to_string(),
        Dtype::U8 => "U8".to_string(),
        Dtype::I8 => "I8".to_string(),
        Dtype::U16 => "U16".to_string(),
        Dtype::I16 => "I16".to_string(),
        Dtype::I32 => "I32".to_string(),
        Dtype::I64 => "I64".to_string(),
        Dtype::F64 => "F64".to_string(),
        Dtype::BOOL => "BOOL".to_string(),
        _ => "OTHER".to_string(),
    }
}

/// Dtype tag + shape + raw tensor bytes as stored in `.safetensors`.
pub type SafetensorsTensorRecord = (String, Vec<usize>, Vec<u8>);

/// Load tensors with dtype + shape metadata (for GPT-2 and other HF loaders).
pub fn load_safetensors_typed(
    path: impl AsRef<Path>,
) -> FgResult<HashMap<String, SafetensorsTensorRecord>> {
    let bytes = std::fs::read(path.as_ref()).map_err(|e| FgError::io(e.to_string()))?;
    let st = SafeTensors::deserialize(&bytes).map_err(|e| FgError::io(e.to_string()))?;
    let mut out = HashMap::new();
    for name in st.names() {
        let view = st.tensor(name).map_err(|e| FgError::io(e.to_string()))?;
        let dtype = safetensors_dtype_tag(&view);
        let shape = view.shape().to_vec();
        let data = view.data().to_vec();
        out.insert(name.to_string(), (dtype, shape, data));
    }
    Ok(out)
}

/// Load named tensors from a `.safetensors` file into host memory (CPU bytes).
pub fn load_safetensors_bytes(path: impl AsRef<Path>) -> FgResult<HashMap<String, Vec<u8>>> {
    let bytes = std::fs::read(path.as_ref()).map_err(|e| FgError::io(e.to_string()))?;
    let st = SafeTensors::deserialize(&bytes).map_err(|e| FgError::io(e.to_string()))?;
    let mut out = HashMap::new();
    for name in st.names() {
        let view = st.tensor(name).map_err(|e| FgError::io(e.to_string()))?;
        let data = view.data().to_vec();
        out.insert(name.to_string(), data);
    }
    Ok(out)
}

/// Placeholder: convert raw float tensor bytes into a Burn tensor on device `device`.
pub fn raw_f32_tensor<B: Backend>(
    data: &[u8],
    shape: [usize; 2],
    device: &B::Device,
) -> FgResult<Tensor<B, 2>> {
    let expected = shape[0] * shape[1] * 4;
    if data.len() != expected {
        return Err(FgError::shape("byte length mismatch for f32 tensor"));
    }
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| {
            let arr = [c[0], c[1], c[2], c[3]];
            f32::from_le_bytes(arr)
        })
        .collect();
    Ok(Tensor::<B, 1>::from_floats(floats.as_slice(), device).reshape(shape))
}

pub fn bf16_bytes_to_f32(data: &[u8]) -> FgResult<Vec<f32>> {
    if data.len() % 2 != 0 {
        return Err(FgError::shape("bf16 byte length must be even"));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for c in data.chunks_exact(2) {
        let hi = u16::from_le_bytes([c[0], c[1]]) as u32;
        out.push(f32::from_bits(hi << 16));
    }
    Ok(out)
}

pub fn fp16_bytes_to_f32(data: &[u8]) -> FgResult<Vec<f32>> {
    if data.len() % 2 != 0 {
        return Err(FgError::shape("fp16 byte length must be even"));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for c in data.chunks_exact(2) {
        let half = u16::from_le_bytes([c[0], c[1]]);
        let sign = ((half >> 15) & 0x1) as u32;
        let exp = ((half >> 10) & 0x1f) as u32;
        let frac = (half & 0x03ff) as u32;
        let f32_bits = if exp == 0 {
            if frac == 0 {
                sign << 31
            } else {
                let mut e = -14i32;
                let mut m = frac;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03ff;
                (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
            }
        } else if exp == 0x1f {
            (sign << 31) | 0x7f80_0000 | (frac << 13)
        } else {
            (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13)
        };
        out.push(f32::from_bits(f32_bits));
    }
    Ok(out)
}
