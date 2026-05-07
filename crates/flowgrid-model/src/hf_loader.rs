//! Hugging Face **`safetensors`** loader scaffold (GPT-2 family).

use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::{FgError, FgResult};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Parsed HF `config.json` subset for GPT-2 style models.
#[derive(Debug, Deserialize)]
pub struct Gpt2StyleConfigJson {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
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
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(Tensor::<B, 1>::from_floats(floats.as_slice(), device).reshape(shape))
}
