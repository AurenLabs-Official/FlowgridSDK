//! GPT-2 → [`NanoGpt`] weight mapping (HF `safetensors`, Conv1D layout: weights are `[nx, nf]`,
//! e.g. fused QKV is `[d_model, 3 * d_model]` — not row-stacked `[3d, d]`).

use std::collections::HashMap;

use burn::tensor::{backend::Backend, Tensor};
use flowgrid_tensor::{FgError, FgResult};

use crate::nano_gpt::NanoGpt;

fn vec_f32_from_tensor_bytes(dtype: &str, data: &[u8]) -> FgResult<Vec<f32>> {
    match dtype {
        "F32" => {
            if data.len() % 4 != 0 {
                return Err(FgError::shape("f32 tensor byte length"));
            }
            Ok(data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        "BF16" => bf16_bytes_to_f32_slice(data),
        "F16" => fp16_bytes_to_f32_slice(data),
        other => Err(FgError::config(format!("unsupported dtype {other}"))),
    }
}

fn bf16_bytes_to_f32_slice(data: &[u8]) -> FgResult<Vec<f32>> {
    if data.len() % 2 != 0 {
        return Err(FgError::shape("bf16 tensor byte length"));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for c in data.chunks_exact(2) {
        let hi = u16::from_le_bytes([c[0], c[1]]) as u32;
        out.push(f32::from_bits(hi << 16));
    }
    Ok(out)
}

fn fp16_bytes_to_f32_slice(data: &[u8]) -> FgResult<Vec<f32>> {
    if data.len() % 2 != 0 {
        return Err(FgError::shape("fp16 tensor byte length"));
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

pub fn tensor_f2_from_bytes<B: Backend>(
    name: &str,
    dtype: &str,
    data: &[u8],
    shape: &[usize],
    device: &B::Device,
) -> FgResult<Tensor<B, 2>> {
    let flat = vec_f32_from_tensor_bytes(dtype, data)?;
    let expected: usize = shape.iter().product();
    if flat.len() != expected {
        return Err(FgError::shape(format!(
            "tensor {name}: expected {expected} elems, got {}",
            flat.len()
        )));
    }
    let s0 = *shape.first().unwrap_or(&0);
    let s1 = *shape.get(1).unwrap_or(&0);
    if shape.len() != 2 {
        return Err(FgError::shape(format!("tensor {name}: need rank-2 view")));
    }
    Ok(Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([s0, s1]))
}

pub fn tensor_f1_from_bytes<B: Backend>(
    name: &str,
    dtype: &str,
    data: &[u8],
    len: usize,
    device: &B::Device,
) -> FgResult<Tensor<B, 1>> {
    let flat = vec_f32_from_tensor_bytes(dtype, data)?;
    if flat.len() != len {
        return Err(FgError::shape(format!(
            "tensor {name}: expected len {len}, got {}",
            flat.len()
        )));
    }
    Ok(Tensor::<B, 1>::from_floats(flat.as_slice(), device))
}

/// Tensor names for validating a 1-block GPT-2-compatible checkpoint.
pub fn tensor_names_for_layers(n_layer: usize) -> Vec<String> {
    let mut v = vec![
        "transformer.wte.weight".to_string(),
        "transformer.wpe.weight".to_string(),
        "lm_head.weight".to_string(),
    ];
    for i in 0..n_layer {
        v.push(format!("transformer.h.{i}.attn.c_attn.weight"));
        v.push(format!("transformer.h.{i}.attn.c_attn.bias"));
        v.push(format!("transformer.h.{i}.attn.c_proj.weight"));
        v.push(format!("transformer.h.{i}.attn.c_proj.bias"));
        v.push(format!("transformer.h.{i}.mlp.c_fc.weight"));
        v.push(format!("transformer.h.{i}.mlp.c_fc.bias"));
        v.push(format!("transformer.h.{i}.mlp.c_proj.weight"));
        v.push(format!("transformer.h.{i}.mlp.c_proj.bias"));
    }
    v
}

/// Load core GPT-2 projection weights into [`NanoGpt`] configured with `use_rope: false`.
/// LayerNorm parameters stay at init values (HF `ln_*` mapping is a follow-up).
pub fn load_gpt2_into_nano_gpt<B: Backend>(
    model: &mut NanoGpt<B>,
    named: &HashMap<String, (String, Vec<usize>, Vec<u8>)>,
    n_embd: usize,
    n_head: usize,
    device: &B::Device,
) -> FgResult<()> {
    let n_layer = model.blocks.len();
    let d = n_embd;
    if (d / n_head.max(1)) * n_head.max(1) != d {
        return Err(FgError::config("n_embd must be divisible by n_head"));
    }

    let wte = named
        .get("transformer.wte.weight")
        .ok_or_else(|| FgError::shape("missing transformer.wte.weight"))?;
    let w_tok = tensor_f2_from_bytes::<B>("wte", &wte.0, &wte.2, &wte.1, device)?;
    model.tok_emb.weight = model.tok_emb.weight.clone().map(|_| w_tok.clone());

    let wpe = named
        .get("transformer.wpe.weight")
        .ok_or_else(|| FgError::shape("missing transformer.wpe.weight"))?;
    let w_pos = tensor_f2_from_bytes::<B>("wpe", &wpe.0, &wpe.2, &wpe.1, device)?;
    model.pos_emb.weight = model.pos_emb.weight.clone().map(|_| w_pos.clone());

    let vocab = model.tok_emb.weight.dims()[0];
    let lm = named
        .get("lm_head.weight")
        .ok_or_else(|| FgError::shape("missing lm_head.weight"))?;
    let w_lm = tensor_f2_from_bytes::<B>("lm_head", &lm.0, &lm.2, &lm.1, device)?;
    let w_lm = match (lm.1.first().copied(), lm.1.get(1).copied()) {
        (Some(a), Some(b)) if a == d && b == vocab => w_lm,
        (Some(a), Some(b)) if a == vocab && b == d => w_lm.transpose(),
        _ => {
            return Err(FgError::shape(format!(
                "lm_head.weight: expected `[{}, {}]` or transposed, got {:?}",
                d, vocab, lm.1
            )));
        }
    };
    model.head.weight = model.head.weight.clone().map(|_| w_lm.clone());

    for i in 0..n_layer {
        let blk = &mut model.blocks[i];
        let attn = &mut blk.attn;
        let cq = named
            .get(&format!("transformer.h.{i}.attn.c_attn.weight"))
            .ok_or_else(|| FgError::shape("c_attn.weight"))?;
        let cb = named
            .get(&format!("transformer.h.{i}.attn.c_attn.bias"))
            .ok_or_else(|| FgError::shape("c_attn.bias"))?;
        // HF `Conv1D` stores `weight` as `[nx, nf]` (input dim × output dim). Combined QKV uses
        // `nf = 3 * d`, `nx = d` → `[d, 3 * d]`; Q/K/V are column blocks (each `[d, d]`).
        let wf = tensor_f2_from_bytes::<B>("c_attn_w", &cq.0, &cq.2, &cq.1, device)?;
        let b_all = tensor_f1_from_bytes::<B>("c_attn_b", &cb.0, &cb.2, 3 * d, device)?;

        let wq = wf.clone().narrow(1, 0, d);
        let wk = wf.clone().narrow(1, d, d);
        let wv = wf.narrow(1, 2 * d, d);
        let bq = b_all.clone().narrow(0, 0, d);
        let bk = b_all.clone().narrow(0, d, d);
        let bv = b_all.narrow(0, 2 * d, d);

        attn.q_proj.base.weight = attn.q_proj.base.weight.clone().map(|_| wq.clone());
        attn.k_proj.base.weight = attn.k_proj.base.weight.clone().map(|_| wk.clone());
        attn.v_proj.base.weight = attn.v_proj.base.weight.clone().map(|_| wv.clone());
        if let Some(b) = attn.q_proj.base.bias.as_mut() {
            *b = b.clone().map(|_| bq.clone());
        }
        if let Some(b) = attn.k_proj.base.bias.as_mut() {
            *b = b.clone().map(|_| bk.clone());
        }
        if let Some(b) = attn.v_proj.base.bias.as_mut() {
            *b = b.clone().map(|_| bv.clone());
        }

        let cpw = named
            .get(&format!("transformer.h.{i}.attn.c_proj.weight"))
            .ok_or_else(|| FgError::shape("c_proj.weight"))?;
        let cpb = named
            .get(&format!("transformer.h.{i}.attn.c_proj.bias"))
            .ok_or_else(|| FgError::shape("c_proj.bias"))?;
        let wcp = tensor_f2_from_bytes::<B>("c_proj_w", &cpw.0, &cpw.2, &cpw.1, device)?;
        let bcp = tensor_f1_from_bytes::<B>("c_proj_b", &cpb.0, &cpb.2, d, device)?;
        attn.o_proj.base.weight = attn.o_proj.base.weight.clone().map(|_| wcp.clone());
        if let Some(b) = attn.o_proj.base.bias.as_mut() {
            *b = b.clone().map(|_| bcp.clone());
        }

        let mlp = &mut blk.mlp;
        let fcw = named
            .get(&format!("transformer.h.{i}.mlp.c_fc.weight"))
            .ok_or_else(|| FgError::shape("c_fc.weight"))?;
        let fcb = named
            .get(&format!("transformer.h.{i}.mlp.c_fc.bias"))
            .ok_or_else(|| FgError::shape("c_fc.bias"))?;
        // `mlp.c_fc`: Conv1D `nf = 4 * d`, `nx = d` → `[d, 4 * d]` matches Burn `Linear(d, 4d)`.
        let wfc = tensor_f2_from_bytes::<B>("c_fc_w", &fcw.0, &fcw.2, &fcw.1, device)?;
        let bfc = tensor_f1_from_bytes::<B>("c_fc_b", &fcb.0, &fcb.2, 4 * d, device)?;
        mlp.up.base.weight = mlp.up.base.weight.clone().map(|_| wfc.clone());
        if let Some(b) = mlp.up.base.bias.as_mut() {
            *b = b.clone().map(|_| bfc.clone());
        }

        let prw = named
            .get(&format!("transformer.h.{i}.mlp.c_proj.weight"))
            .ok_or_else(|| FgError::shape("mlp.c_proj.weight"))?;
        let prb = named
            .get(&format!("transformer.h.{i}.mlp.c_proj.bias"))
            .ok_or_else(|| FgError::shape("mlp.c_proj.bias"))?;
        // `mlp.c_proj`: Conv1D `nf = d`, `nx = 4 * d` → `[4 * d, d]` matches Burn `Linear(4d, d)`.
        let wpr = tensor_f2_from_bytes::<B>("mlp_proj_w", &prw.0, &prw.2, &prw.1, device)?;
        let bpr = tensor_f1_from_bytes::<B>("mlp_proj_b", &prb.0, &prb.2, d, device)?;
        mlp.down.base.weight = mlp.down.base.weight.clone().map(|_| wpr.clone());
        if let Some(b) = mlp.down.base.bias.as_mut() {
            *b = b.clone().map(|_| bpr.clone());
        }
    }
    Ok(())
}

pub fn expected_keys() -> Vec<&'static str> {
    vec![
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "lm_head.weight",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_attn.bias",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.attn.c_proj.bias",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_fc.bias",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.h.0.mlp.c_proj.bias",
    ]
}
