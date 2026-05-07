//! Apply [`LoraSpec`] to a [`NanoGpt`] after init or checkpoint load.

use burn::nn::Linear;
use burn::tensor::backend::Backend;

use crate::attention::CausalSelfAttn;
use crate::lora::{LoraLinear, LoraLinearConfig, LoraSpec, LoraTarget};
use crate::mlp::Mlp;
use crate::nano_gpt::{Block, NanoGpt};

/// Wrap [`LoraLinear`] layers marked in `spec` with a fresh low-r adapter; other layers keep their
/// (possibly zero-contribution) adapters unchanged.
pub fn attach_lora<B: Backend>(model: NanoGpt<B>, spec: &LoraSpec) -> NanoGpt<B> {
    let device = model.tok_emb.weight.val().device();
    let lc = LoraLinearConfig {
        r: spec.r.max(1),
        alpha: spec.alpha,
    };
    let blocks = model
        .blocks
        .into_iter()
        .map(|blk| Block {
            attn: attach_attn::<B>(blk.attn, spec, &lc, &device),
            mlp: attach_mlp::<B>(blk.mlp, spec, &lc, &device),
            pre_norm: blk.pre_norm,
            post_norm: blk.post_norm,
        })
        .collect();
    NanoGpt {
        tok_emb: model.tok_emb,
        pos_emb: model.pos_emb,
        blocks,
        head: model.head,
    }
}

fn attach_attn<B: Backend>(
    attn: CausalSelfAttn<B>,
    spec: &LoraSpec,
    lc: &LoraLinearConfig,
    device: &B::Device,
) -> CausalSelfAttn<B> {
    CausalSelfAttn {
        q_proj: maybe_rewrap(attn.q_proj, spec, LoraTarget::Q, lc, device),
        k_proj: maybe_rewrap(attn.k_proj, spec, LoraTarget::K, lc, device),
        v_proj: maybe_rewrap(attn.v_proj, spec, LoraTarget::V, lc, device),
        o_proj: maybe_rewrap(attn.o_proj, spec, LoraTarget::O, lc, device),
        n_head: attn.n_head,
        n_kv_head: attn.n_kv_head,
        d_k: attn.d_k,
        use_rope: attn.use_rope,
        rope_theta: attn.rope_theta,
        max_seq: attn.max_seq,
    }
}

fn attach_mlp<B: Backend>(
    mlp: Mlp<B>,
    spec: &LoraSpec,
    lc: &LoraLinearConfig,
    device: &B::Device,
) -> Mlp<B> {
    Mlp {
        up: maybe_rewrap(mlp.up, spec, LoraTarget::Up, lc, device),
        down: maybe_rewrap(mlp.down, spec, LoraTarget::Down, lc, device),
        dropout: mlp.dropout,
    }
}

fn maybe_rewrap<B: Backend>(
    layer: LoraLinear<B>,
    spec: &LoraSpec,
    target: LoraTarget,
    lc: &LoraLinearConfig,
    device: &B::Device,
) -> LoraLinear<B> {
    if spec.targets.contains(&target) {
        let base: Linear<B> = layer.merged_linear();
        lc.init(base, device)
    } else {
        layer
    }
}
