use burn::backend::NdArray;
use flowgrid_model::hf::llama::decode_self_attn_q_proj;

type B = NdArray<f32>;

#[test]
fn llama_q_proj_decodes_f32_square() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    let n = 4usize;
    let mut vals: Vec<f32> = Vec::with_capacity(n * n);
    for i in 0..(n * n) {
        vals.push(i as f32 * 0.25);
    }
    let mut bytes = Vec::with_capacity(n * n * 4);
    for v in &vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let t = decode_self_attn_q_proj::<B>("F32", &bytes, n, &device).unwrap();
    assert_eq!(t.dims(), [n, n]);
}
