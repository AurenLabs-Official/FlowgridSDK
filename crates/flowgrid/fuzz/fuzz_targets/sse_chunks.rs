#![no_main]

use libfuzzer_sys::fuzz_target;

/// Split arbitrary bytes into random-ish chunks for incremental SSE decoding.
fn chunk_payload(data: &[u8]) -> Vec<Vec<u8>> {
    if data.is_empty() {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < data.len() {
        let stride = (data[i] as usize % 24).max(1);
        let end = (i + stride).min(data.len());
        out.push(data[i..end].to_vec());
        i = end;
    }
    out
}

fuzz_target!(|data: &[u8]| {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let _ = rt.block_on(flowgrid::sse_fuzz_support::decode_sse_event_count(chunk_payload(
        data,
    )));
});
