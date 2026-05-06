//! Criterion benchmarks for JSON deserialize and SSE framing (no network).

use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use futures::stream;

fn bench_deserialize_contracts(c: &mut Criterion) {
    #[cfg(feature = "openai")]
    {
        let raw =
            include_str!("../tests/fixtures/contracts/openai_chat_completion_v1_deserialize.json");
        c.bench_function("deserialize_openai_chat_completion_contract", |b| {
            b.iter(|| {
                let v: flowgrid::ChatCompletion =
                    serde_json::from_str(black_box(raw)).expect("parse");
                black_box(v.message_content());
            });
        });
    }
    #[cfg(feature = "anthropic")]
    {
        let raw = include_str!("../tests/fixtures/contracts/anthropic_message_v1_deserialize.json");
        c.bench_function("deserialize_anthropic_message_contract", |b| {
            b.iter(|| {
                let v: flowgrid::Message = serde_json::from_str(black_box(raw)).expect("parse");
                black_box(v.text_concat());
            });
        });
    }
}

fn bench_sse_next_event(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let payload = black_box("event: delta\ndata: {\"chunk\":1}\n\n");

    #[cfg(feature = "openai")]
    c.bench_function("sse_openai_next_event_small", |b| {
        b.to_async(&rt).iter(|| async {
            let stream = stream::iter(vec![Ok(Bytes::copy_from_slice(payload.as_bytes()))]);
            let mut dec = flowgrid::SseStream::new(stream);
            let ev = dec.next_event().await.expect("sse err").expect("event");
            black_box(ev.data.len());
        });
    });

    #[cfg(feature = "anthropic")]
    c.bench_function("sse_anthropic_next_event_small", |b| {
        b.to_async(&rt).iter(|| async {
            let stream = stream::iter(vec![Ok(Bytes::copy_from_slice(payload.as_bytes()))]);
            let mut dec = flowgrid::SseStream::new(stream);
            let ev = dec.next_event().await.expect("sse err").expect("event");
            black_box(ev.data.len());
        });
    });
}

criterion_group!(benches, bench_deserialize_contracts, bench_sse_next_event);
criterion_main!(benches);
