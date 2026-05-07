#[macro_export]
macro_rules! fg_span {
    ($name:expr, run_id=$rid:expr, step=$step:expr) => {
        tracing::info_span!(
            $name,
            run_id = %$rid,
            step = $step as u64,
            tokens_per_sec = tracing::field::Empty,
            gpu_mem_mb = tracing::field::Empty,
            phase = tracing::field::Empty
        )
    };
}
