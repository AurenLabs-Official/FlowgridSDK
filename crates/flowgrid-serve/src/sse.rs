pub fn frame(data: &str) -> String {
    format!("data: {data}\n\n")
}

/// SSE frame with explicit `event:` line (before `data:`), then blank line terminator.
pub fn frame_event(event: &str, data: &str) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

pub fn done() -> String {
    "data: [DONE]\n\n".to_string()
}
