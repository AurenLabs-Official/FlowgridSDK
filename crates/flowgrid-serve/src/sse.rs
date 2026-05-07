pub fn frame(data: &str) -> String {
    format!("data: {data}\n\n")
}

pub fn done() -> String {
    "data: [DONE]\n\n".to_string()
}
