//! Multipart helpers for `files`, `images`, and `audio` uploads.

use crate::error::{Error, Result};
use reqwest::multipart::Part;
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

/// Build a file part from in-memory bytes.
pub fn part_from_bytes(
    filename: impl Into<String>,
    content_type: &str,
    data: Vec<u8>,
) -> Result<Part> {
    Part::bytes(data)
        .file_name(filename.into())
        .mime_str(content_type)
        .map_err(|e| Error::Config(e.to_string()))
}

/// Build a file part by reading a local path (async).
pub async fn part_from_path(
    path: impl AsRef<Path>,
    filename: Option<String>,
    content_type: &str,
) -> std::result::Result<Part, std::io::Error> {
    let path = path.as_ref();
    let mut file = File::open(path).await?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).await?;
    let name = filename.unwrap_or_else(|| {
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("upload.bin")
            .to_string()
    });
    Part::bytes(buf)
        .file_name(name)
        .mime_str(content_type)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
