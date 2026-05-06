#[cfg(feature = "openai")]
pub mod oai {
    use bytes::Bytes;
    use futures::Stream;
    use tokio::io::AsyncBufReadExt;
    use tokio::io::BufReader;
    use tokio_util::io::StreamReader;

    use crate::internal::error::oai::Result;

    /// One parsed SSE message block (terminated by a blank line).
    #[derive(Debug, Clone)]
    pub struct SseEvent {
        /// `event:` field when present; empty if omitted.
        pub event: String,
        /// `data:` payload (joined if multiple lines).
        pub data: String,
        /// `id:` when present.
        pub id: Option<String>,
    }

    /// Incremental SSE decoder over a chunked byte stream.
    pub struct SseStream<S> {
        inner: BufReader<StreamReader<S, Bytes>>,
        line: String,
    }

    impl<S> SseStream<S>
    where
        S: Stream<Item = std::result::Result<Bytes, std::io::Error>> + Unpin,
    {
        /// Wrap a body stream (`bytes_stream()` mapped to `io::Error`).
        pub fn new(stream: S) -> Self {
            Self {
                inner: BufReader::new(StreamReader::new(stream)),
                line: String::new(),
            }
        }

        /// Read the next complete SSE event.
        pub async fn next_event(&mut self) -> Result<Option<SseEvent>> {
            let mut event = String::new();
            let mut data_lines: Vec<String> = Vec::new();
            let mut id: Option<String> = None;
            loop {
                self.line.clear();
                let n = self
                    .inner
                    .read_line(&mut self.line)
                    .await
                    .map_err(|e| crate::internal::error::oai::Error::Sse(e.to_string()))?;
                if n == 0 {
                    if event.is_empty() && data_lines.is_empty() && id.is_none() {
                        return Ok(None);
                    }
                    break;
                }
                let line = self.line.trim_end_matches(['\r', '\n']);
                if line.is_empty() {
                    if event.is_empty() && data_lines.is_empty() && id.is_none() {
                        continue;
                    }
                    break;
                }
                if let Some(rest) = line.strip_prefix("event:") {
                    event = rest.trim().to_string();
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data_lines.push(rest.trim_start().to_string());
                } else if let Some(rest) = line.strip_prefix("id:") {
                    id = Some(rest.trim().to_string());
                }
            }
            if data_lines.is_empty() && event.is_empty() && id.is_none() {
                return Ok(None);
            }
            Ok(Some(SseEvent {
                event,
                data: data_lines.join("\n"),
                id,
            }))
        }
    }
}

#[cfg(feature = "anthropic")]
pub mod clu {
    use bytes::Bytes;
    use futures::Stream;
    use tokio::io::AsyncBufReadExt;
    use tokio::io::BufReader;
    use tokio_util::io::StreamReader;

    use crate::internal::error::clu::Result;

    #[derive(Debug, Clone)]
    pub struct SseEvent {
        pub event: String,
        pub data: String,
        pub id: Option<String>,
    }

    pub struct SseStream<S> {
        inner: BufReader<StreamReader<S, Bytes>>,
        line: String,
    }

    impl<S> SseStream<S>
    where
        S: Stream<Item = std::result::Result<Bytes, std::io::Error>> + Unpin,
    {
        pub fn new(stream: S) -> Self {
            Self {
                inner: BufReader::new(StreamReader::new(stream)),
                line: String::new(),
            }
        }

        pub async fn next_event(&mut self) -> Result<Option<SseEvent>> {
            let mut event = String::new();
            let mut data_lines: Vec<String> = Vec::new();
            let mut id: Option<String> = None;
            loop {
                self.line.clear();
                let n = self
                    .inner
                    .read_line(&mut self.line)
                    .await
                    .map_err(|e| crate::internal::error::clu::Error::Sse(e.to_string()))?;
                if n == 0 {
                    if event.is_empty() && data_lines.is_empty() && id.is_none() {
                        return Ok(None);
                    }
                    break;
                }
                let line = self.line.trim_end_matches(['\r', '\n']);
                if line.is_empty() {
                    if event.is_empty() && data_lines.is_empty() && id.is_none() {
                        continue;
                    }
                    break;
                }
                if let Some(rest) = line.strip_prefix("event:") {
                    event = rest.trim().to_string();
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data_lines.push(rest.trim_start().to_string());
                } else if let Some(rest) = line.strip_prefix("id:") {
                    id = Some(rest.trim().to_string());
                }
            }
            if data_lines.is_empty() && event.is_empty() && id.is_none() {
                return Ok(None);
            }
            Ok(Some(SseEvent {
                event,
                data: data_lines.join("\n"),
                id,
            }))
        }
    }
}
