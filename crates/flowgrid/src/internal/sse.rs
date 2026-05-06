#[cfg(any(feature = "openai", feature = "anthropic"))]
pub(crate) mod common {
    use bytes::Bytes;
    use futures::Stream;
    use std::io;
    use tokio::io::AsyncBufReadExt;
    use tokio::io::BufReader;
    use tokio_util::io::StreamReader;

    /// One parsed SSE message block (terminated by a blank line).
    #[derive(Debug, Clone, PartialEq, Eq)]
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
        S: Stream<Item = std::result::Result<Bytes, io::Error>> + Unpin,
    {
        /// Wrap a body stream (`bytes_stream()` mapped to `io::Error`).
        pub fn new(stream: S) -> Self {
            Self {
                inner: BufReader::new(StreamReader::new(stream)),
                line: String::new(),
            }
        }

        /// Read the next complete SSE event (I/O errors only; see provider wrappers for API errors).
        pub async fn read_next_event(&mut self) -> io::Result<Option<SseEvent>> {
            let mut event = String::new();
            let mut data_lines: Vec<String> = Vec::new();
            let mut id: Option<String> = None;
            loop {
                self.line.clear();
                let n = self.inner.read_line(&mut self.line).await?;
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
                let line = line.trim_start();
                // SSE comment lines start with `:`.
                if line.starts_with(':') {
                    continue;
                }
                if let Some(rest) = line.strip_prefix("retry:") {
                    let _ = rest;
                    continue;
                }
                if let Some(rest) = line.strip_prefix("event:") {
                    event = rest.trim().to_string();
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data_lines.push(rest.trim_start().to_string());
                } else if let Some(rest) = line.strip_prefix("id:") {
                    id = Some(rest.trim().to_string());
                }
                // Unknown field names are ignored (per common SSE clients).
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

#[cfg(feature = "openai")]
pub mod oai {
    use bytes::Bytes;
    use futures::Stream;

    pub use super::common::SseEvent;

    use crate::internal::error::oai::Result;

    /// Incremental SSE decoder over a chunked byte stream.
    pub struct SseStream<S> {
        inner: super::common::SseStream<S>,
    }

    impl<S> SseStream<S>
    where
        S: Stream<Item = std::result::Result<Bytes, std::io::Error>> + Unpin,
    {
        /// Wrap a body stream (`bytes_stream()` mapped to `io::Error`).
        pub fn new(stream: S) -> Self {
            Self {
                inner: super::common::SseStream::new(stream),
            }
        }

        /// Read the next complete SSE event.
        pub async fn next_event(&mut self) -> Result<Option<SseEvent>> {
            self.inner
                .read_next_event()
                .await
                .map_err(|e| crate::internal::error::oai::Error::Sse(e.to_string()))
        }

        /// Turn this decoder into a [`futures::Stream`] of events (same errors as [`next_event`]).
        ///
        /// Prefer this when using `StreamExt::next()` / stream combinators instead of a manual loop.
        pub fn into_event_stream(self) -> impl futures::Stream<Item = Result<SseEvent>> + Send
        where
            S: Send + 'static,
        {
            futures::stream::try_unfold(self, |mut sse| async move {
                match sse.next_event().await {
                    Ok(Some(ev)) => Ok(Some((ev, sse))),
                    Ok(None) => Ok(None),
                    Err(e) => Err(e),
                }
            })
        }
    }
}

#[cfg(feature = "anthropic")]
pub mod clu {
    use bytes::Bytes;
    use futures::Stream;

    pub use super::common::SseEvent;

    use crate::internal::error::clu::Result;

    pub struct SseStream<S> {
        inner: super::common::SseStream<S>,
    }

    impl<S> SseStream<S>
    where
        S: Stream<Item = std::result::Result<Bytes, std::io::Error>> + Unpin,
    {
        pub fn new(stream: S) -> Self {
            Self {
                inner: super::common::SseStream::new(stream),
            }
        }

        pub async fn next_event(&mut self) -> Result<Option<SseEvent>> {
            self.inner
                .read_next_event()
                .await
                .map_err(|e| crate::internal::error::clu::Error::Sse(e.to_string()))
        }

        /// Turn this decoder into a [`futures::Stream`] of events (same errors as [`next_event`]).
        ///
        /// Semantics match the OpenAI helper of the same name when both providers are enabled.
        pub fn into_event_stream(self) -> impl futures::Stream<Item = Result<SseEvent>> + Send
        where
            S: Send + 'static,
        {
            futures::stream::try_unfold(self, |mut sse| async move {
                match sse.next_event().await {
                    Ok(Some(ev)) => Ok(Some((ev, sse))),
                    Ok(None) => Ok(None),
                    Err(e) => Err(e),
                }
            })
        }
    }
}

#[cfg(all(test, any(feature = "openai", feature = "anthropic")))]
mod sse_tests {
    use super::common::SseStream;
    use bytes::Bytes;
    use futures::stream;
    use std::io;

    fn chunked_body(parts: Vec<&'static str>) -> impl futures::Stream<Item = io::Result<Bytes>> {
        stream::iter(parts.into_iter().map(|s| Ok(Bytes::from(s))))
    }

    #[tokio::test]
    async fn sse_splits_across_chunks() {
        let s = chunked_body(vec!["da", "ta: {\"a\":1}", "\n", "\n"]);
        let mut dec = SseStream::new(s);
        let ev = dec.read_next_event().await.unwrap().unwrap();
        assert_eq!(ev.data, r#"{"a":1}"#);
        assert!(dec.read_next_event().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn sse_ignores_comment_and_retry_lines() {
        let raw = ":heartbeat\nretry: 3000\ndata: hello\n\n";
        let mut dec = SseStream::new(chunked_body(vec![raw]));
        let ev = dec.read_next_event().await.unwrap().unwrap();
        assert_eq!(ev.data, "hello");
        assert!(dec.read_next_event().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn sse_multiple_data_lines() {
        let raw = "event: x\ndata: line1\ndata: line2\n\n";
        let mut dec = SseStream::new(chunked_body(vec![raw]));
        let ev = dec.read_next_event().await.unwrap().unwrap();
        assert_eq!(ev.event, "x");
        assert_eq!(ev.data, "line1\nline2");
    }
}
