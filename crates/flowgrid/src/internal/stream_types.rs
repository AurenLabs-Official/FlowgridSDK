//! Shared streaming primitives (HTTP body chunks, SSE over bytes).

use bytes::Bytes;
use futures::Stream;
use std::pin::Pin;
use std::result::Result as StdResult;

/// Boxed byte stream from the HTTP client (e.g. `post_stream_bytes`).
pub type BoxedByteStream = Pin<Box<dyn Stream<Item = StdResult<Bytes, std::io::Error>> + Send>>;
