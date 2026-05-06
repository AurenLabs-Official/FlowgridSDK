//! Cooperative stream cancellation using [`tokio_util::sync::CancellationToken`].

use futures::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

/// Waits for either the next item from `stream` or for `cancel` to be triggered.
///
/// This complements [`ExecuteOptions::timeout`](crate::ExecuteOptions), which bounds how long the
/// remote may work, but does not replace app-level **shutdown** where you want to stop reading
/// SSE chunks promptly.
pub async fn stream_next_until_cancelled<S, T>(
    stream: &mut S,
    cancel: &CancellationToken,
) -> Option<T>
where
    S: Stream<Item = T> + Unpin,
{
    tokio::select! {
        _ = cancel.cancelled() => None,
        item = stream.next() => item,
    }
}
