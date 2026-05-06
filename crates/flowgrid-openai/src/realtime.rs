//! Realtime WebSocket helper (feature `realtime`).
//!
//! Establishes a `wss://api.openai.com/v1/realtime` connection with the same authentication style
//! as HTTP requests.

use crate::client::OpenAI;
use crate::error::{Error, Result};
use http::header::{HeaderValue, AUTHORIZATION};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::MaybeTlsStream;
use tokio_tungstenite::WebSocketStream;

/// Connected realtime socket.
pub type RealtimeSocket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// Connect to the Realtime API using an [`OpenAI`] API key.
pub async fn connect(client: &OpenAI, model: &str) -> Result<RealtimeSocket> {
    let mut url = url::Url::parse("wss://api.openai.com/v1/realtime").map_err(Error::Url)?;
    url.query_pairs_mut().append_pair("model", model);
    let mut req = url
        .as_str()
        .into_client_request()
        .map_err(Error::Ws)?;
    let key = format!("Bearer {}", client.transport.config.api_key);
    {
        let headers = req.headers_mut();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&key).map_err(|e| Error::Config(e.to_string()))?,
        );
        headers.insert(
            http::HeaderName::from_static("openai-beta"),
            HeaderValue::from_static("realtime=v1"),
        );
    }
    let (ws, _) = connect_async(req).await.map_err(Error::Ws)?;
    Ok(ws)
}
