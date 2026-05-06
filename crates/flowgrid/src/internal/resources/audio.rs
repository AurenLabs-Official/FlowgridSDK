use crate::internal::oai::OpenAI;
use crate::internal::oai::Result;
use reqwest::multipart::Form;
use serde_json::Value;

/// Audio API (`client.audio`, feature `audio`).
pub struct AudioClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> AudioClient<'a> {
    pub(crate) fn new(inner: &'a OpenAI) -> Self {
        Self { inner }
    }

    /// `client.audio.transcriptions`
    pub fn transcriptions(&self) -> TranscriptionsClient<'_> {
        TranscriptionsClient { inner: self.inner }
    }

    /// `client.audio.translations`
    pub fn translations(&self) -> TranslationsClient<'_> {
        TranslationsClient { inner: self.inner }
    }

    /// `client.audio.speech`
    pub fn speech(&self) -> SpeechClient<'_> {
        SpeechClient { inner: self.inner }
    }
}

/// Transcriptions sub-resource.
pub struct TranscriptionsClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> TranscriptionsClient<'a> {
    /// `POST /audio/transcriptions`
    pub async fn create(&self, form: Form) -> Result<Value> {
        let (v, _) = self
            .inner
            .transport
            .post_multipart_json("audio/transcriptions", form)
            .await?;
        Ok(v)
    }
}

/// Translations sub-resource.
pub struct TranslationsClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> TranslationsClient<'a> {
    /// `POST /audio/translations`
    pub async fn create(&self, form: Form) -> Result<Value> {
        let (v, _) = self
            .inner
            .transport
            .post_multipart_json("audio/translations", form)
            .await?;
        Ok(v)
    }
}

/// Speech synthesis sub-resource.
pub struct SpeechClient<'a> {
    inner: &'a OpenAI,
}

impl<'a> SpeechClient<'a> {
    /// `POST /audio/speech`
    pub async fn create(&self, body: &Value) -> Result<Vec<u8>> {
        let (bytes, _) = self
            .inner
            .transport
            .post_json_bytes("audio/speech", body)
            .await?;
        Ok(bytes)
    }
}
