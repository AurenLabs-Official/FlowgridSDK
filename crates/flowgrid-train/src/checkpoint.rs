//! Checkpoint helpers (Burn recorders).

use burn::record::{BinBytesRecorder, FullPrecisionSettings};

pub fn recorder_settings() -> FullPrecisionSettings {
    FullPrecisionSettings
}

pub type BytesRecorder = BinBytesRecorder<FullPrecisionSettings>;
