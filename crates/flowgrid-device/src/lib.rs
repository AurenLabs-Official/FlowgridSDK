//! Parse **`FLOWGRID_DEVICE`** for preview LLM binaries (`flowgrid-llm`, `flowgrid-serve`).
//!
//! # Values (case-insensitive)
//!
//! - `cpu` — use NdArray CPU (default when the binary is built without GPU features).
//! - `wgpu`, `gpu`, `best` — prefer the Burn wgpu backend when built with `--features gpu-wgpu`.
//! - `wgpu:0`, `discrete:0` — first discrete GPU index for wgpu.
//! - `integrated:0` — integrated GPU index.
//!
//! Unknown or malformed values fall back to **`cpu`** (silent in this crate; binaries may log).
//!
//! **MSRV:** This crate stays dependency-free so `flowgrid` stable SDK is unaffected.

/// Normalized device request from the environment.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceRequest {
    /// NdArray CPU backend.
    #[default]
    Cpu,
    /// Burn wgpu: best available adapter.
    WgpuBest,
    /// Burn wgpu: discrete adapter index.
    WgpuDiscrete(usize),
    /// Burn wgpu: integrated adapter index.
    WgpuIntegrated(usize),
}

impl DeviceRequest {
    /// Read `FLOWGRID_DEVICE` or default **`cpu`**.
    pub fn from_env() -> Self {
        let Ok(raw) = std::env::var("FLOWGRID_DEVICE") else {
            return Self::Cpu;
        };
        Self::parse(&raw)
    }

    /// Parse a single token (e.g. from CLI flag tests).
    pub fn parse(raw: &str) -> Self {
        let s = raw.trim();
        if s.is_empty() {
            return Self::Cpu;
        }
        let lower = s.to_lowercase();
        match lower.as_str() {
            "cpu" | "ndarray" => Self::Cpu,
            "wgpu" | "gpu" | "best" | "auto" => Self::WgpuBest,
            _ => {
                if let Some(rest) = lower
                    .strip_prefix("wgpu:")
                    .or_else(|| lower.strip_prefix("discrete:"))
                {
                    if let Ok(i) = rest.trim().parse::<usize>() {
                        return Self::WgpuDiscrete(i);
                    }
                }
                if let Some(rest) = lower.strip_prefix("integrated:") {
                    if let Ok(i) = rest.trim().parse::<usize>() {
                        return Self::WgpuIntegrated(i);
                    }
                }
                Self::Cpu
            }
        }
    }

    pub fn wants_wgpu(&self) -> bool {
        !matches!(self, Self::Cpu)
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Cpu => "cpu".into(),
            Self::WgpuBest => "wgpu(best)".into(),
            Self::WgpuDiscrete(i) => format!("wgpu(discrete:{i})"),
            Self::WgpuIntegrated(i) => format!("wgpu(integrated:{i})"),
        }
    }
}

/// True if env hints at GPU while this build may only support CPU.
pub fn gpu_requested_in_env() -> bool {
    DeviceRequest::from_env().wants_wgpu()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_cpu_is_backend_neutral() {
        assert_eq!(DeviceRequest::Cpu.describe(), "cpu");
    }

    #[test]
    fn parse_wgpu_variants() {
        assert_eq!(DeviceRequest::parse("cpu"), DeviceRequest::Cpu);
        assert_eq!(DeviceRequest::parse("WGPU"), DeviceRequest::WgpuBest);
        assert_eq!(
            DeviceRequest::parse("wgpu:2"),
            DeviceRequest::WgpuDiscrete(2)
        );
        assert_eq!(
            DeviceRequest::parse("discrete:1"),
            DeviceRequest::WgpuDiscrete(1)
        );
        assert_eq!(
            DeviceRequest::parse("integrated:0"),
            DeviceRequest::WgpuIntegrated(0)
        );
    }
}
