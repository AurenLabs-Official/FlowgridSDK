//! Burn backend selection for `flowgrid-llm`.
//!
//! Default: **`NdArray`** on CPU. With **`--features gpu-wgpu`**, uses Burn **`Wgpu`**
//! when [`flowgrid_device::DeviceRequest`] is not [`Cpu`](flowgrid_device::DeviceRequest::Cpu).

use burn::tensor::backend::Backend;
use flowgrid_device::DeviceRequest;

#[cfg(not(feature = "gpu-wgpu"))]
use burn::backend::{Autodiff, NdArray};

#[cfg(not(feature = "gpu-wgpu"))]
pub type DiffBackend = Autodiff<NdArray<f32>>;
#[cfg(not(feature = "gpu-wgpu"))]
pub type InferBackend = NdArray<f32>;

#[cfg(feature = "gpu-wgpu")]
use burn::backend::{Autodiff, Wgpu};

#[cfg(feature = "gpu-wgpu")]
pub type DiffBackend = Autodiff<Wgpu>;
#[cfg(feature = "gpu-wgpu")]
pub type InferBackend = Wgpu;

/// Active compute device (matches [`DiffBackend`] / [`InferBackend`]).
pub type DiffDevice = <DiffBackend as Backend>::Device;

pub fn infer_device() -> DiffDevice {
    let req = DeviceRequest::from_env();
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        if req.wants_wgpu() {
            tracing::warn!(
                requested = %req.describe(),
                "FLOWGRID_DEVICE requests GPU but this binary was built without `gpu-wgpu`; using CPU NdArray"
            );
        }
        burn_ndarray::NdArrayDevice::Cpu
    }
    #[cfg(feature = "gpu-wgpu")]
    {
        use burn_wgpu::WgpuDevice;
        let dev = match req {
            DeviceRequest::Cpu => WgpuDevice::Cpu,
            DeviceRequest::WgpuBest => WgpuDevice::BestAvailable,
            DeviceRequest::WgpuDiscrete(i) => WgpuDevice::DiscreteGpu(i),
            DeviceRequest::WgpuIntegrated(i) => WgpuDevice::IntegratedGpu(i),
        };
        tracing::info!(device = %req.describe(), "flowgrid-llm using Burn Wgpu backend");
        dev
    }
}
