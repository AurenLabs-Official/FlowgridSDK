//! Burn inference backend for `flowgrid-serve` (`NdArray` default; optional **`gpu-wgpu`**).

use burn::tensor::backend::Backend;
use flowgrid_device::DeviceRequest;

#[cfg(not(feature = "gpu-wgpu"))]
use burn::backend::NdArray;

#[cfg(not(feature = "gpu-wgpu"))]
pub type InferB = NdArray<f32>;

#[cfg(feature = "gpu-wgpu")]
use burn::backend::Wgpu;

#[cfg(feature = "gpu-wgpu")]
pub type InferB = Wgpu;

pub type InferDevice = <InferB as Backend>::Device;

pub fn infer_device() -> InferDevice {
    let req = DeviceRequest::from_env();
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        if req.wants_wgpu() {
            tracing::warn!(
                requested = %req.describe(),
                "FLOWGRID_DEVICE requests GPU; rebuild with `cargo build -p flowgrid-serve --features gpu-wgpu` or use CPU"
            );
        }
        burn_ndarray::NdArrayDevice::Cpu
    }
    #[cfg(feature = "gpu-wgpu")]
    {
        use burn_wgpu::WgpuDevice;
        let dev: WgpuDevice = match req {
            DeviceRequest::Cpu => WgpuDevice::Cpu,
            DeviceRequest::WgpuBest => WgpuDevice::BestAvailable,
            DeviceRequest::WgpuDiscrete(i) => WgpuDevice::DiscreteGpu(i),
            DeviceRequest::WgpuIntegrated(i) => WgpuDevice::IntegratedGpu(i),
        };
        tracing::info!(device = %req.describe(), "flowgrid-serve inference on Burn Wgpu");
        dev
    }
}
