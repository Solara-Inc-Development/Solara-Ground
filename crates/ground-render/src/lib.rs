//! Native `wgpu` + `winit` voxel rendering foundation for Solara-Ground (Windows & Linux).

use anyhow::{Context, Result};
use glam::{Mat4, Vec3, Vec4};
use ground_world::mesh::PackedVoxelVertex;
use std::sync::Arc;
use winit::window::Window;

/// Returns the `wgpu` vertex buffer layout matching `PackedVoxelVertex` (8 bytes/vertex).
pub const fn packed_voxel_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Uint32x2];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<PackedVoxelVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// First-person 3D voxel camera producing `[0, 1]` depth view-projection matrices for `wgpu`.
#[derive(Debug, Clone)]
pub struct VoxelCamera {
    pub position: Vec3,
    pub yaw_rad: f32,
    pub pitch_rad: f32,
    pub fov_y_rad: f32,
    pub aspect: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Default for VoxelCamera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 80.0, 0.0),
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            fov_y_rad: 70.0_f32.to_radians(),
            aspect: 16.0 / 9.0,
            z_near: 0.05,
            z_far: 1024.0,
        }
    }
}

impl VoxelCamera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw_rad.sin_cos();
        let (sp, cp) = self.pitch_rad.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp).normalize()
    }

    pub fn view_proj_matrix(&self) -> Mat4 {
        let dir = self.forward();
        let view = Mat4::look_to_rh(self.position, dir, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov_y_rad, self.aspect, self.z_near, self.z_far);
        proj * view
    }
}

/// Six-plane view frustum extracted via Gribb-Hartmann for fast CPU chunk AABB culling.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_view_proj(vp: Mat4) -> Self {
        let r0 = vp.row(0);
        let r1 = vp.row(1);
        let r2 = vp.row(2);
        let r3 = vp.row(3);

        let raw = [
            r3 + r0, // left
            r3 - r0, // right
            r3 + r1, // bottom
            r3 - r1, // top
            r2,      // near ([0, 1] wgpu depth convention)
            r3 - r2, // far
        ];

        let mut planes = [Vec4::ZERO; 6];
        for (i, p) in raw.into_iter().enumerate() {
            let inv_len = p.truncate().length().recip();
            planes[i] = p * inv_len;
        }
        Self { planes }
    }

    /// Returns `true` if the axis-aligned bounding box `[min, max]` is inside or intersects the frustum.
    #[inline]
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            let px = if plane.x >= 0.0 { max.x } else { min.x };
            let py = if plane.y >= 0.0 { max.y } else { min.y };
            let pz = if plane.z >= 0.0 { max.z } else { min.z };
            if plane.x * px + plane.y * py + plane.z * pz + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Target GPU backends for Solara-Ground (Vulkan on Linux/Windows + DX12 on Windows).
pub fn preferred_backends() -> wgpu::Backends {
    wgpu::Backends::VULKAN | wgpu::Backends::DX12
}

/// Active `wgpu` swapchain & device state bound to a `winit` window.
pub struct GpuContext {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: preferred_backends(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window)
            .context("create wgpu window surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("no compatible Vulkan/DX12 GPU adapter found")?;

        tracing::info!(adapter = ?adapter.get_info(), "initialized Solara-Ground GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("solara-ground-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("request wgpu logical device")?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Ok(Self {
            surface,
            device,
            queue,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_culls_chunks_behind_camera_and_keeps_visible_chunks() {
        let cam = VoxelCamera {
            position: Vec3::ZERO,
            yaw_rad: 0.0, // Looking along +X
            pitch_rad: 0.0,
            ..Default::default()
        };
        let frustum = Frustum::from_view_proj(cam.view_proj_matrix());

        // Chunk directly ahead at +X = [32..48]
        assert!(frustum.intersects_aabb(Vec3::new(32.0, -8.0, -8.0), Vec3::new(48.0, 8.0, 8.0)));
        // Chunk directly behind at -X = [-48..-32]
        assert!(!frustum.intersects_aabb(Vec3::new(-48.0, -8.0, -8.0), Vec3::new(-32.0, 8.0, 8.0)));
    }

    #[test]
    fn vertex_layout_stride_is_eight_bytes() {
        assert_eq!(packed_voxel_vertex_layout().array_stride, 8);
    }
}
