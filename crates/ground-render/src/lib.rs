//! Native `wgpu` + `winit` 3D voxel renderer for Solara-Ground (Windows & Linux).

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use ground_world::mesh::{PackedVoxelVertex, SectionMesh};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

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
            position: Vec3::new(8.0, -56.0, 8.0),
            yaw_rad: 0.0,
            pitch_rad: -0.25,
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

    pub fn right_xz(&self) -> Vec3 {
        let (sy, cy) = self.yaw_rad.sin_cos();
        Vec3::new(-sy, 0.0, cy).normalize()
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

        let raw = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2];
        let mut planes = [Vec4::ZERO; 6];
        for (i, p) in raw.into_iter().enumerate() {
            let inv_len = p.truncate().length().recip();
            planes[i] = p * inv_len;
        }
        Self { planes }
    }

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

pub fn preferred_backends() -> wgpu::Backends {
    wgpu::Backends::VULKAN | wgpu::Backends::DX12
}

/// Positioned CPU chunk-section mesh ready to be uploaded to GPU buffers.
#[derive(Debug, Clone)]
pub struct WorldSectionMesh {
    pub origin: Vec3,
    pub mesh: SectionMesh,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniformRaw {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SectionUniformRaw {
    origin: [f32; 4],
}

struct GpuSectionDraw {
    origin: Vec3,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
    bind_group: wgpu::BindGroup,
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("solara-ground-depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Complete `wgpu` voxel rendering pipeline with depth testing and frustum-culled section draws.
pub struct VoxelRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    camera_buf: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    section_bgl: wgpu::BindGroupLayout,
    sections: Vec<GpuSectionDraw>,
}

impl VoxelRenderer {
    pub async fn new(window: Arc<Window>, initial_sections: &[WorldSectionMesh]) -> Result<Self> {
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
        let depth_view = create_depth_view(&device, config.width, config.height);

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let section_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("section-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-uniform"),
            contents: bytemuck::bytes_of(&CameraUniformRaw {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("voxel.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-pipeline-layout"),
            bind_group_layouts: &[&camera_bgl, &section_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("voxel-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[packed_voxel_vertex_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let mut renderer = Self {
            surface,
            device,
            queue,
            config,
            depth_view,
            pipeline,
            camera_buf,
            camera_bg,
            section_bgl,
            sections: Vec::new(),
        };
        renderer.upload_sections(initial_sections);
        Ok(renderer)
    }

    pub fn upload_sections(&mut self, world_sections: &[WorldSectionMesh]) {
        self.sections.clear();
        for ws in world_sections {
            if ws.mesh.is_empty() {
                continue;
            }
            let vertex_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("section-vbo"),
                    contents: bytemuck::cast_slice(&ws.mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let index_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("section-ibo"),
                    contents: bytemuck::cast_slice(&ws.mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            let origin_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("section-origin-ubo"),
                    contents: bytemuck::bytes_of(&SectionUniformRaw {
                        origin: [ws.origin.x, ws.origin.y, ws.origin.z, 0.0],
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("section-bg"),
                layout: &self.section_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: origin_buf.as_entire_binding(),
                }],
            });
            self.sections.push(GpuSectionDraw {
                origin: ws.origin,
                vertex_buf,
                index_buf,
                index_count: ws.mesh.indices.len() as u32,
                bind_group,
            });
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_view =
                create_depth_view(&self.device, self.config.width, self.config.height);
        }
    }

    pub fn render(&mut self, camera: &VoxelCamera) -> std::result::Result<(), wgpu::SurfaceError> {
        let vp = camera.view_proj_matrix();
        let frustum = Frustum::from_view_proj(vp);
        self.queue.write_buffer(
            &self.camera_buf,
            0,
            bytemuck::bytes_of(&CameraUniformRaw {
                view_proj: vp.to_cols_array_2d(),
            }),
        );

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel-frame-encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("voxel-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Overworld sky blue (#78A7FF)
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.20,
                            g: 0.42,
                            b: 0.95,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.camera_bg, &[]);

            for sec in &self.sections {
                let min = sec.origin;
                let max = sec.origin + Vec3::splat(16.0);
                if !frustum.intersects_aabb(min, max) {
                    continue;
                }
                rpass.set_bind_group(1, &sec.bind_group, &[]);
                rpass.set_vertex_buffer(0, sec.vertex_buf.slice(..));
                rpass.set_index_buffer(sec.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..sec.index_count, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

struct VoxelApp {
    window_title: String,
    initial_sections: Vec<WorldSectionMesh>,
    camera: VoxelCamera,
    window: Option<Arc<Window>>,
    renderer: Option<VoxelRenderer>,
    pressed_keys: HashSet<KeyCode>,
    last_frame: Instant,
}

impl ApplicationHandler for VoxelApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(self.window_title.clone())
            .with_inner_size(PhysicalSize::new(1280, 720));
        match event_loop.create_window(attrs) {
            Ok(win) => {
                let win = Arc::new(win);
                let size = win.inner_size();
                self.camera.aspect = size.width.max(1) as f32 / size.height.max(1) as f32;
                match pollster::block_on(VoxelRenderer::new(
                    Arc::clone(&win),
                    &self.initial_sections,
                )) {
                    Ok(renderer) => {
                        self.window = Some(win);
                        self.renderer = Some(renderer);
                        self.last_frame = Instant::now();
                    }
                    Err(err) => {
                        tracing::error!(error = %err, "failed to initialize wgpu renderer");
                        event_loop.exit();
                    }
                }
            }
            Err(err) => {
                tracing::error!(error = %err, "failed to create window");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.camera.aspect = size.width as f32 / size.height as f32;
                }
                if let Some(r) = &mut self.renderer {
                    r.resize(size);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                if code == KeyCode::Escape && state == ElementState::Pressed {
                    event_loop.exit();
                    return;
                }
                match state {
                    ElementState::Pressed => {
                        self.pressed_keys.insert(code);
                    }
                    ElementState::Released => {
                        self.pressed_keys.remove(&code);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                let speed = if self.pressed_keys.contains(&KeyCode::ControlLeft) {
                    28.0
                } else {
                    12.0
                };
                let turn_speed = 2.0;

                if self.pressed_keys.contains(&KeyCode::ArrowLeft) {
                    self.camera.yaw_rad -= turn_speed * dt;
                }
                if self.pressed_keys.contains(&KeyCode::ArrowRight) {
                    self.camera.yaw_rad += turn_speed * dt;
                }
                if self.pressed_keys.contains(&KeyCode::ArrowUp) {
                    self.camera.pitch_rad = (self.camera.pitch_rad + turn_speed * dt).min(1.54);
                }
                if self.pressed_keys.contains(&KeyCode::ArrowDown) {
                    self.camera.pitch_rad = (self.camera.pitch_rad - turn_speed * dt).max(-1.54);
                }

                let fwd = self.camera.forward();
                let right = self.camera.right_xz();
                if self.pressed_keys.contains(&KeyCode::KeyW) {
                    self.camera.position += fwd * speed * dt;
                }
                if self.pressed_keys.contains(&KeyCode::KeyS) {
                    self.camera.position -= fwd * speed * dt;
                }
                if self.pressed_keys.contains(&KeyCode::KeyD) {
                    self.camera.position += right * speed * dt;
                }
                if self.pressed_keys.contains(&KeyCode::KeyA) {
                    self.camera.position -= right * speed * dt;
                }
                if self.pressed_keys.contains(&KeyCode::Space) {
                    self.camera.position.y += speed * dt;
                }
                if self.pressed_keys.contains(&KeyCode::ShiftLeft) {
                    self.camera.position.y -= speed * dt;
                }

                if let Some(r) = &mut self.renderer {
                    match r.render(&self.camera) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            if let Some(w) = &self.window {
                                r.resize(w.inner_size());
                            }
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(err) => tracing::warn!(?err, "wgpu surface error"),
                    }
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

/// Opens the native `winit` + `wgpu` 3D voxel window and renders `sections` interactively.
pub fn run_voxel_window(
    title: impl Into<String>,
    camera: VoxelCamera,
    sections: Vec<WorldSectionMesh>,
) -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = VoxelApp {
        window_title: title.into(),
        initial_sections: sections,
        camera,
        window: None,
        renderer: None,
        pressed_keys: HashSet::new(),
        last_frame: Instant::now(),
    };
    event_loop.run_app(&mut app).context("run winit event loop")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_culls_chunks_behind_camera_and_keeps_visible_chunks() {
        let cam = VoxelCamera {
            position: Vec3::ZERO,
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            ..Default::default()
        };
        let frustum = Frustum::from_view_proj(cam.view_proj_matrix());

        assert!(frustum.intersects_aabb(Vec3::new(32.0, -8.0, -8.0), Vec3::new(48.0, 8.0, 8.0)));
        assert!(!frustum.intersects_aabb(Vec3::new(-48.0, -8.0, -8.0), Vec3::new(-32.0, 8.0, 8.0)));
    }

    #[test]
    fn vertex_layout_stride_is_eight_bytes() {
        assert_eq!(packed_voxel_vertex_layout().array_stride, 8);
    }
}
