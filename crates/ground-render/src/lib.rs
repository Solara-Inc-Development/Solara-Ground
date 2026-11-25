//! Native `wgpu` + `winit` 3D voxel, GPU mega-buffer + `multi_draw_indexed_indirect`,
//! multiplayer player entity, and 2D HUD renderer for Solara-Ground (Windows & Linux).

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use ground_world::assets::{
    ATLAS_LAYER_COUNT, BlockTextureAtlas, TILE_SIZE, procedural_block_color,
};
use ground_world::generated_blocks;
use ground_world::mesh::{PackedVoxelVertex, SectionMesh};
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// 9-slot building block palette displayed in the client Hotbar HUD (matching Solara-Sky slots 0..=8).
pub const HUD_HOTBAR_BLOCKS: [u32; 9] = [
    generated_blocks::default_state::STONE,
    generated_blocks::default_state::COBBLESTONE,
    generated_blocks::default_state::OAK_PLANKS,
    generated_blocks::default_state::OAK_LOG,
    generated_blocks::default_state::DIRT,
    generated_blocks::default_state::GRASS_BLOCK,
    generated_blocks::default_state::SAND,
    generated_blocks::default_state::GLASS,
    generated_blocks::default_state::GLOWSTONE,
];

/// Non-blocking updates pushed from the background network & `rayon` worker to the render thread.
#[derive(Debug)]
pub enum RenderSceneUpdate {
    ReplaceSections(Vec<WorldSectionMesh>),
    RemotePlayers(Vec<RemotePlayerVisual>),
    TeleportCamera {
        position: Vec3,
        yaw_rad: f32,
        pitch_rad: f32,
    },
}

/// Non-blocking gameplay actions pushed from the `winit` window thread to the network worker.
#[derive(Debug, Clone, Copy)]
pub enum ClientWindowAction {
    Move {
        feet_pos: [f64; 3],
        yaw_deg: f32,
        pitch_deg: f32,
        on_ground: bool,
    },
    SelectHotbarSlot(u16),
    BreakLook {
        eye_pos: [f64; 3],
        look_dir: [f64; 3],
    },
    PlaceLook {
        eye_pos: [f64; 3],
        look_dir: [f64; 3],
    },
}

/// Optional live multiplayer channels wired into `run_voxel_window`.
pub struct LiveRenderBridge {
    pub scene_rx: Receiver<RenderSceneUpdate>,
    pub action_tx: Sender<ClientWindowAction>,
}

/// Returns the `wgpu` vertex buffer layout matching `PackedVoxelVertex` (8 bytes/vertex).
pub const fn packed_voxel_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Uint32x2];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<PackedVoxelVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// Exact 20-byte GPU `VkDrawIndexedIndirectCommand` / WebGPU `DrawIndexedIndirectArgs` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct DrawIndexedIndirectCmd {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

const _: () = assert!(std::mem::size_of::<DrawIndexedIndirectCmd>() == 20);

/// Sub-allocated section draw metadata inside the contiguous GPU Mega-Buffer.
#[derive(Debug, Clone, Copy)]
pub struct MegaSectionSlot {
    pub origin: Vec3,
    pub cmd: DrawIndexedIndirectCmd,
}

/// Contiguous CPU staging payload packing all world chunk sections into a single VBO, single IBO,
/// and single section-origin storage array for zero-bind-switch / `multi_draw_indexed_indirect` rendering.
#[derive(Debug, Default)]
pub struct PackedMegaMesh {
    pub vertices: Vec<PackedVoxelVertex>,
    pub indices: Vec<u32>,
    pub origins: Vec<[f32; 4]>,
    pub slots: Vec<MegaSectionSlot>,
}

pub fn pack_mega_mesh_batch(world_sections: &[WorldSectionMesh]) -> PackedMegaMesh {
    let total_verts: usize = world_sections.iter().map(|s| s.mesh.vertices.len()).sum();
    let total_indices: usize = world_sections.iter().map(|s| s.mesh.indices.len()).sum();
    let non_empty = world_sections.iter().filter(|s| !s.mesh.is_empty()).count();

    let mut vertices = Vec::with_capacity(total_verts);
    let mut indices = Vec::with_capacity(total_indices);
    let mut origins = Vec::with_capacity(non_empty.max(1));
    let mut slots = Vec::with_capacity(non_empty);

    for ws in world_sections {
        if ws.mesh.is_empty() {
            continue;
        }
        let slot_idx = origins.len() as u32;
        let base_vertex = vertices.len() as i32;
        let first_index = indices.len() as u32;
        let index_count = ws.mesh.indices.len() as u32;

        vertices.extend_from_slice(&ws.mesh.vertices);
        indices.extend_from_slice(&ws.mesh.indices);
        origins.push([ws.origin.x, ws.origin.y, ws.origin.z, 0.0]);
        slots.push(MegaSectionSlot {
            origin: ws.origin,
            cmd: DrawIndexedIndirectCmd {
                index_count,
                instance_count: 1,
                first_index,
                base_vertex,
                first_instance: slot_idx,
            },
        });
    }

    if origins.is_empty() {
        origins.push([0.0, 0.0, 0.0, 0.0]);
    }

    PackedMegaMesh {
        vertices,
        indices,
        origins,
        slots,
    }
}

/// Filters `slots` against `frustum` into `out_cmds` without allocating.
pub fn build_frustum_culled_indirect_cmds(
    slots: &[MegaSectionSlot],
    frustum: &Frustum,
    out_cmds: &mut Vec<DrawIndexedIndirectCmd>,
) {
    out_cmds.clear();
    for s in slots {
        if frustum.intersects_aabb(s.origin, s.origin + Vec3::splat(16.0)) {
            out_cmds.push(s.cmd);
        }
    }
}

/// Compact 16-byte 3D vertex used for articulated remote multiplayer player models.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct EntityVertex {
    pub pos: [f32; 3],
    pub color: [u8; 4],
}

const fn entity_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Unorm8x4];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<EntityVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// Compact 12-byte 2D NDC vertex used for the Crosshair and 9-Slot Hotbar HUD overlay.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct HudVertex {
    pub pos_ndc: [f32; 2],
    pub color: [u8; 4],
}

const fn hud_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Unorm8x4];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<HudVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// Snapshot of a visible remote player rendered in the 3D world.
#[derive(Debug, Clone, Copy)]
pub struct RemotePlayerVisual {
    pub entity_id: i32,
    pub feet_pos: Vec3,
    pub yaw_deg: f32,
}

fn push_oriented_box(
    verts: &mut Vec<EntityVertex>,
    indices: &mut Vec<u32>,
    origin: Vec3,
    yaw_rad: f32,
    local_min: Vec3,
    local_max: Vec3,
    base_rgba: [u8; 4],
) {
    let (sy, cy) = yaw_rad.sin_cos();
    let rot = |p: Vec3| -> [f32; 3] {
        let rx = p.x * cy - p.z * sy;
        let rz = p.x * sy + p.z * cy;
        [origin.x + rx, origin.y + p.y, origin.z + rz]
    };

    let (x0, y0, z0) = (local_min.x, local_min.y, local_min.z);
    let (x1, y1, z1) = (local_max.x, local_max.y, local_max.z);

    let faces: [([Vec3; 4], f32); 6] = [
        (
            [
                Vec3::new(x0, y1, z0),
                Vec3::new(x0, y1, z1),
                Vec3::new(x1, y1, z1),
                Vec3::new(x1, y1, z0),
            ],
            1.0,
        ),
        (
            [
                Vec3::new(x0, y0, z1),
                Vec3::new(x0, y0, z0),
                Vec3::new(x1, y0, z0),
                Vec3::new(x1, y0, z1),
            ],
            0.55,
        ),
        (
            [
                Vec3::new(x0, y0, z1),
                Vec3::new(x1, y0, z1),
                Vec3::new(x1, y1, z1),
                Vec3::new(x0, y1, z1),
            ],
            0.86,
        ),
        (
            [
                Vec3::new(x1, y0, z0),
                Vec3::new(x0, y0, z0),
                Vec3::new(x0, y1, z0),
                Vec3::new(x1, y1, z0),
            ],
            0.72,
        ),
        (
            [
                Vec3::new(x1, y0, z1),
                Vec3::new(x1, y0, z0),
                Vec3::new(x1, y1, z0),
                Vec3::new(x1, y1, z1),
            ],
            0.78,
        ),
        (
            [
                Vec3::new(x0, y0, z0),
                Vec3::new(x0, y0, z1),
                Vec3::new(x0, y1, z1),
                Vec3::new(x0, y1, z0),
            ],
            0.78,
        ),
    ];

    for (quad, shade) in faces {
        let color = [
            ((f32::from(base_rgba[0]) * shade) as u8),
            ((f32::from(base_rgba[1]) * shade) as u8),
            ((f32::from(base_rgba[2]) * shade) as u8),
            base_rgba[3],
        ];
        let base = verts.len() as u32;
        for corner in quad {
            verts.push(EntityVertex {
                pos: rot(corner),
                color,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Builds a batched 3D Minecraft-proportioned player mesh (head, torso, arms, legs) for all remote players.
pub fn build_remote_players_mesh(players: &[RemotePlayerVisual]) -> (Vec<EntityVertex>, Vec<u32>) {
    let mut verts = Vec::with_capacity(players.len() * 144);
    let mut indices = Vec::with_capacity(players.len() * 216);

    for p in players {
        let yaw_rad = p.yaw_deg.to_radians();
        let hue = (p.entity_id.unsigned_abs().wrapping_mul(2654435761)) & 0xFF;
        let tunic = [
            48 + ((hue & 0x3F) as u8),
            156 + (((hue >> 2) & 0x3F) as u8),
            188,
            255,
        ];
        let skin = [226, 178, 146, 255];
        let trousers = [58, 66, 128, 255];

        push_oriented_box(
            &mut verts,
            &mut indices,
            p.feet_pos,
            yaw_rad,
            Vec3::new(-0.25, 0.0, -0.125),
            Vec3::new(0.0, 0.75, 0.125),
            trousers,
        );
        push_oriented_box(
            &mut verts,
            &mut indices,
            p.feet_pos,
            yaw_rad,
            Vec3::new(0.0, 0.0, -0.125),
            Vec3::new(0.25, 0.75, 0.125),
            trousers,
        );
        push_oriented_box(
            &mut verts,
            &mut indices,
            p.feet_pos,
            yaw_rad,
            Vec3::new(-0.25, 0.75, -0.125),
            Vec3::new(0.25, 1.50, 0.125),
            tunic,
        );
        push_oriented_box(
            &mut verts,
            &mut indices,
            p.feet_pos,
            yaw_rad,
            Vec3::new(-0.50, 0.75, -0.125),
            Vec3::new(-0.25, 1.50, 0.125),
            tunic,
        );
        push_oriented_box(
            &mut verts,
            &mut indices,
            p.feet_pos,
            yaw_rad,
            Vec3::new(0.25, 0.75, -0.125),
            Vec3::new(0.50, 1.50, 0.125),
            tunic,
        );
        push_oriented_box(
            &mut verts,
            &mut indices,
            p.feet_pos,
            yaw_rad,
            Vec3::new(-0.25, 1.35, -0.25),
            Vec3::new(0.25, 1.85, 0.25),
            skin,
        );
    }

    (verts, indices)
}

fn push_hud_rect_px(
    out: &mut Vec<HudVertex>,
    screen_w: f32,
    screen_h: f32,
    rect_px: [f32; 4],
    color: [u8; 4],
) {
    let [x, y, w, h] = rect_px;
    let x0 = (x / screen_w) * 2.0 - 1.0;
    let x1 = ((x + w) / screen_w) * 2.0 - 1.0;
    let y0 = 1.0 - (y / screen_h) * 2.0;
    let y1 = 1.0 - ((y + h) / screen_h) * 2.0;

    let v0 = HudVertex {
        pos_ndc: [x0, y1],
        color,
    };
    let v1 = HudVertex {
        pos_ndc: [x1, y1],
        color,
    };
    let v2 = HudVertex {
        pos_ndc: [x1, y0],
        color,
    };
    let v3 = HudVertex {
        pos_ndc: [x0, y0],
        color,
    };
    out.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
}

/// Builds pixel-crisp 2D HUD triangles for the center Crosshair and 9-Slot Hotbar.
pub fn build_hud_vertices(width: u32, height: u32, selected_slot: usize) -> Vec<HudVertex> {
    let sw = width.max(1) as f32;
    let sh = height.max(1) as f32;
    let mut out = Vec::with_capacity(256);

    let cx = (sw * 0.5).floor();
    let cy = (sh * 0.5).floor();
    push_hud_rect_px(
        &mut out,
        sw,
        sh,
        [cx - 9.0, cy - 2.0, 18.0, 4.0],
        [20, 20, 24, 180],
    );
    push_hud_rect_px(
        &mut out,
        sw,
        sh,
        [cx - 2.0, cy - 9.0, 4.0, 18.0],
        [20, 20, 24, 180],
    );
    push_hud_rect_px(
        &mut out,
        sw,
        sh,
        [cx - 8.0, cy - 1.0, 16.0, 2.0],
        [245, 248, 255, 235],
    );
    push_hud_rect_px(
        &mut out,
        sw,
        sh,
        [cx - 1.0, cy - 8.0, 2.0, 16.0],
        [245, 248, 255, 235],
    );

    let slot_px = 44.0f32;
    let gap_px = 4.0f32;
    let total_w = 9.0 * slot_px + 8.0 * gap_px + 12.0;
    let bar_x = ((sw - total_w) * 0.5).floor();
    let bar_y = (sh - slot_px - 22.0).max(8.0);

    push_hud_rect_px(
        &mut out,
        sw,
        sh,
        [bar_x, bar_y - 6.0, total_w, slot_px + 12.0],
        [18, 20, 26, 196],
    );

    let sel = selected_slot.min(8);
    for (i, &state_id) in HUD_HOTBAR_BLOCKS.iter().enumerate() {
        let sx = bar_x + 6.0 + (i as f32) * (slot_px + gap_px);
        let sy = bar_y;

        if i == sel {
            push_hud_rect_px(
                &mut out,
                sw,
                sh,
                [sx - 3.0, sy - 3.0, slot_px + 6.0, slot_px + 6.0],
                [248, 250, 255, 250],
            );
        }

        push_hud_rect_px(
            &mut out,
            sw,
            sh,
            [sx, sy, slot_px, slot_px],
            [36, 40, 48, 230],
        );

        let top_c = procedural_block_color(state_id, true);
        let side_c = procedural_block_color(state_id, false);
        let shaded_side = [
            ((u16::from(side_c[0]) * 4) / 5) as u8,
            ((u16::from(side_c[1]) * 4) / 5) as u8,
            ((u16::from(side_c[2]) * 4) / 5) as u8,
            255,
        ];
        push_hud_rect_px(
            &mut out,
            sw,
            sh,
            [sx + 8.0, sy + 8.0, slot_px - 16.0, 10.0],
            [top_c[0], top_c[1], top_c[2], 255],
        );
        push_hud_rect_px(
            &mut out,
            sw,
            sh,
            [sx + 8.0, sy + 18.0, slot_px - 16.0, slot_px - 26.0],
            shaded_side,
        );
    }

    out
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
            z_far: 512.0,
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

/// Positioned CPU chunk-section mesh ready to be packed into the GPU Mega-Buffer.
#[derive(Debug, Clone)]
pub struct WorldSectionMesh {
    pub origin: Vec3,
    pub mesh: SectionMesh,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniformRaw {
    view_proj: [[f32; 4]; 4],
    camera_pos_fog: [f32; 4],
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

/// Sodium-class single-VBO Mega-Buffer `wgpu` renderer with `multi_draw_indexed_indirect`
/// (zero per-section buffer/bind-group switches), 3D remote players, and 2D HUD.
pub struct VoxelRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    atlas_view: wgpu::TextureView,
    atlas_sampler: wgpu::Sampler,
    voxel_global_bgl: wgpu::BindGroupLayout,
    voxel_pipeline: wgpu::RenderPipeline,
    entity_pipeline: wgpu::RenderPipeline,
    hud_pipeline: wgpu::RenderPipeline,
    camera_buf: wgpu::Buffer,
    voxel_global_bg: wgpu::BindGroup,
    entity_camera_bg: wgpu::BindGroup,
    supports_mdi: bool,
    mega_vbo: Option<wgpu::Buffer>,
    mega_ibo: Option<wgpu::Buffer>,
    indirect_buf: wgpu::Buffer,
    indirect_capacity: usize,
    section_slots: Vec<MegaSectionSlot>,
    visible_cmds: Vec<DrawIndexedIndirectCmd>,
    last_visible_sections: usize,
    entity_vbo: Option<wgpu::Buffer>,
    entity_ibo: Option<wgpu::Buffer>,
    entity_index_count: u32,
    remote_player_count: usize,
    hud_vbo: Option<wgpu::Buffer>,
    hud_vertex_count: u32,
    selected_slot: usize,
}

impl VoxelRenderer {
    pub async fn new(
        window: Arc<Window>,
        atlas: &BlockTextureAtlas,
        initial_sections: &[WorldSectionMesh],
        initial_players: &[RemotePlayerVisual],
    ) -> Result<Self> {
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

        let desired_mdi =
            wgpu::Features::MULTI_DRAW_INDIRECT | wgpu::Features::INDIRECT_FIRST_INSTANCE;
        let supports_mdi = adapter.features().contains(desired_mdi);
        let required_features = if supports_mdi {
            desired_mdi
        } else {
            wgpu::Features::empty()
        };

        tracing::info!(
            adapter = ?adapter.get_info(),
            multi_draw_indirect = supports_mdi,
            "initialized Solara-Ground GPU adapter"
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("solara-ground-device"),
                    required_features,
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

        let atlas_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("block-atlas-2d-array"),
                size: wgpu::Extent3d {
                    width: TILE_SIZE,
                    height: TILE_SIZE,
                    depth_or_array_layers: ATLAS_LAYER_COUNT,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &atlas.pixels,
        );
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("block-atlas-view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("block-atlas-nearest-repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let voxel_global_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-global-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let entity_camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("entity-camera-bgl"),
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
                camera_pos_fog: [0.0, 0.0, 0.0, 256.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let dummy_origins_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("origins-storage-init"),
            contents: bytemuck::cast_slice(&[[0.0f32; 4]]),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let voxel_global_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel-global-bg"),
            layout: &voxel_global_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dummy_origins_buf.as_entire_binding(),
                },
            ],
        });

        let entity_camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entity-camera-bg"),
            layout: &entity_camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let indirect_capacity = 4096usize;
        let indirect_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mega-indirect-draw-buf"),
            size: (indirect_capacity * std::mem::size_of::<DrawIndexedIndirectCmd>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let voxel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("voxel.wgsl").into()),
        });
        let entity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("entity-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("entity.wgsl").into()),
        });
        let hud_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("hud.wgsl").into()),
        });

        let voxel_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-pipeline-layout"),
            bind_group_layouts: &[&voxel_global_bgl],
            push_constant_ranges: &[],
        });
        let voxel_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("voxel-pipeline"),
            layout: Some(&voxel_layout),
            vertex: wgpu::VertexState {
                module: &voxel_shader,
                entry_point: Some("vs_main"),
                buffers: &[packed_voxel_vertex_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &voxel_shader,
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

        let entity_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("entity-pipeline-layout"),
            bind_group_layouts: &[&entity_camera_bgl],
            push_constant_ranges: &[],
        });
        let entity_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("entity-pipeline"),
            layout: Some(&entity_layout),
            vertex: wgpu::VertexState {
                module: &entity_shader,
                entry_point: Some("vs_entity"),
                buffers: &[entity_vertex_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &entity_shader,
                entry_point: Some("fs_entity"),
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

        let hud_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hud-pipeline-layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let hud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud-pipeline"),
            layout: Some(&hud_layout),
            vertex: wgpu::VertexState {
                module: &hud_shader,
                entry_point: Some("vs_hud"),
                buffers: &[hud_vertex_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &hud_shader,
                entry_point: Some("fs_hud"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
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
            atlas_view,
            atlas_sampler,
            voxel_global_bgl,
            voxel_pipeline,
            entity_pipeline,
            hud_pipeline,
            camera_buf,
            voxel_global_bg,
            entity_camera_bg,
            supports_mdi,
            mega_vbo: None,
            mega_ibo: None,
            indirect_buf,
            indirect_capacity,
            section_slots: Vec::new(),
            visible_cmds: Vec::with_capacity(indirect_capacity),
            last_visible_sections: 0,
            entity_vbo: None,
            entity_ibo: None,
            entity_index_count: 0,
            remote_player_count: 0,
            hud_vbo: None,
            hud_vertex_count: 0,
            selected_slot: 0,
        };
        renderer.upload_sections(initial_sections);
        renderer.upload_remote_players(initial_players);
        renderer.rebuild_hud();
        Ok(renderer)
    }

    pub fn selected_slot(&self) -> usize {
        self.selected_slot
    }

    pub fn draw_stats(&self) -> (usize, usize, usize, bool) {
        (
            self.last_visible_sections,
            self.section_slots.len(),
            self.remote_player_count,
            self.supports_mdi,
        )
    }

    pub fn set_selected_slot(&mut self, slot: usize) {
        let clamped = slot.min(8);
        if self.selected_slot != clamped {
            self.selected_slot = clamped;
            self.rebuild_hud();
        }
    }

    fn rebuild_hud(&mut self) {
        let verts = build_hud_vertices(self.config.width, self.config.height, self.selected_slot);
        self.hud_vertex_count = verts.len() as u32;
        if verts.is_empty() {
            self.hud_vbo = None;
            return;
        }
        self.hud_vbo = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("hud-vbo"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
        );
    }

    pub fn upload_remote_players(&mut self, players: &[RemotePlayerVisual]) {
        self.remote_player_count = players.len();
        let (verts, indices) = build_remote_players_mesh(players);
        self.entity_index_count = indices.len() as u32;
        if indices.is_empty() {
            self.entity_vbo = None;
            self.entity_ibo = None;
            return;
        }
        self.entity_vbo = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("remote-players-vbo"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
        );
        self.entity_ibo = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("remote-players-ibo"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
        );
    }

    /// Packs all world sections into 1 single Mega-VBO, 1 single Mega-IBO, and 1 single
    /// storage buffer of section origins.
    pub fn upload_sections(&mut self, world_sections: &[WorldSectionMesh]) {
        let packed = pack_mega_mesh_batch(world_sections);
        self.section_slots = packed.slots;

        if packed.indices.is_empty() {
            self.mega_vbo = None;
            self.mega_ibo = None;
            return;
        }

        if self.section_slots.len() > self.indirect_capacity {
            self.indirect_capacity = self.section_slots.len().next_power_of_two();
            self.indirect_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mega-indirect-draw-buf"),
                size: (self.indirect_capacity * std::mem::size_of::<DrawIndexedIndirectCmd>())
                    as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        self.mega_vbo = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mega-voxel-vbo"),
                    contents: bytemuck::cast_slice(&packed.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
        );
        self.mega_ibo = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mega-voxel-ibo"),
                    contents: bytemuck::cast_slice(&packed.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
        );
        let origins_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mega-section-origins-ssbo"),
                contents: bytemuck::cast_slice(&packed.origins),
                usage: wgpu::BufferUsages::STORAGE,
            });

        self.voxel_global_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel-global-bg"),
            layout: &self.voxel_global_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: origins_buf.as_entire_binding(),
                },
            ],
        });
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_view =
                create_depth_view(&self.device, self.config.width, self.config.height);
            self.rebuild_hud();
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
                camera_pos_fog: [
                    camera.position.x,
                    camera.position.y,
                    camera.position.z,
                    camera.z_far * 0.65,
                ],
            }),
        );

        build_frustum_culled_indirect_cmds(&self.section_slots, &frustum, &mut self.visible_cmds);
        self.last_visible_sections = self.visible_cmds.len();

        if self.supports_mdi && !self.visible_cmds.is_empty() {
            self.queue.write_buffer(
                &self.indirect_buf,
                0,
                bytemuck::cast_slice(&self.visible_cmds),
            );
        }

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxel-frame-encoder"),
            });

        // Pass 1: Single-Bind Mega-Buffer 3D World + 3D Remote Players
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("voxel-3d-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
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

            if !self.visible_cmds.is_empty()
                && let (Some(vbo), Some(ibo)) = (&self.mega_vbo, &self.mega_ibo)
            {
                rpass.set_pipeline(&self.voxel_pipeline);
                rpass.set_bind_group(0, &self.voxel_global_bg, &[]);
                rpass.set_vertex_buffer(0, vbo.slice(..));
                rpass.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint32);

                if self.supports_mdi {
                    rpass.multi_draw_indexed_indirect(
                        &self.indirect_buf,
                        0,
                        self.visible_cmds.len() as u32,
                    );
                } else {
                    for cmd in &self.visible_cmds {
                        rpass.draw_indexed(
                            cmd.first_index..(cmd.first_index + cmd.index_count),
                            cmd.base_vertex,
                            cmd.first_instance..(cmd.first_instance + 1),
                        );
                    }
                }
            }

            if self.entity_index_count > 0
                && let (Some(vbo), Some(ibo)) = (&self.entity_vbo, &self.entity_ibo)
            {
                rpass.set_pipeline(&self.entity_pipeline);
                rpass.set_bind_group(0, &self.entity_camera_bg, &[]);
                rpass.set_vertex_buffer(0, vbo.slice(..));
                rpass.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..self.entity_index_count, 0, 0..1);
            }
        }

        // Pass 2: 2D Crosshair & 9-Slot Hotbar HUD Overlay
        if self.hud_vertex_count > 0
            && let Some(hud_vbo) = &self.hud_vbo
        {
            let mut hpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hud-2d-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            hpass.set_pipeline(&self.hud_pipeline);
            hpass.set_vertex_buffer(0, hud_vbo.slice(..));
            hpass.draw(0..self.hud_vertex_count, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

struct VoxelApp {
    base_title: String,
    atlas: BlockTextureAtlas,
    initial_sections: Vec<WorldSectionMesh>,
    initial_players: Vec<RemotePlayerVisual>,
    bridge: Option<LiveRenderBridge>,
    camera: VoxelCamera,
    window: Option<Arc<Window>>,
    renderer: Option<VoxelRenderer>,
    pressed_keys: HashSet<KeyCode>,
    last_frame: Instant,
    net_move_accum: f32,
    fps_timer: f32,
    fps_frames: u32,
}

impl ApplicationHandler for VoxelApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(self.base_title.clone())
            .with_inner_size(PhysicalSize::new(1280, 720));
        match event_loop.create_window(attrs) {
            Ok(win) => {
                let win = Arc::new(win);
                let size = win.inner_size();
                self.camera.aspect = size.width.max(1) as f32 / size.height.max(1) as f32;
                match pollster::block_on(VoxelRenderer::new(
                    Arc::clone(&win),
                    &self.atlas,
                    &self.initial_sections,
                    &self.initial_players,
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
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                if let Some(bridge) = &self.bridge {
                    let eye = [
                        f64::from(self.camera.position.x),
                        f64::from(self.camera.position.y),
                        f64::from(self.camera.position.z),
                    ];
                    let fwd = self.camera.forward();
                    let dir = [f64::from(fwd.x), f64::from(fwd.y), f64::from(fwd.z)];
                    match button {
                        MouseButton::Left => {
                            let _ = bridge.action_tx.send(ClientWindowAction::BreakLook {
                                eye_pos: eye,
                                look_dir: dir,
                            });
                        }
                        MouseButton::Right => {
                            let _ = bridge.action_tx.send(ClientWindowAction::PlaceLook {
                                eye_pos: eye,
                                look_dir: dir,
                            });
                        }
                        _ => {}
                    }
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
                if state == ElementState::Pressed {
                    let hotbar_key = match code {
                        KeyCode::Digit1 => Some(0usize),
                        KeyCode::Digit2 => Some(1),
                        KeyCode::Digit3 => Some(2),
                        KeyCode::Digit4 => Some(3),
                        KeyCode::Digit5 => Some(4),
                        KeyCode::Digit6 => Some(5),
                        KeyCode::Digit7 => Some(6),
                        KeyCode::Digit8 => Some(7),
                        KeyCode::Digit9 => Some(8),
                        _ => None,
                    };
                    if let Some(slot) = hotbar_key {
                        if let Some(r) = &mut self.renderer {
                            r.set_selected_slot(slot);
                        }
                        if let Some(b) = &self.bridge {
                            let _ = b
                                .action_tx
                                .send(ClientWindowAction::SelectHotbarSlot(slot as u16));
                        }
                    }
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

                // Drain any non-blocking live server scene updates
                if let Some(bridge) = &self.bridge {
                    while let Ok(update) = bridge.scene_rx.try_recv() {
                        match update {
                            RenderSceneUpdate::ReplaceSections(secs) => {
                                if let Some(r) = &mut self.renderer {
                                    r.upload_sections(&secs);
                                }
                            }
                            RenderSceneUpdate::RemotePlayers(players) => {
                                if let Some(r) = &mut self.renderer {
                                    r.upload_remote_players(&players);
                                }
                            }
                            RenderSceneUpdate::TeleportCamera {
                                position,
                                yaw_rad,
                                pitch_rad,
                            } => {
                                self.camera.position = position;
                                self.camera.yaw_rad = yaw_rad;
                                self.camera.pitch_rad = pitch_rad;
                            }
                        }
                    }
                }

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

                // Stream 20 Hz movement updates to server
                self.net_move_accum += dt;
                if self.net_move_accum >= 0.05 {
                    self.net_move_accum = 0.0;
                    if let Some(bridge) = &self.bridge {
                        let _ = bridge.action_tx.send(ClientWindowAction::Move {
                            feet_pos: [
                                f64::from(self.camera.position.x),
                                f64::from(self.camera.position.y) - 1.62,
                                f64::from(self.camera.position.z),
                            ],
                            yaw_deg: self.camera.yaw_rad.to_degrees(),
                            pitch_deg: self.camera.pitch_rad.to_degrees(),
                            on_ground: false,
                        });
                    }
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

                // Live FPS & MSPT profiler HUD in window title bar
                self.fps_frames += 1;
                self.fps_timer += dt;
                if self.fps_timer >= 0.5 {
                    let fps = (self.fps_frames as f32 / self.fps_timer).round() as u32;
                    let ms = (self.fps_timer * 1000.0) / (self.fps_frames.max(1) as f32);
                    if let (Some(w), Some(r)) = (&self.window, &self.renderer) {
                        let (vis, tot, pcnt, mdi) = r.draw_stats();
                        let mode = if mdi { "MDI" } else { "MegaBatch" };
                        w.set_title(&format!(
                            "{} | {} FPS ({:.2} ms) | {}: {}/{} secs | Players: {} | Slot: {}",
                            self.base_title,
                            fps,
                            ms,
                            mode,
                            vis,
                            tot,
                            pcnt,
                            r.selected_slot() + 1
                        ));
                    }
                    self.fps_frames = 0;
                    self.fps_timer = 0.0;
                }

                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

/// Opens the native `winit` + `wgpu` 3D voxel window with Mega-Buffer MDI, live multiplayer bridge, and HUD.
pub fn run_voxel_window(
    title: impl Into<String>,
    atlas: BlockTextureAtlas,
    camera: VoxelCamera,
    sections: Vec<WorldSectionMesh>,
    remote_players: Vec<RemotePlayerVisual>,
    bridge: Option<LiveRenderBridge>,
) -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = VoxelApp {
        base_title: title.into(),
        atlas,
        initial_sections: sections,
        initial_players: remote_players,
        bridge,
        camera,
        window: None,
        renderer: None,
        pressed_keys: HashSet::new(),
        last_frame: Instant::now(),
        net_move_accum: 0.0,
        fps_timer: 0.0,
        fps_frames: 0,
    };
    event_loop.run_app(&mut app).context("run winit event loop")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ground_world::ChunkSection;
    use ground_world::mesh::mesh_section;

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

    #[test]
    fn mega_buffer_packs_sections_and_generates_frustum_culled_indirect_draws() {
        let mut s0 = ChunkSection::new_air();
        s0.set(1, 1, 1, generated_blocks::default_state::STONE);
        let m0 = mesh_section(&s0);

        let sections = vec![
            WorldSectionMesh {
                origin: Vec3::new(32.0, 0.0, 0.0), // in front (+X)
                mesh: m0.clone(),
            },
            WorldSectionMesh {
                origin: Vec3::new(-64.0, 0.0, 0.0), // behind (-X)
                mesh: m0,
            },
        ];
        let packed = pack_mega_mesh_batch(&sections);
        assert_eq!(packed.slots.len(), 2);
        assert_eq!(packed.slots[0].cmd.base_vertex, 0);
        assert_eq!(packed.slots[0].cmd.first_instance, 0);
        assert_eq!(packed.slots[1].cmd.base_vertex, 24);
        assert_eq!(packed.slots[1].cmd.first_instance, 1);

        let cam = VoxelCamera {
            position: Vec3::ZERO,
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            ..Default::default()
        };
        let frustum = Frustum::from_view_proj(cam.view_proj_matrix());
        let mut visible = Vec::new();
        build_frustum_culled_indirect_cmds(&packed.slots, &frustum, &mut visible);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].first_instance, 0);
    }

    #[test]
    fn remote_player_and_hud_meshes_generate_expected_primitives() {
        let (p_verts, p_idx) = build_remote_players_mesh(&[RemotePlayerVisual {
            entity_id: 7,
            feet_pos: Vec3::new(8.0, -58.0, 8.0),
            yaw_deg: 45.0,
        }]);
        assert_eq!(p_verts.len(), 144);
        assert_eq!(p_idx.len(), 216);

        let hud = build_hud_vertices(1280, 720, 3);
        assert!(hud.len() >= 180);
        assert_eq!(hud.len() % 6, 0);
    }
}
