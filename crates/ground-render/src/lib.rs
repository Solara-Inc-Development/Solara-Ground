//! Native `wgpu` + `winit` 3D voxel, GPU mega-buffer + `multi_draw_indexed_indirect`,
//! 3D players & mobs, target block selection outline, FPS mouse-look, and 2D Survival UI renderer
//! for Solara-Ground (Windows & Linux).

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use ground_data::{generated_entities, generated_items};
use ground_effects::ClientEffects;
use ground_particles::{Particle, ParticleWorld};
use ground_progression::ProgressionState;
use ground_ui::{HudModel, QUICK_CRAFT_RECIPES, UiSlotDisplay, UiVertex, build_full_ui_vertices};
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
use winit::event::{
    DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

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
    ReplaceSections {
        sections: Vec<WorldSectionMesh>,
        chunk_count: usize,
    },
    Entities(Vec<RemoteEntityVisual>),
    TeleportCamera {
        position: Vec3,
        yaw_rad: f32,
        pitch_rad: f32,
    },
    Vitals {
        health: f32,
        food: i32,
    },
    DayTime(i64),
    InventorySlot {
        slot: usize,
        item_id: u32,
        count: u8,
    },
    ChatLine(String),
    TabHint(String),
    TargetBlock(Option<[i32; 3]>),
    BlockBreakParticles {
        pos: [i32; 3],
        state_id: u32,
    },
    WeatherRain(bool),
    Experience {
        progress: f32,
        level: i32,
        total_xp: i32,
    },
    MobEffectUpdate {
        effect_id: u32,
        amplifier: u8,
        duration_ticks: i32,
    },
    MobEffectRemove(u32),
}

/// Non-blocking gameplay actions pushed from the `winit` window thread to the network worker.
#[derive(Debug, Clone)]
pub enum ClientWindowAction {
    TickCamera {
        eye_pos: [f64; 3],
        look_dir: [f64; 3],
        wish_xz: [f64; 2],
        jump: bool,
        survival_physics: bool,
        yaw_deg: f32,
        pitch_deg: f32,
        dt: f64,
    },
    SelectHotbarSlot(u16),
    PrimaryAttackOrBreak {
        eye_pos: [f64; 3],
        look_dir: [f64; 3],
    },
    PlaceLook {
        eye_pos: [f64; 3],
        look_dir: [f64; 3],
        yaw_deg: f32,
        pitch_deg: f32,
    },
    SendChatOrCommand(String),
    RequestTabComplete(String),
    Respawn,
}

/// Live multiplayer channels wired into `run_voxel_window`.
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

#[derive(Debug, Clone, Copy)]
pub struct MegaSectionSlot {
    pub origin: Vec3,
    pub cmd: DrawIndexedIndirectCmd,
}

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

const fn ui_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Unorm8x4];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<UiVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// Snapshot of a visible remote player or Overworld mob rendered in the 3D world.
#[derive(Debug, Clone, Copy)]
pub struct RemoteEntityVisual {
    pub entity_id: i32,
    pub entity_type: u32,
    pub feet_pos: Vec3,
    pub yaw_deg: f32,
}

pub type RemotePlayerVisual = RemoteEntityVisual;

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

/// Builds a batched 3D mesh for all visible remote players, optional `F5` local player,
/// 8 Overworld mob species, 3D particles & rain streaks, and target-block selection wireframe!
pub fn build_scene_entities_particles_and_outline_mesh(
    entities: &[RemoteEntityVisual],
    local_player: Option<RemoteEntityVisual>,
    particles: &[Particle],
    target_block: Option<[i32; 3]>,
) -> (Vec<EntityVertex>, Vec<u32>) {
    let mut verts = Vec::with_capacity(entities.len() * 144 + particles.len() * 24 + 432);
    let mut indices = Vec::with_capacity(entities.len() * 216 + particles.len() * 36 + 648);

    for e in entities.iter().chain(local_player.as_ref()) {
        let yaw_rad = e.yaw_deg.to_radians();
        match e.entity_type {
            generated_entities::id::ARROW => {
                // 3D wooden shaft + steel arrowhead + white feather fletching!
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.025, -0.025, -0.34),
                    Vec3::new(0.025, 0.025, 0.24),
                    [168, 126, 72, 255],
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.045, -0.045, 0.24),
                    Vec3::new(0.045, 0.045, 0.36),
                    [210, 218, 228, 255],
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.06, -0.06, -0.40),
                    Vec3::new(0.06, 0.06, -0.30),
                    [246, 248, 252, 255],
                );
            }
            generated_entities::id::ITEM => {
                let spin = (e.entity_id as f32) * 0.75 + e.yaw_deg.to_radians();
                let loot_pos = e.feet_pos + Vec3::new(0.0, 0.16, 0.0);
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    loot_pos,
                    spin,
                    Vec3::new(-0.14, 0.0, -0.14),
                    Vec3::new(0.14, 0.28, 0.14),
                    [246, 214, 92, 255],
                );
            }
            generated_entities::id::CHICKEN => {
                let feathers = [248, 248, 250, 255];
                let beak = [232, 168, 42, 255];
                let wattle = [214, 48, 48, 255];
                for &lx in &[-0.09, 0.05] {
                    push_oriented_box(
                        &mut verts,
                        &mut indices,
                        e.feet_pos,
                        yaw_rad,
                        Vec3::new(lx, 0.0, -0.04),
                        Vec3::new(lx + 0.04, 0.24, 0.04),
                        beak,
                    );
                }
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.18, 0.24, -0.22),
                    Vec3::new(0.18, 0.52, 0.22),
                    feathers,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.12, 0.44, 0.14),
                    Vec3::new(0.12, 0.70, 0.34),
                    feathers,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.08, 0.52, 0.34),
                    Vec3::new(0.08, 0.60, 0.44),
                    beak,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.05, 0.44, 0.34),
                    Vec3::new(0.05, 0.52, 0.40),
                    wattle,
                );
            }
            generated_entities::id::PIG
            | generated_entities::id::COW
            | generated_entities::id::SHEEP => {
                let (coat, head_c, leg_h) = match e.entity_type {
                    generated_entities::id::PIG => {
                        ([242, 168, 178, 255], [246, 178, 186, 255], 0.35)
                    }
                    generated_entities::id::COW => ([92, 62, 42, 255], [120, 88, 66, 255], 0.50),
                    _ => ([242, 242, 246, 255], [214, 192, 168, 255], 0.45),
                };
                // 4 legs
                for &(lx, lz) in &[(-0.22, -0.30), (0.10, -0.30), (-0.22, 0.22), (0.10, 0.22)] {
                    push_oriented_box(
                        &mut verts,
                        &mut indices,
                        e.feet_pos,
                        yaw_rad,
                        Vec3::new(lx, 0.0, lz),
                        Vec3::new(lx + 0.14, leg_h, lz + 0.14),
                        head_c,
                    );
                }
                // Body
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.32, leg_h, -0.42),
                    Vec3::new(0.32, leg_h + 0.48, 0.42),
                    coat,
                );
                // Head
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.22, leg_h + 0.22, 0.36),
                    Vec3::new(0.22, leg_h + 0.62, 0.72),
                    head_c,
                );
            }
            generated_entities::id::CREEPER => {
                let creeper_green = [74, 168, 66, 255];
                let creeper_dark = [52, 128, 46, 255];
                for &(lx, lz) in &[(-0.24, -0.24), (0.04, -0.24), (-0.24, 0.06), (0.04, 0.06)] {
                    push_oriented_box(
                        &mut verts,
                        &mut indices,
                        e.feet_pos,
                        yaw_rad,
                        Vec3::new(lx, 0.0, lz),
                        Vec3::new(lx + 0.20, 0.38, lz + 0.18),
                        creeper_dark,
                    );
                }
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.22, 0.38, -0.14),
                    Vec3::new(0.22, 1.22, 0.14),
                    creeper_green,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.25, 1.22, -0.25),
                    Vec3::new(0.25, 1.72, 0.25),
                    creeper_green,
                );
            }
            generated_entities::id::SKELETON => {
                let bone = [226, 224, 214, 255];
                let rib = [194, 192, 182, 255];
                for &lx in &[-0.16, 0.06] {
                    push_oriented_box(
                        &mut verts,
                        &mut indices,
                        e.feet_pos,
                        yaw_rad,
                        Vec3::new(lx, 0.0, -0.06),
                        Vec3::new(lx + 0.10, 0.76, 0.06),
                        bone,
                    );
                }
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.22, 0.76, -0.10),
                    Vec3::new(0.22, 1.48, 0.10),
                    rib,
                );
                for &ax in &[-0.34, 0.22] {
                    push_oriented_box(
                        &mut verts,
                        &mut indices,
                        e.feet_pos,
                        yaw_rad,
                        Vec3::new(ax, 1.24, 0.0),
                        Vec3::new(ax + 0.12, 1.46, 0.62),
                        bone,
                    );
                }
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.24, 1.48, -0.24),
                    Vec3::new(0.24, 1.96, 0.24),
                    bone,
                );
            }
            generated_entities::id::SPIDER => {
                let chitin = [48, 42, 46, 255];
                let eyes = [224, 36, 36, 255];
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.32, 0.22, -0.52),
                    Vec3::new(0.32, 0.66, 0.08),
                    chitin,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.24, 0.24, 0.08),
                    Vec3::new(0.24, 0.62, 0.48),
                    chitin,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.18, 0.40, 0.48),
                    Vec3::new(0.18, 0.52, 0.53),
                    eyes,
                );
                for &lz in &[-0.36, -0.14, 0.08, 0.28] {
                    push_oriented_box(
                        &mut verts,
                        &mut indices,
                        e.feet_pos,
                        yaw_rad,
                        Vec3::new(-0.68, 0.04, lz),
                        Vec3::new(0.68, 0.16, lz + 0.09),
                        chitin,
                    );
                }
            }
            generated_entities::id::ZOMBIE => {
                let green_skin = [92, 152, 82, 255];
                let shirt = [56, 146, 162, 255];
                let pants = [62, 58, 132, 255];
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.25, 0.0, -0.125),
                    Vec3::new(0.25, 0.75, 0.125),
                    pants,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.25, 0.75, -0.125),
                    Vec3::new(0.25, 1.50, 0.125),
                    shirt,
                );
                // Zombie outstretched arms (+Z)
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.50, 1.25, 0.0),
                    Vec3::new(-0.25, 1.50, 0.68),
                    green_skin,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(0.25, 1.25, 0.0),
                    Vec3::new(0.50, 1.50, 0.68),
                    green_skin,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.25, 1.45, -0.25),
                    Vec3::new(0.25, 1.95, 0.25),
                    green_skin,
                );
            }
            _ => {
                let hue = (e.entity_id.unsigned_abs().wrapping_mul(2654435761)) & 0xFF;
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
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.25, 0.0, -0.125),
                    Vec3::new(0.0, 0.75, 0.125),
                    trousers,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(0.0, 0.0, -0.125),
                    Vec3::new(0.25, 0.75, 0.125),
                    trousers,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.25, 0.75, -0.125),
                    Vec3::new(0.25, 1.50, 0.125),
                    tunic,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.50, 0.75, -0.125),
                    Vec3::new(-0.25, 1.50, 0.125),
                    tunic,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(0.25, 0.75, -0.125),
                    Vec3::new(0.50, 1.50, 0.125),
                    tunic,
                );
                push_oriented_box(
                    &mut verts,
                    &mut indices,
                    e.feet_pos,
                    yaw_rad,
                    Vec3::new(-0.25, 1.35, -0.25),
                    Vec3::new(0.25, 1.85, 0.25),
                    skin,
                );
            }
        }
    }

    // 12-edge 3D target block wireframe box
    if let Some([bx, by, bz]) = target_block {
        let base = Vec3::new(bx as f32 - 0.004, by as f32 - 0.004, bz as f32 - 0.004);
        let s = 1.008f32;
        let t = 0.016f32;
        let wire = [16, 18, 22, 235];

        // 4 X-edges
        for &(ey, ez) in &[(0.0, 0.0), (s - t, 0.0), (0.0, s - t), (s - t, s - t)] {
            push_oriented_box(
                &mut verts,
                &mut indices,
                base,
                0.0,
                Vec3::new(0.0, ey, ez),
                Vec3::new(s, ey + t, ez + t),
                wire,
            );
        }
        // 4 Y-edges
        for &(ex, ez) in &[(0.0, 0.0), (s - t, 0.0), (0.0, s - t), (s - t, s - t)] {
            push_oriented_box(
                &mut verts,
                &mut indices,
                base,
                0.0,
                Vec3::new(ex, 0.0, ez),
                Vec3::new(ex + t, s, ez + t),
                wire,
            );
        }
        // 4 Z-edges
        for &(ex, ey) in &[(0.0, 0.0), (s - t, 0.0), (0.0, s - t), (s - t, s - t)] {
            push_oriented_box(
                &mut verts,
                &mut indices,
                base,
                0.0,
                Vec3::new(ex, ey, 0.0),
                Vec3::new(ex + t, ey + t, s),
                wire,
            );
        }
    }

    // 3D voxel block-break shards, melee hit sparks & falling Overworld rain streaks
    for p in particles {
        push_oriented_box(
            &mut verts,
            &mut indices,
            p.pos,
            0.0,
            -p.half_extents,
            p.half_extents,
            p.color,
        );
    }

    (verts, indices)
}

/// Builds a batched 3D mesh for entities and optional target-block wireframe.
pub fn build_entities_and_outline_mesh(
    entities: &[RemoteEntityVisual],
    target_block: Option<[i32; 3]>,
) -> (Vec<EntityVertex>, Vec<u32>) {
    build_scene_entities_particles_and_outline_mesh(entities, None, &[], target_block)
}

pub fn build_remote_players_mesh(players: &[RemoteEntityVisual]) -> (Vec<EntityVertex>, Vec<u32>) {
    build_entities_and_outline_mesh(players, None)
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

    pub fn forward_xz(&self) -> Vec3 {
        let (sy, cy) = self.yaw_rad.sin_cos();
        Vec3::new(cy, 0.0, sy).normalize()
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
    sky_fog_sun: [f32; 4],
}

/// Computes dynamic Overworld `(sky_rgb, sunlight_multiplier)` from MC `26.3` `day_time` (`0..24000`).
pub fn compute_sky_fog_and_sun(day_time_ticks: i64) -> ([f32; 3], f32) {
    let t = (day_time_ticks.rem_euclid(24_000) as f32) / 24_000.0;
    // t=0.0 (06:00 dawn), t=0.25 (12:00 noon), t=0.52 (sunset), t=0.75 (midnight)
    if t < 0.45 {
        ([0.20, 0.44, 0.96], 1.0)
    } else if t < 0.56 {
        let k = (t - 0.45) / 0.11;
        (
            [0.20 + 0.62 * k, 0.44 - 0.10 * k, 0.96 - 0.58 * k],
            1.0 - 0.45 * k,
        )
    } else if t < 0.64 {
        let k = (t - 0.56) / 0.08;
        (
            [0.82 - 0.78 * k, 0.34 - 0.29 * k, 0.38 - 0.24 * k],
            0.55 - 0.33 * k,
        )
    } else if t < 0.92 {
        ([0.04, 0.05, 0.14], 0.22)
    } else {
        let k = (t - 0.92) / 0.08;
        (
            [0.04 + 0.16 * k, 0.05 + 0.39 * k, 0.14 + 0.82 * k],
            0.22 + 0.78 * k,
        )
    }
}

/// Resolves zero-allocation `UiSlotDisplay` metadata & clean-room colors for any MC `26.3` item ID!
pub fn ui_slot_for_item(item_id: u32, count: u8) -> UiSlotDisplay {
    if item_id == 0 || count == 0 {
        return UiSlotDisplay::default();
    }
    let Some(def) = generated_items::by_id(item_id) else {
        return UiSlotDisplay {
            count,
            label: "item",
            top_color: [180, 180, 190, 255],
            side_color: [140, 140, 150, 255],
        };
    };
    let short = def.key.strip_prefix("minecraft:").unwrap_or(def.key);
    if let Some(st) = generated_blocks::BLOCK_RANGES
        .iter()
        .find(|r| r.name == def.key)
        .map(|r| r.default_state_id)
    {
        return UiSlotDisplay {
            count,
            label: short,
            top_color: procedural_block_color(st, true),
            side_color: procedural_block_color(st, false),
        };
    }
    let (top_color, side_color) = if short.contains("diamond") {
        ([110, 246, 238, 255], [72, 198, 192, 255])
    } else if short.contains("iron") {
        ([228, 232, 238, 255], [178, 184, 192, 255])
    } else if short.contains("gold") {
        ([250, 216, 86, 255], [212, 168, 48, 255])
    } else if short.contains("stone_") {
        ([156, 158, 164, 255], [116, 118, 124, 255])
    } else if short.contains("wooden_") || short == "stick" {
        ([182, 142, 86, 255], [138, 102, 58, 255])
    } else if short == "coal" {
        ([54, 54, 60, 255], [36, 36, 42, 255])
    } else if short.contains("bread") || short.contains("cooked") {
        ([218, 152, 74, 255], [172, 108, 46, 255])
    } else {
        ([226, 116, 106, 255], [182, 82, 74, 255])
    };
    UiSlotDisplay {
        count,
        label: short,
        top_color,
        side_color,
    }
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

fn hotbar_icon_colors() -> ([[u8; 4]; 9], [[u8; 4]; 9]) {
    let mut tops = [[0u8; 4]; 9];
    let mut sides = [[0u8; 4]; 9];
    for (i, &st) in HUD_HOTBAR_BLOCKS.iter().enumerate() {
        tops[i] = procedural_block_color(st, true);
        sides[i] = procedural_block_color(st, false);
    }
    (tops, sides)
}

/// Sodium-class single-VBO Mega-Buffer `wgpu` renderer with `multi_draw_indexed_indirect`,
/// 3D players & mobs, target block selection outline, and full 2D `ground-ui` Survival HUD.
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
    entities_cache: Vec<RemoteEntityVisual>,
    local_player_model: Option<RemoteEntityVisual>,
    particle_world: ParticleWorld,
    progression: ProgressionState,
    client_effects: ClientEffects,
    target_block: Option<[i32; 3]>,
    entity_vbo: Option<wgpu::Buffer>,
    entity_ibo: Option<wgpu::Buffer>,
    entity_index_count: u32,
    hud_vbo: Option<wgpu::Buffer>,
    hud_vertex_count: u32,
    hud_model: HudModel,
}

impl VoxelRenderer {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        window: Arc<Window>,
        atlas: &BlockTextureAtlas,
        initial_sections: &[WorldSectionMesh],
        initial_entities: &[RemoteEntityVisual],
        initial_health: (f32, i32),
        initial_day_time: i64,
        initial_inventory: &[(u32, u8); 40],
        initial_xp: (f32, i32, i32),
        initial_survival: bool,
        initial_chunk_count: usize,
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                sky_fog_sun: [0.20, 0.44, 0.96, 1.0],
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
                buffers: &[ui_vertex_layout()],
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

        let (tops, sides) = hotbar_icon_colors();
        let mut inv_slots = [UiSlotDisplay::default(); 36];
        for (idx, &(iid, cnt)) in initial_inventory.iter().take(36).enumerate() {
            inv_slots[idx] = ui_slot_for_item(iid, cnt);
        }
        let mut progression = ProgressionState {
            armor_items: [0; 4],
            xp_progress: initial_xp.0,
            xp_level: initial_xp.1,
            total_xp: initial_xp.2,
        };
        for (a_idx, &(iid, cnt)) in initial_inventory[36..40].iter().enumerate() {
            if cnt > 0 {
                progression.set_armor_slot(a_idx, iid);
            }
        }
        let hud_model = HudModel {
            width: config.width,
            height: config.height,
            selected_slot: 0,
            hotbar_top_colors: tops,
            hotbar_side_colors: sides,
            inventory_slots: inv_slots,
            health: initial_health.0,
            food: initial_health.1,
            armor_points: progression.total_armor_points(),
            xp_level: progression.xp_level,
            xp_progress: progression.xp_progress,
            day_time_ticks: initial_day_time,
            hand_swing: 0.0,
            survival_physics: initial_survival,
            paused: false,
            inventory_open: false,
            chat_open: false,
            chat_input: String::new(),
            tab_hint: String::new(),
            fps: 0,
            frame_ms: 0.0,
            camera_pos: [8.0, -56.0, 8.0],
            chunk_count: initial_chunk_count,
            entity_count: initial_entities.len(),
            effect_badges: Vec::new(),
            chat_lines: Vec::new(),
        };

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
            entities_cache: initial_entities.to_vec(),
            local_player_model: None,
            particle_world: ParticleWorld::new(),
            progression,
            client_effects: ClientEffects::new(),
            target_block: None,
            entity_vbo: None,
            entity_ibo: None,
            entity_index_count: 0,
            hud_vbo: None,
            hud_vertex_count: 0,
            hud_model,
        };
        renderer.upload_sections(initial_sections);
        renderer.rebuild_entity_and_outline_buffers();
        renderer.rebuild_hud();
        Ok(renderer)
    }

    pub fn hud_mut(&mut self) -> &mut HudModel {
        &mut self.hud_model
    }

    pub fn draw_stats(&self) -> (usize, usize, usize, bool) {
        (
            self.last_visible_sections,
            self.section_slots.len(),
            self.entities_cache.len(),
            self.supports_mdi,
        )
    }

    pub fn set_selected_slot(&mut self, slot: usize) {
        let clamped = slot.min(8);
        if self.hud_model.selected_slot != clamped {
            self.hud_model.selected_slot = clamped;
            self.rebuild_hud();
        }
    }

    pub fn rebuild_hud(&mut self) {
        self.hud_model.width = self.config.width;
        self.hud_model.height = self.config.height;
        self.hud_model.entity_count = self.entities_cache.len();
        let verts = build_full_ui_vertices(&self.hud_model);
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

    pub fn set_entities(&mut self, entities: Vec<RemoteEntityVisual>) {
        self.entities_cache = entities;
        self.rebuild_entity_and_outline_buffers();
    }

    pub fn set_target_block(&mut self, block: Option<[i32; 3]>) {
        if self.target_block != block {
            self.target_block = block;
            self.rebuild_entity_and_outline_buffers();
        }
    }

    pub fn set_local_player_model(&mut self, model: Option<RemoteEntityVisual>) {
        self.local_player_model = model;
        self.rebuild_entity_and_outline_buffers();
    }

    pub fn spawn_block_break_particles(&mut self, pos: [i32; 3], state_id: u32) {
        let rgba = procedural_block_color(state_id, true);
        self.particle_world.spawn_block_break(pos, rgba);
        self.rebuild_entity_and_outline_buffers();
    }

    pub fn set_weather_rain(&mut self, raining: bool) {
        self.particle_world.set_raining(raining);
    }

    pub fn update_armor_slot(&mut self, armor_idx: usize, item_id: u32, count: u8) {
        self.progression
            .set_armor_slot(armor_idx, if count > 0 { item_id } else { 0 });
        self.hud_model.armor_points = self.progression.total_armor_points();
    }

    pub fn update_experience(&mut self, progress: f32, level: i32, total_xp: i32) {
        self.progression.xp_progress = progress;
        self.progression.xp_level = level;
        self.progression.total_xp = total_xp;
        self.hud_model.xp_progress = progress;
        self.hud_model.xp_level = level;
    }

    pub fn apply_mob_effect(&mut self, effect_id: u32, amplifier: u8, duration_ticks: i32) {
        self.client_effects
            .apply_or_refresh(effect_id, amplifier, duration_ticks);
        self.hud_model.effect_badges = self.client_effects.hud_badges();
    }

    pub fn remove_mob_effect(&mut self, effect_id: u32) {
        self.client_effects.remove(effect_id);
        self.hud_model.effect_badges = self.client_effects.hud_badges();
    }

    pub fn step_effects(&mut self, dt: f32) -> bool {
        if self.client_effects.step(dt) {
            self.hud_model.effect_badges = self.client_effects.hud_badges();
            true
        } else {
            false
        }
    }

    pub fn speed_multiplier(&self) -> f32 {
        self.client_effects.speed_multiplier()
    }

    pub fn step_particles(&mut self, dt: f32, camera_pos: Vec3) {
        let had_active =
            !self.particle_world.particles().is_empty() || self.particle_world.is_raining();
        if had_active {
            self.particle_world.step(dt, camera_pos);
            self.rebuild_entity_and_outline_buffers();
        }
    }

    fn rebuild_entity_and_outline_buffers(&mut self) {
        let (verts, indices) = build_scene_entities_particles_and_outline_mesh(
            &self.entities_cache,
            self.local_player_model,
            self.particle_world.particles(),
            self.target_block,
        );
        self.entity_index_count = indices.len() as u32;
        if indices.is_empty() {
            self.entity_vbo = None;
            self.entity_ibo = None;
            return;
        }
        self.entity_vbo = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("entities-outline-vbo"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
        );
        self.entity_ibo = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("entities-outline-ibo"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
        );
    }

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
        let (mut sky_rgb, mut sun_mult) = compute_sky_fog_and_sun(self.hud_model.day_time_ticks);
        if self.particle_world.is_raining() {
            sky_rgb = [sky_rgb[0] * 0.62, sky_rgb[1] * 0.66, sky_rgb[2] * 0.72];
            sun_mult = (sun_mult * 0.72).max(0.20);
        }
        if self.client_effects.has_night_vision() {
            sky_rgb = [
                sky_rgb[0].max(0.16),
                sky_rgb[1].max(0.22),
                sky_rgb[2].max(0.38),
            ];
            sun_mult = sun_mult.max(0.88);
        }

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
                sky_fog_sun: [sky_rgb[0], sky_rgb[1], sky_rgb[2], sun_mult],
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

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("voxel-3d-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(sky_rgb[0]),
                            g: f64::from(sky_rgb[1]),
                            b: f64::from(sky_rgb[2]),
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

pub struct WindowLaunchConfig {
    pub title: String,
    pub atlas: BlockTextureAtlas,
    pub camera: VoxelCamera,
    pub sections: Vec<WorldSectionMesh>,
    pub entities: Vec<RemoteEntityVisual>,
    pub initial_health: (f32, i32),
    pub initial_day_time: i64,
    pub initial_inventory: [(u32, u8); 40],
    pub initial_xp: (f32, i32, i32),
    pub initial_survival: bool,
    pub initial_chunks: usize,
    pub bridge: Option<LiveRenderBridge>,
}

struct VoxelApp {
    cfg: Option<WindowLaunchConfig>,
    base_title: String,
    bridge: Option<LiveRenderBridge>,
    camera: VoxelCamera,
    window: Option<Arc<Window>>,
    renderer: Option<VoxelRenderer>,
    pressed_keys: HashSet<KeyCode>,
    paused: bool,
    inventory_open: bool,
    chat_open: bool,
    cursor_locked: bool,
    survival_physics: bool,
    third_person: bool,
    swing_timer: f32,
    backpack_cycle_idx: usize,
    last_frame: Instant,
    fps_timer: f32,
    fps_frames: u32,
}

impl VoxelApp {
    fn set_cursor_locked(&mut self, lock: bool) {
        if let Some(win) = &self.window {
            if lock {
                if win.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                    let _ = win.set_cursor_grab(CursorGrabMode::Confined);
                }
                win.set_cursor_visible(false);
                self.cursor_locked = true;
            } else {
                let _ = win.set_cursor_grab(CursorGrabMode::None);
                win.set_cursor_visible(true);
                self.cursor_locked = false;
            }
        }
    }
}

impl ApplicationHandler for VoxelApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let Some(cfg) = self.cfg.take() else {
            return;
        };
        self.base_title = cfg.title;
        self.camera = cfg.camera;
        self.bridge = cfg.bridge;
        self.survival_physics = cfg.initial_survival;

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
                    &cfg.atlas,
                    &cfg.sections,
                    &cfg.entities,
                    cfg.initial_health,
                    cfg.initial_day_time,
                    &cfg.initial_inventory,
                    cfg.initial_xp,
                    cfg.initial_survival,
                    cfg.initial_chunks,
                )) {
                    Ok(renderer) => {
                        self.window = Some(win);
                        self.renderer = Some(renderer);
                        self.set_cursor_locked(true);
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

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if !self.paused
            && !self.inventory_open
            && !self.chat_open
            && self.cursor_locked
            && let DeviceEvent::MouseMotion { delta: (dx, dy) } = event
        {
            let sens = 0.0024_f32;
            self.camera.yaw_rad += (dx as f32) * sens;
            self.camera.pitch_rad = (self.camera.pitch_rad - (dy as f32) * sens).clamp(-1.54, 1.54);
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
            WindowEvent::MouseWheel { delta, .. } => {
                if self.paused || self.inventory_open || self.chat_open {
                    return;
                }
                let step: isize = match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        if y > 0.1 {
                            -1
                        } else if y < -0.1 {
                            1
                        } else {
                            0
                        }
                    }
                    MouseScrollDelta::PixelDelta(p) => {
                        if p.y > 4.0 {
                            -1
                        } else if p.y < -4.0 {
                            1
                        } else {
                            0
                        }
                    }
                };
                if step != 0
                    && let Some(r) = &mut self.renderer
                {
                    let cur = r.hud_mut().selected_slot as isize;
                    let next = (cur + step).rem_euclid(9) as usize;
                    r.set_selected_slot(next);
                    if let Some(b) = &self.bridge {
                        let _ = b
                            .action_tx
                            .send(ClientWindowAction::SelectHotbarSlot(next as u16));
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                if self.chat_open || self.inventory_open {
                    return;
                }
                if self.paused {
                    self.paused = false;
                    self.set_cursor_locked(true);
                    if let Some(r) = &mut self.renderer {
                        r.hud_mut().paused = false;
                        r.rebuild_hud();
                    }
                    return;
                }
                if !self.cursor_locked {
                    self.set_cursor_locked(true);
                }
                self.swing_timer = 1.0;
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
                            let _ =
                                bridge
                                    .action_tx
                                    .send(ClientWindowAction::PrimaryAttackOrBreak {
                                        eye_pos: eye,
                                        look_dir: dir,
                                    });
                        }
                        MouseButton::Right => {
                            let _ = bridge.action_tx.send(ClientWindowAction::PlaceLook {
                                eye_pos: eye,
                                look_dir: dir,
                                yaw_deg: self.camera.yaw_rad.to_degrees(),
                                pitch_deg: self.camera.pitch_rad.to_degrees(),
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
                        text,
                        ..
                    },
                ..
            } => {
                if self.chat_open {
                    if state == ElementState::Pressed {
                        match code {
                            KeyCode::Escape => {
                                self.chat_open = false;
                                self.set_cursor_locked(true);
                                if let Some(r) = &mut self.renderer {
                                    r.hud_mut().chat_open = false;
                                    r.hud_mut().chat_input.clear();
                                    r.hud_mut().tab_hint.clear();
                                    r.rebuild_hud();
                                }
                            }
                            KeyCode::Enter => {
                                let msg = self
                                    .renderer
                                    .as_mut()
                                    .map(|r| std::mem::take(&mut r.hud_mut().chat_input))
                                    .unwrap_or_default();
                                self.chat_open = false;
                                self.set_cursor_locked(true);
                                if let Some(r) = &mut self.renderer {
                                    r.hud_mut().chat_open = false;
                                    r.hud_mut().tab_hint.clear();
                                    r.rebuild_hud();
                                }
                                if !msg.trim().is_empty()
                                    && let Some(b) = &self.bridge
                                {
                                    let _ = b
                                        .action_tx
                                        .send(ClientWindowAction::SendChatOrCommand(msg));
                                }
                            }
                            KeyCode::Backspace => {
                                if let Some(r) = &mut self.renderer {
                                    r.hud_mut().chat_input.pop();
                                    let cur = r.hud_mut().chat_input.clone();
                                    r.rebuild_hud();
                                    if cur.starts_with('/')
                                        && let Some(b) = &self.bridge
                                    {
                                        let _ = b
                                            .action_tx
                                            .send(ClientWindowAction::RequestTabComplete(cur));
                                    }
                                }
                            }
                            KeyCode::Tab => {
                                if let Some(r) = &mut self.renderer {
                                    let hint = r.hud_mut().tab_hint.clone();
                                    if !hint.is_empty() {
                                        let input = &mut r.hud_mut().chat_input;
                                        if let Some((prefix, _)) = input.rsplit_once(' ') {
                                            *input = format!("{prefix} {hint}");
                                        } else {
                                            *input = format!("/{hint} ");
                                        }
                                        r.hud_mut().tab_hint.clear();
                                        r.rebuild_hud();
                                    } else if r.hud_mut().chat_input.starts_with('/')
                                        && let Some(b) = &self.bridge
                                    {
                                        let _ = b.action_tx.send(
                                            ClientWindowAction::RequestTabComplete(
                                                r.hud_mut().chat_input.clone(),
                                            ),
                                        );
                                    }
                                }
                            }
                            _ => {
                                if let Some(txt) = text.as_deref()
                                    && let Some(r) = &mut self.renderer
                                {
                                    let mut changed = false;
                                    for ch in txt.chars() {
                                        if ch.is_ascii() && !ch.is_ascii_control() {
                                            r.hud_mut().chat_input.push(ch);
                                            changed = true;
                                        }
                                    }
                                    if changed {
                                        let cur = r.hud_mut().chat_input.clone();
                                        r.rebuild_hud();
                                        if cur.starts_with('/')
                                            && let Some(b) = &self.bridge
                                        {
                                            let _ = b
                                                .action_tx
                                                .send(ClientWindowAction::RequestTabComplete(cur));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    return;
                }

                if state == ElementState::Pressed {
                    match code {
                        KeyCode::Escape => {
                            if self.inventory_open {
                                self.inventory_open = false;
                                self.set_cursor_locked(true);
                                if let Some(r) = &mut self.renderer {
                                    r.hud_mut().inventory_open = false;
                                    r.rebuild_hud();
                                }
                                return;
                            }
                            self.paused = !self.paused;
                            self.set_cursor_locked(!self.paused);
                            if let Some(r) = &mut self.renderer {
                                r.hud_mut().paused = self.paused;
                                r.rebuild_hud();
                            }
                            return;
                        }
                        KeyCode::KeyE => {
                            self.paused = false;
                            self.inventory_open = !self.inventory_open;
                            self.set_cursor_locked(!self.inventory_open);
                            self.pressed_keys.clear();
                            if let Some(r) = &mut self.renderer {
                                r.hud_mut().paused = false;
                                r.hud_mut().inventory_open = self.inventory_open;
                                r.rebuild_hud();
                            }
                            return;
                        }
                        KeyCode::KeyT | KeyCode::Slash if !self.paused && !self.inventory_open => {
                            self.chat_open = true;
                            self.set_cursor_locked(false);
                            self.pressed_keys.clear();
                            if let Some(r) = &mut self.renderer {
                                r.hud_mut().chat_open = true;
                                r.hud_mut().chat_input = if code == KeyCode::Slash {
                                    "/".into()
                                } else {
                                    String::new()
                                };
                                r.hud_mut().tab_hint.clear();
                                r.rebuild_hud();
                            }
                            return;
                        }
                        KeyCode::KeyX if self.inventory_open => {
                            if let Some(r) = &mut self.renderer {
                                let sel = r.hud_mut().selected_slot.min(8);
                                for off in 0..27usize {
                                    let bp_idx = 9 + ((self.backpack_cycle_idx + off) % 27);
                                    if r.hud_mut().inventory_slots[bp_idx].count > 0 {
                                        r.hud_mut().inventory_slots.swap(sel, bp_idx);
                                        self.backpack_cycle_idx = (bp_idx - 9 + 1) % 27;
                                        r.rebuild_hud();
                                        break;
                                    }
                                }
                            }
                            return;
                        }
                        KeyCode::KeyQ if self.paused => {
                            event_loop.exit();
                            return;
                        }
                        KeyCode::KeyF => {
                            self.survival_physics = !self.survival_physics;
                            if let Some(r) = &mut self.renderer {
                                r.hud_mut().survival_physics = self.survival_physics;
                                r.rebuild_hud();
                            }
                        }
                        KeyCode::F5 => {
                            self.third_person = !self.third_person;
                            if !self.third_person
                                && let Some(r) = &mut self.renderer
                            {
                                r.set_local_player_model(None);
                            }
                        }
                        KeyCode::KeyR => {
                            if let Some(b) = &self.bridge {
                                let _ = b.action_tx.send(ClientWindowAction::Respawn);
                            }
                        }
                        _ => {}
                    }

                    let digit_key = match code {
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
                    if let Some(slot) = digit_key {
                        if self.inventory_open {
                            if let Some(&(recipe_key, _)) = QUICK_CRAFT_RECIPES.get(slot)
                                && let Some(b) = &self.bridge
                            {
                                let _ = b.action_tx.send(ClientWindowAction::SendChatOrCommand(
                                    format!("/craft {recipe_key}"),
                                ));
                            }
                        } else {
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

                let mut hud_dirty = false;
                if let Some(bridge) = &self.bridge {
                    while let Ok(update) = bridge.scene_rx.try_recv() {
                        match update {
                            RenderSceneUpdate::ReplaceSections {
                                sections,
                                chunk_count,
                            } => {
                                if let Some(r) = &mut self.renderer {
                                    r.hud_mut().chunk_count = chunk_count;
                                    r.upload_sections(&sections);
                                    hud_dirty = true;
                                }
                            }
                            RenderSceneUpdate::Entities(list) => {
                                if let Some(r) = &mut self.renderer {
                                    r.set_entities(list);
                                    hud_dirty = true;
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
                            RenderSceneUpdate::Vitals { health, food } => {
                                if let Some(r) = &mut self.renderer {
                                    r.hud_mut().health = health;
                                    r.hud_mut().food = food;
                                    hud_dirty = true;
                                }
                            }
                            RenderSceneUpdate::DayTime(dt_ticks) => {
                                if let Some(r) = &mut self.renderer {
                                    r.hud_mut().day_time_ticks = dt_ticks;
                                    hud_dirty = true;
                                }
                            }
                            RenderSceneUpdate::InventorySlot {
                                slot,
                                item_id,
                                count,
                            } => {
                                if let Some(r) = &mut self.renderer {
                                    if slot < r.hud_mut().inventory_slots.len() {
                                        r.hud_mut().inventory_slots[slot] =
                                            ui_slot_for_item(item_id, count);
                                        hud_dirty = true;
                                    } else if (36..40).contains(&slot) {
                                        r.update_armor_slot(slot - 36, item_id, count);
                                        hud_dirty = true;
                                    }
                                }
                            }
                            RenderSceneUpdate::ChatLine(line) => {
                                if let Some(r) = &mut self.renderer {
                                    let lines = &mut r.hud_mut().chat_lines;
                                    lines.push(line);
                                    if lines.len() > 16 {
                                        lines.remove(0);
                                    }
                                    hud_dirty = true;
                                }
                            }
                            RenderSceneUpdate::TabHint(hint) => {
                                if let Some(r) = &mut self.renderer {
                                    r.hud_mut().tab_hint = hint;
                                    hud_dirty = true;
                                }
                            }
                            RenderSceneUpdate::TargetBlock(tb) => {
                                if let Some(r) = &mut self.renderer {
                                    r.set_target_block(tb);
                                }
                            }
                            RenderSceneUpdate::BlockBreakParticles { pos, state_id } => {
                                if let Some(r) = &mut self.renderer {
                                    r.spawn_block_break_particles(pos, state_id);
                                }
                            }
                            RenderSceneUpdate::WeatherRain(raining) => {
                                if let Some(r) = &mut self.renderer {
                                    r.set_weather_rain(raining);
                                }
                            }
                            RenderSceneUpdate::Experience {
                                progress,
                                level,
                                total_xp,
                            } => {
                                if let Some(r) = &mut self.renderer {
                                    r.update_experience(progress, level, total_xp);
                                    hud_dirty = true;
                                }
                            }
                            RenderSceneUpdate::MobEffectUpdate {
                                effect_id,
                                amplifier,
                                duration_ticks,
                            } => {
                                if let Some(r) = &mut self.renderer {
                                    r.apply_mob_effect(effect_id, amplifier, duration_ticks);
                                    hud_dirty = true;
                                }
                            }
                            RenderSceneUpdate::MobEffectRemove(effect_id) => {
                                if let Some(r) = &mut self.renderer {
                                    r.remove_mob_effect(effect_id);
                                    hud_dirty = true;
                                }
                            }
                        }
                    }
                }

                if self.swing_timer > 0.0 {
                    self.swing_timer = (self.swing_timer - dt * 4.5).max(0.0);
                    if let Some(r) = &mut self.renderer {
                        r.hud_mut().hand_swing = 1.0 - self.swing_timer;
                    }
                    hud_dirty = true;
                } else if let Some(r) = &mut self.renderer
                    && r.hud_mut().hand_swing > 0.0
                {
                    r.hud_mut().hand_swing = 0.0;
                    hud_dirty = true;
                }

                if !self.paused && !self.inventory_open && !self.chat_open {
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
                        self.camera.pitch_rad =
                            (self.camera.pitch_rad - turn_speed * dt).max(-1.54);
                    }

                    let sprint = self.pressed_keys.contains(&KeyCode::ControlLeft);
                    let fwd_xz = self.camera.forward_xz();
                    let right_xz = self.camera.right_xz();
                    let mut move_xz = Vec3::ZERO;
                    if self.pressed_keys.contains(&KeyCode::KeyW) {
                        move_xz += fwd_xz;
                    }
                    if self.pressed_keys.contains(&KeyCode::KeyS) {
                        move_xz -= fwd_xz;
                    }
                    if self.pressed_keys.contains(&KeyCode::KeyD) {
                        move_xz += right_xz;
                    }
                    if self.pressed_keys.contains(&KeyCode::KeyA) {
                        move_xz -= right_xz;
                    }
                    if move_xz.length_squared() > 0.001 {
                        move_xz = move_xz.normalize();
                    }

                    if !self.survival_physics {
                        let fly_speed = if sprint { 28.0 } else { 12.0 };
                        let fwd_3d = self.camera.forward();
                        if self.pressed_keys.contains(&KeyCode::KeyW) {
                            self.camera.position += fwd_3d * fly_speed * dt;
                        }
                        if self.pressed_keys.contains(&KeyCode::KeyS) {
                            self.camera.position -= fwd_3d * fly_speed * dt;
                        }
                        if self.pressed_keys.contains(&KeyCode::KeyD) {
                            self.camera.position += right_xz * fly_speed * dt;
                        }
                        if self.pressed_keys.contains(&KeyCode::KeyA) {
                            self.camera.position -= right_xz * fly_speed * dt;
                        }
                        if self.pressed_keys.contains(&KeyCode::Space) {
                            self.camera.position.y += fly_speed * dt;
                        }
                        if self.pressed_keys.contains(&KeyCode::ShiftLeft) {
                            self.camera.position.y -= fly_speed * dt;
                        }
                    }

                    if let Some(bridge) = &self.bridge {
                        let spd_mult = self
                            .renderer
                            .as_ref()
                            .map_or(1.0, VoxelRenderer::speed_multiplier);
                        let walk_spd = (if sprint { 5.612 } else { 4.317 }) * f64::from(spd_mult);
                        let fwd = self.camera.forward();
                        let _ = bridge.action_tx.send(ClientWindowAction::TickCamera {
                            eye_pos: [
                                f64::from(self.camera.position.x),
                                f64::from(self.camera.position.y),
                                f64::from(self.camera.position.z),
                            ],
                            look_dir: [f64::from(fwd.x), f64::from(fwd.y), f64::from(fwd.z)],
                            wish_xz: [
                                f64::from(move_xz.x) * walk_spd,
                                f64::from(move_xz.z) * walk_spd,
                            ],
                            jump: self.pressed_keys.contains(&KeyCode::Space),
                            survival_physics: self.survival_physics,
                            yaw_deg: self.camera.yaw_rad.to_degrees(),
                            pitch_deg: self.camera.pitch_rad.to_degrees(),
                            dt: f64::from(dt),
                        });
                    }
                }

                self.fps_frames += 1;
                self.fps_timer += dt;
                if self.fps_timer >= 0.4 {
                    let fps = (self.fps_frames as f32 / self.fps_timer).round() as u32;
                    let ms = (self.fps_timer * 1000.0) / (self.fps_frames.max(1) as f32);
                    if let (Some(w), Some(r)) = (&self.window, &mut self.renderer) {
                        r.hud_mut().fps = fps;
                        r.hud_mut().frame_ms = ms;
                        r.hud_mut().camera_pos = self.camera.position.to_array();
                        hud_dirty = true;
                        let (vis, tot, ecnt, mdi) = r.draw_stats();
                        let mode = if mdi { "MDI" } else { "MegaBatch" };
                        w.set_title(&format!(
                            "{} | {} FPS ({:.2} ms) | {}: {}/{} secs | Entities: {}",
                            self.base_title, fps, ms, mode, vis, tot, ecnt
                        ));
                    }
                    self.fps_frames = 0;
                    self.fps_timer = 0.0;
                }

                if let Some(r) = &mut self.renderer
                    && r.step_effects(dt)
                {
                    hud_dirty = true;
                }

                if hud_dirty && let Some(r) = &mut self.renderer {
                    r.hud_mut().camera_pos = self.camera.position.to_array();
                    r.rebuild_hud();
                }

                if let Some(r) = &mut self.renderer {
                    r.step_particles(dt, self.camera.position);
                    let mut render_cam = self.camera.clone();
                    if self.third_person {
                        let feet = self.camera.position - Vec3::new(0.0, 1.62, 0.0);
                        r.set_local_player_model(Some(RemoteEntityVisual {
                            entity_id: -1,
                            entity_type: generated_entities::id::PLAYER,
                            feet_pos: feet,
                            yaw_deg: 90.0 - self.camera.yaw_rad.to_degrees(),
                        }));
                        render_cam.position = self.camera.position - self.camera.forward() * 3.8
                            + Vec3::new(0.0, 0.35, 0.0);
                    }
                    match r.render(&render_cam) {
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

/// Opens the native `winit` + `wgpu` 3D voxel window with FPS mouse-look, Survival HUD, and Pause Menu.
pub fn run_voxel_window(launch: WindowLaunchConfig) -> Result<()> {
    let event_loop = EventLoop::new().context("create winit event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = VoxelApp {
        cfg: Some(launch),
        base_title: String::new(),
        bridge: None,
        camera: VoxelCamera::default(),
        window: None,
        renderer: None,
        pressed_keys: HashSet::new(),
        paused: false,
        inventory_open: false,
        chat_open: false,
        cursor_locked: false,
        survival_physics: true,
        third_person: false,
        swing_timer: 0.0,
        backpack_cycle_idx: 0,
        last_frame: Instant::now(),
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
    fn mega_buffer_and_mob_outline_meshes_build_cleanly() {
        let mut s0 = ChunkSection::new_air();
        s0.set(1, 1, 1, generated_blocks::default_state::STONE);
        let packed = pack_mega_mesh_batch(&[WorldSectionMesh {
            origin: Vec3::new(32.0, 0.0, 0.0),
            mesh: mesh_section(&s0),
        }]);
        assert_eq!(packed.slots.len(), 1);

        let (verts, idx) = build_entities_and_outline_mesh(
            &[
                RemoteEntityVisual {
                    entity_id: 1,
                    entity_type: generated_entities::id::PIG,
                    feet_pos: Vec3::ZERO,
                    yaw_deg: 0.0,
                },
                RemoteEntityVisual {
                    entity_id: 2,
                    entity_type: generated_entities::id::ZOMBIE,
                    feet_pos: Vec3::new(2.0, 0.0, 0.0),
                    yaw_deg: 90.0,
                },
                RemoteEntityVisual {
                    entity_id: 3,
                    entity_type: generated_entities::id::CREEPER,
                    feet_pos: Vec3::new(4.0, 0.0, 0.0),
                    yaw_deg: 180.0,
                },
                RemoteEntityVisual {
                    entity_id: 4,
                    entity_type: generated_entities::id::SPIDER,
                    feet_pos: Vec3::new(6.0, 0.0, 0.0),
                    yaw_deg: 270.0,
                },
            ],
            Some([4, -59, 4]),
        );
        assert!(verts.len() > 400);
        assert!(idx.len() > 600);
    }
}
