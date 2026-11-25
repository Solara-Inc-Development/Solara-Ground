use crate::assets::block_face_material;
use crate::{ChunkSection, SECTION_SIDE, generated_blocks};
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;

/// Tightly packed 8-byte voxel vertex designed for minimal GPU VRAM and bandwidth.
///
/// Layout:
/// - `word0`:
///   - bits `0..=4`: `x` in `0..=16` (supports full-section greedy merged quads!)
///   - bits `5..=9`: `y` in `0..=16`
///   - bits `10..=14`: `z` in `0..=16`
///   - bits `15..=17`: normal direction (`0..6`: `-X, +X, -Y, +Y, -Z, +Z`)
///   - bits `18..=19`: ambient occlusion level (`0..=3`)
///   - bits `20..=27`: packed sky/block light (`(sky << 4) | block`)
///   - bits `28..=31`: 2D block texture array layer (`0..=15`)
/// - `word1`:
///   - packed `RGBA8` tint (`r | (g << 8) | (b << 16) | (a << 24)`)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct PackedVoxelVertex {
    pub word0: u32,
    pub word1: u32,
}

const _: () = assert!(std::mem::size_of::<PackedVoxelVertex>() == 8);

impl PackedVoxelVertex {
    #[inline]
    pub const fn pack(
        pos: [u8; 3],
        face: FaceDir,
        ao: u8,
        light: u8,
        tex_layer: u8,
        rgba: [u8; 4],
    ) -> Self {
        let [x, y, z] = pos;
        let w0 = (x as u32 & 0x1F)
            | ((y as u32 & 0x1F) << 5)
            | ((z as u32 & 0x1F) << 10)
            | ((face as u32 & 0x07) << 15)
            | ((ao as u32 & 0x03) << 18)
            | ((light as u32) << 20)
            | (((tex_layer as u32) & 0x0F) << 28);
        let w1 = (rgba[0] as u32)
            | ((rgba[1] as u32) << 8)
            | ((rgba[2] as u32) << 16)
            | ((rgba[3] as u32) << 24);
        Self {
            word0: w0,
            word1: w1,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceDir {
    NegX = 0,
    PosX = 1,
    NegY = 2,
    PosY = 3,
    NegZ = 4,
    PosZ = 5,
}

impl FaceDir {
    pub const ALL: [Self; 6] = [
        Self::NegX,
        Self::PosX,
        Self::NegY,
        Self::PosY,
        Self::NegZ,
        Self::PosZ,
    ];

    #[inline]
    pub const fn offset(self) -> (isize, isize, isize) {
        match self {
            Self::NegX => (-1, 0, 0),
            Self::PosX => (1, 0, 0),
            Self::NegY => (0, -1, 0),
            Self::PosY => (0, 1, 0),
            Self::NegZ => (0, 0, -1),
            Self::PosZ => (0, 0, 1),
        }
    }

    #[inline]
    const fn coords_from_slice_uv(self, s: usize, u: usize, v: usize) -> (usize, usize, usize) {
        match self {
            Self::NegX | Self::PosX => (s, v, u),
            Self::NegY | Self::PosY => (u, s, v),
            Self::NegZ | Self::PosZ => (u, v, s),
        }
    }

    #[inline]
    pub const fn quad_corners_sized(self, s: u8, u: u8, v: u8, w: u8, h: u8) -> [(u8, u8, u8); 4] {
        match self {
            Self::NegX => [(s, v, u), (s, v, u + w), (s, v + h, u + w), (s, v + h, u)],
            Self::PosX => [
                (s + 1, v, u + w),
                (s + 1, v, u),
                (s + 1, v + h, u),
                (s + 1, v + h, u + w),
            ],
            Self::NegY => [(u, s, v + h), (u, s, v), (u + w, s, v), (u + w, s, v + h)],
            Self::PosY => [
                (u, s + 1, v),
                (u, s + 1, v + h),
                (u + w, s + 1, v + h),
                (u + w, s + 1, v),
            ],
            Self::NegZ => [(u + w, v, s), (u, v, s), (u, v + h, s), (u + w, v + h, s)],
            Self::PosZ => [
                (u, v, s + 1),
                (u + w, v, s + 1),
                (u + w, v + h, s + 1),
                (u, v + h, s + 1),
            ],
        }
    }
}

/// CPU-side mesh output ready for upload to `wgpu` vertex & index buffers.
#[derive(Debug, Default, Clone)]
pub struct SectionMesh {
    pub vertices: Vec<PackedVoxelVertex>,
    pub indices: Vec<u32>,
}

impl SectionMesh {
    pub fn quad_count(&self) -> usize {
        self.indices.len() / 6
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

#[inline]
fn is_opaque_occluder(state_id: u32) -> bool {
    !generated_blocks::is_air_state(state_id)
        && state_id != generated_blocks::default_state::GLASS
        && state_id != generated_blocks::default_state::WATER
}

#[inline]
fn should_cull_face(self_state: u32, neighbor_state: u32) -> bool {
    if generated_blocks::is_air_state(neighbor_state) {
        return false;
    }
    if self_state == neighbor_state
        && (self_state == generated_blocks::default_state::WATER
            || self_state == generated_blocks::default_state::GLASS)
    {
        return true;
    }
    is_opaque_occluder(neighbor_state)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceKey {
    tex_layer: u8,
    rgba: [u8; 4],
    light: u8,
    ao: u8,
}

#[inline]
fn sample_column_block(
    sec: &ChunkSection,
    below: Option<&ChunkSection>,
    above: Option<&ChunkSection>,
    x: isize,
    y: isize,
    z: isize,
) -> u32 {
    if !(0..16).contains(&x) || !(0..16).contains(&z) {
        return generated_blocks::default_state::AIR;
    }
    let (ux, uz) = (x as usize, z as usize);
    if (0..16).contains(&y) {
        sec.get(ux, y as usize, uz)
    } else if y < 0 {
        below.map_or(generated_blocks::default_state::AIR, |b| b.get(ux, 15, uz))
    } else {
        above.map_or(generated_blocks::default_state::AIR, |a| a.get(ux, 0, uz))
    }
}

#[inline]
fn sample_face_light(
    sec: &ChunkSection,
    below: Option<&ChunkSection>,
    above: Option<&ChunkSection>,
    pos: (usize, usize, usize),
    npos: (isize, isize, isize),
    self_state: u32,
) -> u8 {
    let emissive = if self_state == generated_blocks::default_state::GLOWSTONE {
        15u8
    } else {
        0u8
    };
    let (x, y, z) = pos;
    let (nx, ny, nz) = npos;

    let (sky, blk) = if (0..16).contains(&nx) && (0..16).contains(&nz) {
        let (ux, uz) = (nx as usize, nz as usize);
        if (0..16).contains(&ny) {
            let uy = ny as usize;
            (sec.sky_light(ux, uy, uz), sec.block_light(ux, uy, uz))
        } else if ny < 0 {
            below.map_or_else(
                || (if sec.is_uniform_dark_sky() { 0 } else { 15 }, 0),
                |b| (b.sky_light(ux, 15, uz), b.block_light(ux, 15, uz)),
            )
        } else {
            above.map_or((15, 0), |a| {
                (a.sky_light(ux, 0, uz), a.block_light(ux, 0, uz))
            })
        }
    } else {
        let sky_est = if sec.is_uniform_dark_sky() {
            0
        } else if y + 1 < 16 && !is_opaque_occluder(sec.get(x, y + 1, z)) {
            sec.sky_light(x, y + 1, z)
        } else {
            13
        };
        (sky_est, sec.block_light(x, y, z))
    };

    ((sky & 0x0F) << 4) | (blk.max(emissive) & 0x0F)
}

#[inline]
fn sample_face_ao(
    sec: &ChunkSection,
    below: Option<&ChunkSection>,
    above: Option<&ChunkSection>,
    npos: (isize, isize, isize),
    face: FaceDir,
) -> u8 {
    let (nx, ny, nz) = npos;
    let tangents: [(isize, isize, isize); 4] = match face {
        FaceDir::NegX | FaceDir::PosX => [(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],
        FaceDir::NegY | FaceDir::PosY => [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)],
        FaceDir::NegZ | FaceDir::PosZ => [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)],
    };
    let mut occ = 0usize;
    for (tx, ty, tz) in tangents {
        if is_opaque_occluder(sample_column_block(
            sec,
            below,
            above,
            nx + tx,
            ny + ty,
            nz + tz,
        )) {
            occ += 1;
        }
    }
    3u8.saturating_sub(occ.min(3) as u8)
}

/// Builds a 2D greedy-merged, face-culled, light/AO-baked compact voxel mesh for `section`,
/// optionally reading vertical neighbor sections `below` and `above` to eliminate seam faces.
pub fn mesh_section_with_neighbors(
    section: &ChunkSection,
    below: Option<&ChunkSection>,
    above: Option<&ChunkSection>,
) -> SectionMesh {
    if section.is_empty() {
        return SectionMesh::default();
    }

    let mut mesh = SectionMesh {
        vertices: Vec::with_capacity(256),
        indices: Vec::with_capacity(384),
    };

    for face in FaceDir::ALL {
        let (dx, dy, dz) = face.offset();
        let is_top = face == FaceDir::PosY;
        let is_bottom = face == FaceDir::NegY;

        for s in 0..SECTION_SIDE {
            let mut mask = [None::<FaceKey>; SECTION_SIDE * SECTION_SIDE];
            let mut any_visible = false;

            for v in 0..SECTION_SIDE {
                for u in 0..SECTION_SIDE {
                    let (x, y, z) = face.coords_from_slice_uv(s, u, v);
                    let state = section.get(x, y, z);
                    if generated_blocks::is_air_state(state) {
                        continue;
                    }

                    let (nx, ny, nz) = (x as isize + dx, y as isize + dy, z as isize + dz);
                    let neighbor = sample_column_block(section, below, above, nx, ny, nz);
                    if should_cull_face(state, neighbor) {
                        continue;
                    }

                    let (tex_layer, rgba) = block_face_material(state, is_top, is_bottom);
                    let light =
                        sample_face_light(section, below, above, (x, y, z), (nx, ny, nz), state);
                    let ao = sample_face_ao(section, below, above, (nx, ny, nz), face);

                    mask[v * SECTION_SIDE + u] = Some(FaceKey {
                        tex_layer,
                        rgba,
                        light,
                        ao,
                    });
                    any_visible = true;
                }
            }

            if !any_visible {
                continue;
            }

            for v in 0..SECTION_SIDE {
                let mut u = 0usize;
                while u < SECTION_SIDE {
                    let Some(key) = mask[v * SECTION_SIDE + u] else {
                        u += 1;
                        continue;
                    };

                    let mut w = 1usize;
                    while u + w < SECTION_SIDE && mask[v * SECTION_SIDE + u + w] == Some(key) {
                        w += 1;
                    }

                    let mut h = 1usize;
                    'grow_h: while v + h < SECTION_SIDE {
                        for du in 0..w {
                            if mask[(v + h) * SECTION_SIDE + u + du] != Some(key) {
                                break 'grow_h;
                            }
                        }
                        h += 1;
                    }

                    for dv in 0..h {
                        for du in 0..w {
                            mask[(v + dv) * SECTION_SIDE + u + du] = None;
                        }
                    }

                    let base = mesh.vertices.len() as u32;
                    let corners =
                        face.quad_corners_sized(s as u8, u as u8, v as u8, w as u8, h as u8);
                    for &(cx, cy, cz) in &corners {
                        mesh.vertices.push(PackedVoxelVertex::pack(
                            [cx, cy, cz],
                            face,
                            key.ao,
                            key.light,
                            key.tex_layer,
                            key.rgba,
                        ));
                    }
                    mesh.indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);

                    u += w;
                }
            }
        }
    }

    mesh
}

/// Builds a 2D greedy-merged voxel mesh for an isolated `ChunkSection`.
pub fn mesh_section(section: &ChunkSection) -> SectionMesh {
    mesh_section_with_neighbors(section, None, None)
}

/// Meshes a full vertical chunk column (`sections[0..N]`) with cross-section vertical neighbor culling.
pub fn mesh_chunk_column(sections: &[ChunkSection]) -> Vec<SectionMesh> {
    (0..sections.len())
        .map(|i| {
            let below = i.checked_sub(1).and_then(|idx| sections.get(idx));
            let above = sections.get(i + 1);
            mesh_section_with_neighbors(&sections[i], below, above)
        })
        .collect()
}

/// Meshes a batch of chunk sections concurrently on the background `rayon` worker pool.
pub fn mesh_sections_parallel(sections: &[ChunkSection]) -> Vec<SectionMesh> {
    (0..sections.len())
        .into_par_iter()
        .map(|i| {
            let below = i.checked_sub(1).and_then(|idx| sections.get(idx));
            let above = sections.get(i + 1);
            mesh_section_with_neighbors(&sections[i], below, above)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_section_produces_zero_vertices() {
        let sec = ChunkSection::new_air();
        let mesh = mesh_section(&sec);
        assert!(mesh.is_empty());
    }

    #[test]
    fn single_block_and_2x2x2_cube_greedy_merge_to_six_quads() {
        let stone = generated_blocks::default_state::STONE;
        let mut single = ChunkSection::new_air();
        single.set(4, 4, 4, stone);
        let m1 = mesh_section(&single);
        assert_eq!(m1.quad_count(), 6);
        assert_eq!(m1.vertices.len(), 24);
        assert_eq!(m1.indices.len(), 36);

        let mut cube = ChunkSection::new_air();
        for dx in 0..2 {
            for dy in 0..2 {
                for dz in 0..2 {
                    cube.set(2 + dx, 2 + dy, 2 + dz, stone);
                }
            }
        }
        // Greedy mesher merges each 2x2 face into 1 quad -> 6 quads total (down from 24!)
        let m_cube = mesh_section(&cube);
        assert_eq!(m_cube.quad_count(), 6);
    }

    #[test]
    fn greedy_mesher_merges_16x16_flat_floor_from_576_down_to_6_quads() {
        let stone = generated_blocks::default_state::STONE;
        let mut floor = ChunkSection::new_air();
        for z in 0..16 {
            for x in 0..16 {
                floor.set(x, 4, z, stone);
            }
        }
        let mesh = mesh_section(&floor);
        // 1 top (16x16) + 1 bottom (16x16) + 4 side borders (16x1) = 6 quads total!
        assert_eq!(mesh.quad_count(), 6);
    }
}
