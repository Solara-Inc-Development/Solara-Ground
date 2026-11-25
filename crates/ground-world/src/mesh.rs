use crate::assets::procedural_block_color;
use crate::{ChunkSection, SECTION_SIDE, generated_blocks};
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;

/// Tightly packed 8-byte voxel vertex designed for minimal GPU VRAM and bandwidth.
///
/// Layout:
/// - `word0`:
///   - bits `0..=4`: `x` in `0..=16`
///   - bits `5..=9`: `y` in `0..=16`
///   - bits `10..=14`: `z` in `0..=16`
///   - bits `15..=17`: normal direction (`0..6`: `-X, +X, -Y, +Y, -Z, +Z`)
///   - bits `18..=19`: ambient occlusion level (`0..=3`)
///   - bits `20..=27`: packed sky/block light (`(sky << 4) | block`)
///   - bits `28..=29`: quad corner UV index (`0..=3`)
/// - `word1`:
///   - packed `RGBA8` albedo color (`r | (g << 8) | (b << 16) | (a << 24)`)
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
        corner: u8,
        rgba: [u8; 4],
    ) -> Self {
        let [x, y, z] = pos;
        let w0 = (x as u32 & 0x1F)
            | ((y as u32 & 0x1F) << 5)
            | ((z as u32 & 0x1F) << 10)
            | ((face as u32 & 0x07) << 15)
            | ((ao as u32 & 0x03) << 18)
            | ((light as u32) << 20)
            | (((corner as u32) & 0x03) << 28);
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
    pub const fn quad_corners(self, x: u8, y: u8, z: u8) -> [(u8, u8, u8); 4] {
        match self {
            Self::NegX => [(x, y, z), (x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z)],
            Self::PosX => [
                (x + 1, y, z + 1),
                (x + 1, y, z),
                (x + 1, y + 1, z),
                (x + 1, y + 1, z + 1),
            ],
            Self::NegY => [(x, y, z + 1), (x, y, z), (x + 1, y, z), (x + 1, y, z + 1)],
            Self::PosY => [
                (x, y + 1, z),
                (x, y + 1, z + 1),
                (x + 1, y + 1, z + 1),
                (x + 1, y + 1, z),
            ],
            Self::NegZ => [(x + 1, y, z), (x, y, z), (x, y + 1, z), (x + 1, y + 1, z)],
            Self::PosZ => [
                (x, y, z + 1),
                (x + 1, y, z + 1),
                (x + 1, y + 1, z + 1),
                (x, y + 1, z + 1),
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
fn is_opaque(state_id: u32) -> bool {
    !generated_blocks::is_air_state(state_id) && state_id != generated_blocks::default_state::GLASS
}

/// Builds a face-culled compact voxel mesh for a single 16x16x16 `ChunkSection`.
pub fn mesh_section(section: &ChunkSection) -> SectionMesh {
    if section.is_empty() {
        return SectionMesh::default();
    }

    let mut mesh = SectionMesh {
        vertices: Vec::with_capacity(512),
        indices: Vec::with_capacity(768),
    };

    for y in 0..SECTION_SIDE {
        for z in 0..SECTION_SIDE {
            for x in 0..SECTION_SIDE {
                let state = section.get(x, y, z);
                if generated_blocks::is_air_state(state) {
                    continue;
                }

                for face in FaceDir::ALL {
                    let (dx, dy, dz) = face.offset();
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    let nz = z as isize + dz;

                    let neighbor_opaque = (0..16).contains(&nx)
                        && (0..16).contains(&ny)
                        && (0..16).contains(&nz)
                        && is_opaque(section.get(nx as usize, ny as usize, nz as usize));

                    if neighbor_opaque {
                        continue;
                    }

                    let rgba = procedural_block_color(state, face == FaceDir::PosY);
                    let base = mesh.vertices.len() as u32;
                    let corners = face.quad_corners(x as u8, y as u8, z as u8);
                    for (corner_idx, &(cx, cy, cz)) in corners.iter().enumerate() {
                        mesh.vertices.push(PackedVoxelVertex::pack(
                            [cx, cy, cz],
                            face,
                            3,
                            0xF0,
                            corner_idx as u8,
                            rgba,
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
                }
            }
        }
    }

    mesh
}

/// Meshes a batch of chunk sections concurrently on the background `rayon` worker pool.
pub fn mesh_sections_parallel(sections: &[ChunkSection]) -> Vec<SectionMesh> {
    sections.par_iter().map(mesh_section).collect()
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
    fn single_block_has_six_faces_and_cube_culls_interior_faces() {
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
        let m_cube = mesh_section(&cube);
        assert_eq!(m_cube.quad_count(), 24);

        let batch = mesh_sections_parallel(&[single, cube]);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].quad_count(), 6);
        assert_eq!(batch[1].quad_count(), 24);
    }
}
