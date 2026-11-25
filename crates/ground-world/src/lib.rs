//! Client-side bit-packed paletted chunk sections, MC 26.3 network chunk + light decoding,
//! Mojang local asset resolution, and compact greedy voxel meshing.

pub mod assets;
pub mod generated_blocks;
pub mod mesh;
pub mod physics;

use ground_protocol::buf::PacketReader;
use ground_protocol::error::{ProtocolError, ProtocolResult};
use ground_protocol::nbt::read_network_nbt;
use rayon::prelude::*;

pub const SECTION_SIDE: usize = 16;
pub const SECTION_VOLUME: usize = SECTION_SIDE * SECTION_SIDE * SECTION_SIDE; // 4096
pub const OVERWORLD_MIN_Y: i32 = -64;
pub const OVERWORLD_SECTIONS: usize = 24;

/// Returns the number of active `rayon` CPU worker threads used for parallel chunk decoding & meshing.
pub fn worker_threads() -> usize {
    rayon::current_num_threads()
}

/// Decoded 24-section Overworld chunk column paired with its 2D greedy-meshed `SectionMesh`es.
#[derive(Debug, Clone)]
pub struct DecodedChunkColumn {
    pub pos: ChunkPos,
    pub sections: Vec<ChunkSection>,
    pub meshes: Vec<mesh::SectionMesh>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkPos {
    pub x: i32,
    pub z: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitStorage {
    bits_per_entry: u8,
    entries_per_long: usize,
    mask: u64,
    len: usize,
    data: Vec<u64>,
}

impl BitStorage {
    pub fn new(bits_per_entry: u8, len: usize) -> Self {
        assert!((1..=32).contains(&bits_per_entry));
        let entries_per_long = 64 / usize::from(bits_per_entry);
        let mask = (1u64 << bits_per_entry) - 1;
        let num_longs = len.div_ceil(entries_per_long);
        Self {
            bits_per_entry,
            entries_per_long,
            mask,
            len,
            data: vec![0u64; num_longs],
        }
    }

    pub fn from_longs(bits_per_entry: u8, len: usize, data: Vec<u64>) -> ProtocolResult<Self> {
        if !(1..=32).contains(&bits_per_entry) {
            return Err(ProtocolError::LengthTooLarge {
                len: usize::from(bits_per_entry),
                max: 32,
            });
        }
        let entries_per_long = 64 / usize::from(bits_per_entry);
        let expected_longs = len.div_ceil(entries_per_long);
        if data.len() != expected_longs {
            return Err(ProtocolError::LengthTooLarge {
                len: data.len(),
                max: expected_longs,
            });
        }
        let mask = (1u64 << bits_per_entry) - 1;
        Ok(Self {
            bits_per_entry,
            entries_per_long,
            mask,
            len,
            data,
        })
    }

    pub fn bits_per_entry(&self) -> u8 {
        self.bits_per_entry
    }

    pub fn get(&self, index: usize) -> u32 {
        debug_assert!(index < self.len);
        let long_idx = index / self.entries_per_long;
        let bit_offset = (index % self.entries_per_long) * usize::from(self.bits_per_entry);
        ((self.data[long_idx] >> bit_offset) & self.mask) as u32
    }

    pub fn set(&mut self, index: usize, value: u32) {
        debug_assert!(index < self.len);
        let v = u64::from(value) & self.mask;
        let long_idx = index / self.entries_per_long;
        let bit_offset = (index % self.entries_per_long) * usize::from(self.bits_per_entry);
        self.data[long_idx] =
            (self.data[long_idx] & !(self.mask << bit_offset)) | (v << bit_offset);
    }
}

#[derive(Debug, Clone)]
pub enum PalettedBlocks {
    Single(u32),
    Indirect {
        palette: Vec<u32>,
        storage: BitStorage,
    },
    Direct {
        storage: BitStorage,
    },
}

impl PalettedBlocks {
    const GLOBAL_BITS: u8 = 16;

    pub fn filled(state_id: u32) -> Self {
        Self::Single(state_id)
    }

    pub fn get(&self, index: usize) -> u32 {
        match self {
            Self::Single(id) => *id,
            Self::Indirect { palette, storage } => {
                let p_idx = storage.get(index) as usize;
                palette.get(p_idx).copied().unwrap_or(0)
            }
            Self::Direct { storage } => storage.get(index),
        }
    }

    pub fn set(&mut self, index: usize, state_id: u32) {
        match self {
            Self::Single(current) => {
                if *current == state_id {
                    return;
                }
                let old = *current;
                let mut storage = BitStorage::new(4, SECTION_VOLUME);
                storage.set(index, 1);
                *self = Self::Indirect {
                    palette: vec![old, state_id],
                    storage,
                };
            }
            Self::Indirect { palette, storage } => {
                if let Some(p_idx) = palette.iter().position(|&s| s == state_id) {
                    storage.set(index, p_idx as u32);
                    return;
                }
                let new_p_idx = palette.len();
                let max_entries = 1usize << storage.bits_per_entry();
                if new_p_idx < max_entries {
                    palette.push(state_id);
                    storage.set(index, new_p_idx as u32);
                } else if storage.bits_per_entry() < 8 {
                    let next_bits = storage.bits_per_entry() + 1;
                    let mut next_storage = BitStorage::new(next_bits, SECTION_VOLUME);
                    for i in 0..SECTION_VOLUME {
                        next_storage.set(i, storage.get(i));
                    }
                    palette.push(state_id);
                    next_storage.set(index, new_p_idx as u32);
                    *storage = next_storage;
                } else {
                    let mut direct = BitStorage::new(Self::GLOBAL_BITS, SECTION_VOLUME);
                    for i in 0..SECTION_VOLUME {
                        direct.set(i, palette[storage.get(i) as usize]);
                    }
                    direct.set(index, state_id);
                    *self = Self::Direct { storage: direct };
                }
            }
            Self::Direct { storage } => storage.set(index, state_id),
        }
    }

    /// Decodes a Minecraft 26.3 `PalettedContainer` (`writeFixedSizeLongArray`, no VarInt length prefix).
    pub fn read_network_container(
        r: &mut PacketReader<'_>,
        volume: usize,
        max_indirect_bits: u8,
    ) -> ProtocolResult<Self> {
        let bits = r.read_u8()?;
        if bits == 0 {
            let single = r.read_varint()? as u32;
            return Ok(Self::Single(single));
        }

        let entries_per_long = 64 / usize::from(bits);
        if entries_per_long == 0 {
            return Err(ProtocolError::LengthTooLarge {
                len: usize::from(bits),
                max: 32,
            });
        }
        let long_count = volume.div_ceil(entries_per_long);

        if bits <= max_indirect_bits {
            let pal_len = r.read_usize_bounded(512)?;
            let mut palette = Vec::with_capacity(pal_len);
            for _ in 0..pal_len {
                palette.push(r.read_varint()? as u32);
            }
            let mut longs = Vec::with_capacity(long_count);
            for _ in 0..long_count {
                longs.push(r.read_i64()? as u64);
            }
            let storage = BitStorage::from_longs(bits, volume, longs)?;
            Ok(Self::Indirect { palette, storage })
        } else {
            let mut longs = Vec::with_capacity(long_count);
            for _ in 0..long_count {
                longs.push(r.read_i64()? as u64);
            }
            let storage = BitStorage::from_longs(bits, volume, longs)?;
            Ok(Self::Direct { storage })
        }
    }
}

/// Memory-compact 4-bit section light storage: avoids allocating `2048` bytes for uniform sections.
#[derive(Debug, Clone)]
pub enum SectionLight {
    Uniform(u8),
    Nibbles(Box<[u8; SECTION_VOLUME / 2]>),
}

impl SectionLight {
    pub fn from_nibble_slice(bytes: &[u8]) -> Self {
        if bytes.len() != SECTION_VOLUME / 2 {
            return Self::Uniform(0);
        }
        let first = bytes[0];
        let low = first & 0x0F;
        if (first >> 4) == low && bytes.iter().all(|&b| b == first) {
            return Self::Uniform(low);
        }
        let mut boxed = Box::new([0u8; SECTION_VOLUME / 2]);
        boxed.copy_from_slice(bytes);
        Self::Nibbles(boxed)
    }

    #[inline]
    pub fn get(&self, index: usize) -> u8 {
        match self {
            Self::Uniform(v) => *v,
            Self::Nibbles(arr) => {
                let byte = arr[index >> 1];
                if (index & 1) == 0 {
                    byte & 0x0F
                } else {
                    byte >> 4
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChunkSection {
    non_air_blocks: u16,
    blocks: PalettedBlocks,
    sky_light: SectionLight,
    block_light: SectionLight,
}

impl Default for ChunkSection {
    fn default() -> Self {
        Self::new_air()
    }
}

impl ChunkSection {
    pub fn new_air() -> Self {
        Self {
            non_air_blocks: 0,
            blocks: PalettedBlocks::filled(generated_blocks::default_state::AIR),
            sky_light: SectionLight::Uniform(15),
            block_light: SectionLight::Uniform(0),
        }
    }

    #[inline]
    pub const fn section_index(x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < 16 && y < 16 && z < 16);
        (y << 8) | (z << 4) | x
    }

    pub fn is_empty(&self) -> bool {
        self.non_air_blocks == 0
    }

    pub fn non_air_count(&self) -> u16 {
        self.non_air_blocks
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> u32 {
        self.blocks.get(Self::section_index(x, y, z))
    }

    #[inline]
    pub fn sky_light(&self, x: usize, y: usize, z: usize) -> u8 {
        self.sky_light.get(Self::section_index(x, y, z))
    }

    #[inline]
    pub fn block_light(&self, x: usize, y: usize, z: usize) -> u8 {
        self.block_light.get(Self::section_index(x, y, z))
    }

    #[inline]
    pub fn is_uniform_dark_sky(&self) -> bool {
        matches!(self.sky_light, SectionLight::Uniform(0))
    }

    pub fn set_sky_light_array(&mut self, bytes: &[u8]) {
        self.sky_light = SectionLight::from_nibble_slice(bytes);
    }

    pub fn set_block_light_array(&mut self, bytes: &[u8]) {
        self.block_light = SectionLight::from_nibble_slice(bytes);
    }

    pub fn set_uniform_sky(&mut self, level: u8) {
        self.sky_light = SectionLight::Uniform(level & 0x0F);
    }

    pub fn set_uniform_block(&mut self, level: u8) {
        self.block_light = SectionLight::Uniform(level & 0x0F);
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, state_id: u32) {
        let idx = Self::section_index(x, y, z);
        let prev = self.blocks.get(idx);
        if prev == state_id {
            return;
        }
        let prev_air = generated_blocks::is_air_state(prev);
        let next_air = generated_blocks::is_air_state(state_id);
        if prev_air && !next_air {
            self.non_air_blocks += 1;
        } else if !prev_air && next_air {
            self.non_air_blocks -= 1;
        }
        self.blocks.set(idx, state_id);
    }

    /// Decodes a single MC 26.3 `LevelChunkSection`:
    /// `i16 nonEmptyBlockCount`, `i16 fluidCount`, `PalettedContainer(4096) states`, `PalettedContainer(64) biomes`.
    pub fn read_network(r: &mut PacketReader<'_>) -> ProtocolResult<Self> {
        let non_air_raw = r.read_i16()?;
        let _fluid_raw = r.read_i16()?;
        let blocks = PalettedBlocks::read_network_container(r, SECTION_VOLUME, 8)?;
        let _biomes = PalettedBlocks::read_network_container(r, 64, 3)?;
        Ok(Self {
            non_air_blocks: non_air_raw.max(0) as u16,
            blocks,
            sky_light: SectionLight::Uniform(15),
            block_light: SectionLight::Uniform(0),
        })
    }
}

fn read_bitset(r: &mut PacketReader<'_>) -> ProtocolResult<Vec<u64>> {
    let word_count = r.read_usize_bounded(16)?;
    let mut words = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        words.push(r.read_i64()? as u64);
    }
    Ok(words)
}

#[inline]
fn bitset_get(words: &[u64], bit: usize) -> bool {
    words
        .get(bit >> 6)
        .is_some_and(|w| ((w >> (bit & 63)) & 1) != 0)
}

/// Decodes a full `ClientboundLevelChunkWithLightPacket` payload into `(ChunkPos, Vec<ChunkSection>)`,
/// including block-entity skipping and 4-bit Sky & Block light nibble arrays.
pub fn decode_level_chunk_with_light(
    payload: &[u8],
    section_count: usize,
) -> ProtocolResult<(ChunkPos, Vec<ChunkSection>)> {
    let mut r = PacketReader::new(payload);
    let cx = r.read_i32()?;
    let cz = r.read_i32()?;

    // Skip HEIGHTMAPS_STREAM_CODEC: Map<VarInt, LongArray>
    let hm_count = r.read_usize_bounded(16)?;
    for _ in 0..hm_count {
        let _hm_type = r.read_varint()?;
        let longs_len = r.read_usize_bounded(512)?;
        let _ = r.read_exact(longs_len * 8)?;
    }

    let chunk_bytes = r.read_byte_array(2 * 1024 * 1024)?;
    let mut sr = PacketReader::new(&chunk_bytes);
    let mut sections = Vec::with_capacity(section_count);
    for _ in 0..section_count {
        sections.push(ChunkSection::read_network(&mut sr)?);
    }

    // Skip List<BlockEntityInfo>
    let be_count = r.read_usize_bounded(4096)?;
    for _ in 0..be_count {
        let _packed_xz = r.read_u8()?;
        let _y = r.read_i16()?;
        let _be_type = r.read_varint()?;
        let _optional_nbt = read_network_nbt(&mut r)?;
    }

    if r.remaining() > 0 {
        let sky_mask = read_bitset(&mut r)?;
        let block_mask = read_bitset(&mut r)?;
        let empty_sky_mask = read_bitset(&mut r)?;
        let empty_block_mask = read_bitset(&mut r)?;

        for (sec_idx, sec) in sections.iter_mut().enumerate() {
            if bitset_get(&empty_sky_mask, sec_idx + 1) {
                sec.set_uniform_sky(0);
            }
            if bitset_get(&empty_block_mask, sec_idx + 1) {
                sec.set_uniform_block(0);
            }
        }

        let sky_len = r.read_usize_bounded(64)?;
        let mut sky_bits = (0..(sky_mask.len() * 64)).filter(|&b| bitset_get(&sky_mask, b));
        for _ in 0..sky_len {
            let arr = r.read_byte_array(2048)?;
            if let Some(light_sec) = sky_bits.next()
                && (1..=section_count).contains(&light_sec)
            {
                sections[light_sec - 1].set_sky_light_array(&arr);
            }
        }

        let block_len = r.read_usize_bounded(64)?;
        let mut block_bits = (0..(block_mask.len() * 64)).filter(|&b| bitset_get(&block_mask, b));
        for _ in 0..block_len {
            let arr = r.read_byte_array(2048)?;
            if let Some(light_sec) = block_bits.next()
                && (1..=section_count).contains(&light_sec)
            {
                sections[light_sec - 1].set_block_light_array(&arr);
            }
        }
    }

    Ok((ChunkPos { x: cx, z: cz }, sections))
}

/// Decodes raw `LevelChunkWithLight` network packets, unpacks 4-bit Sky & Block lighting,
/// and builds 2D greedy-merged section meshes concurrently across all CPU cores via `rayon`.
pub fn decode_and_mesh_chunks_parallel(
    raw_packets: &[Vec<u8>],
    section_count: usize,
) -> ProtocolResult<Vec<DecodedChunkColumn>> {
    raw_packets
        .par_iter()
        .map(|raw| {
            let (pos, sections) = decode_level_chunk_with_light(raw, section_count)?;
            let meshes = mesh::mesh_chunk_column(&sections);
            Ok(DecodedChunkColumn {
                pos,
                sections,
                meshes,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_light_compresses_uniform_and_unpacks_nibbles() {
        let uniform = SectionLight::from_nibble_slice(&[0xFF; 2048]);
        assert!(matches!(uniform, SectionLight::Uniform(15)));
        assert_eq!(uniform.get(0), 15);
        assert_eq!(uniform.get(4095), 15);

        let mut mixed = [0u8; 2048];
        mixed[0] = 0xE5; // index 0 = 5, index 1 = 14
        let nibbles = SectionLight::from_nibble_slice(&mixed);
        assert!(matches!(nibbles, SectionLight::Nibbles(_)));
        assert_eq!(nibbles.get(0), 5);
        assert_eq!(nibbles.get(1), 14);
        assert_eq!(nibbles.get(2), 0);
    }
}
