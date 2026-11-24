//! Client-side bit-packed paletted chunk sections, network chunk decoding, and voxel meshing.

pub mod generated_blocks;
pub mod mesh;

use ground_protocol::buf::PacketReader;
use ground_protocol::error::{ProtocolError, ProtocolResult};

pub const SECTION_SIDE: usize = 16;
pub const SECTION_VOLUME: usize = SECTION_SIDE * SECTION_SIDE * SECTION_SIDE; // 4096

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

    /// Decodes a Minecraft 1.18+ / 26.3 paletted block container from a chunk section packet.
    pub fn read_network(r: &mut PacketReader<'_>) -> ProtocolResult<Self> {
        let bits = r.read_u8()?;
        if bits == 0 {
            let single = r.read_varint()? as u32;
            let long_len = r.read_usize_bounded(16)?;
            for _ in 0..long_len {
                let _ = r.read_i64()?;
            }
            return Ok(Self::Single(single));
        }

        let effective_bits = bits.max(4);
        if effective_bits <= 8 {
            let pal_len = r.read_usize_bounded(512)?;
            let mut palette = Vec::with_capacity(pal_len);
            for _ in 0..pal_len {
                palette.push(r.read_varint()? as u32);
            }
            let long_count = r.read_usize_bounded(1024)?;
            let mut longs = Vec::with_capacity(long_count);
            for _ in 0..long_count {
                longs.push(r.read_i64()? as u64);
            }
            let storage = BitStorage::from_longs(effective_bits, SECTION_VOLUME, longs)?;
            Ok(Self::Indirect { palette, storage })
        } else {
            let long_count = r.read_usize_bounded(2048)?;
            let mut longs = Vec::with_capacity(long_count);
            for _ in 0..long_count {
                longs.push(r.read_i64()? as u64);
            }
            let storage = BitStorage::from_longs(effective_bits, SECTION_VOLUME, longs)?;
            Ok(Self::Direct { storage })
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChunkSection {
    non_air_blocks: u16,
    blocks: PalettedBlocks,
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
}
