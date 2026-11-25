//! Mojang EULA-compliant local asset resolver and clean-room procedural fallback palette.
//!
//! `Solara-Ground` never bundles or redistributes Mojang game assets. At runtime, official
//! assets are read from the player's local Mojang asset store downloaded by `solara-launcher`
//! directly from Mojang's official CDN (`indexes/*.json` + `objects/<xx>/<sha1>`).

use crate::generated_blocks;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct AssetObjectEntry {
    pub hash: String,
    pub size: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AssetIndexFile {
    pub objects: HashMap<String, AssetObjectEntry>,
}

/// Resolver over a local Minecraft `assets/` directory (`indexes/` + `objects/`).
#[derive(Debug, Clone, Default)]
pub struct LocalMojangAssets {
    root: PathBuf,
    objects: HashMap<String, AssetObjectEntry>,
}

impl LocalMojangAssets {
    /// Opens an existing local `assets/` directory and loads `indexes/<index_id>.json` if present.
    pub fn open(root: impl Into<PathBuf>, index_id: Option<&str>) -> io::Result<Self> {
        let root = root.into();
        let mut objects = HashMap::new();
        if let Some(id) = index_id {
            let index_path = root.join("indexes").join(format!("{id}.json"));
            if index_path.exists() {
                let raw = fs::read_to_string(&index_path)?;
                let parsed: AssetIndexFile = serde_json::from_str(&raw)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                objects = parsed.objects;
            }
        }
        Ok(Self { root, objects })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn indexed_count(&self) -> usize {
        self.objects.len()
    }

    /// Resolves a logical Mojang asset key (e.g. `minecraft/sounds/ambient/cave/cave1.ogg`)
    /// to its content-addressed local file `<root>/objects/<xx>/<sha1>`.
    pub fn resolve_object_path(&self, asset_key: &str) -> Option<PathBuf> {
        let entry = self.objects.get(asset_key)?;
        if entry.hash.len() < 2 {
            return None;
        }
        let prefix = &entry.hash[..2];
        let path = self.root.join("objects").join(prefix).join(&entry.hash);
        path.exists().then_some(path)
    }
}

/// Clean-room procedural RGBA color (`[r, g, b, a]`) used when rendering blocks prior to
/// loading local Mojang PNG block textures, and as minimap/fallback colors.
pub fn procedural_block_color(state_id: u32, is_top_face: bool) -> [u8; 4] {
    let Some(block) = generated_blocks::block_by_state_id(state_id) else {
        return [200, 80, 200, 255];
    };
    match block.name {
        "minecraft:grass_block" => {
            if is_top_face {
                [106, 176, 76, 255]
            } else {
                [134, 96, 67, 255]
            }
        }
        "minecraft:dirt" | "minecraft:coarse_dirt" => [134, 96, 67, 255],
        "minecraft:stone" | "minecraft:andesite" => [128, 128, 132, 255],
        "minecraft:cobblestone" => [110, 110, 114, 255],
        "minecraft:deepslate" => [72, 72, 78, 255],
        "minecraft:bedrock" => [52, 52, 56, 255],
        "minecraft:sand" => [219, 207, 163, 255],
        "minecraft:water" => [52, 116, 212, 200],
        "minecraft:oak_log" => [109, 85, 50, 255],
        "minecraft:oak_planks" => [162, 130, 78, 255],
        "minecraft:glass" => [210, 235, 245, 128],
        other => {
            // Deterministic hash tint so every distinct block state looks distinguishable
            let mut h = 2166136261u32;
            for &b in other.as_bytes() {
                h ^= u32::from(b);
                h = h.wrapping_mul(16777619);
            }
            [
                96 + ((h & 0x7F) as u8),
                96 + (((h >> 8) & 0x7F) as u8),
                96 + (((h >> 16) & 0x7F) as u8),
                255,
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grass_block_has_green_top_and_brown_sides() {
        let grass = generated_blocks::default_state::GRASS_BLOCK;
        let top = procedural_block_color(grass, true);
        let side = procedural_block_color(grass, false);
        assert!(top[1] > top[0], "top of grass block should be green");
        assert!(
            side[0] > side[1],
            "side of grass block should be earthy brown"
        );
    }
}
