//! Mojang EULA-compliant local asset resolver, PNG block texture loader, and clean-room
//! procedural `16x16` pixel-art fallback texture array.
//!
//! `Solara-Ground` never bundles or redistributes Mojang game assets. At runtime, official
//! assets are read from the player's local Mojang asset store downloaded by `solara-launcher`
//! directly from Mojang's official CDN (`indexes/*.json` + `objects/<xx>/<sha1>`), or from
//! an extracted local `minecraft/textures/block/*.png` resource directory.

use crate::generated_blocks;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Cursor};
use std::path::{Path, PathBuf};

pub const TILE_SIZE: u32 = 16;
pub const TILE_BYTES: usize = (TILE_SIZE as usize) * (TILE_SIZE as usize) * 4; // 1024
pub const ATLAS_LAYER_COUNT: u32 = 16;

pub const LAYER_GRASS_TOP: u8 = 0;
pub const LAYER_GRASS_SIDE: u8 = 1;
pub const LAYER_DIRT: u8 = 2;
pub const LAYER_STONE: u8 = 3;
pub const LAYER_COBBLESTONE: u8 = 4;
pub const LAYER_OAK_PLANKS: u8 = 5;
pub const LAYER_OAK_LOG_SIDE: u8 = 6;
pub const LAYER_OAK_LOG_TOP: u8 = 7;
pub const LAYER_OAK_LEAVES: u8 = 8;
pub const LAYER_SAND: u8 = 9;
pub const LAYER_WATER: u8 = 10;
pub const LAYER_GLASS: u8 = 11;
pub const LAYER_GLOWSTONE: u8 = 12;
pub const LAYER_DEEPSLATE: u8 = 13;
pub const LAYER_BEDROCK: u8 = 14;
pub const LAYER_TINTED_SOLID: u8 = 15;

#[derive(Debug, Clone, Deserialize)]
pub struct AssetObjectEntry {
    pub hash: String,
    pub size: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AssetIndexFile {
    pub objects: HashMap<String, AssetObjectEntry>,
}

/// CPU-side `16 x 16 x 16` RGBA8 2D texture array ready for `wgpu` upload.
#[derive(Debug, Clone)]
pub struct BlockTextureAtlas {
    pub pixels: Vec<u8>,
    pub loaded_mojang_pngs: usize,
}

impl Default for BlockTextureAtlas {
    fn default() -> Self {
        LocalMojangAssets::default().build_block_atlas()
    }
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

    /// Reads raw PNG bytes for `minecraft/textures/block/<name>.png` from either the Mojang
    /// content-addressed object store or a local extracted `minecraft/textures/block/` folder.
    pub fn read_block_png_bytes(&self, png_name: &str) -> Option<Vec<u8>> {
        let key = format!("minecraft/textures/block/{png_name}");
        if let Some(obj_path) = self.resolve_object_path(&key)
            && let Ok(bytes) = fs::read(obj_path)
        {
            return Some(bytes);
        }
        let direct = self
            .root
            .join("minecraft")
            .join("textures")
            .join("block")
            .join(png_name);
        if direct.exists()
            && let Ok(bytes) = fs::read(direct)
        {
            return Some(bytes);
        }
        None
    }

    /// Builds the 16-layer `16x16` RGBA8 block texture array, loading local Mojang PNGs when
    /// available and falling back to clean-room procedural pixel-art tiles otherwise.
    pub fn build_block_atlas(&self) -> BlockTextureAtlas {
        type LayerSpec = (u8, Option<&'static str>, Option<[u8; 4]>);
        const LAYER_SPECS: [LayerSpec; 16] = [
            (
                LAYER_GRASS_TOP,
                Some("grass_block_top.png"),
                Some([112, 184, 82, 255]),
            ),
            (LAYER_GRASS_SIDE, Some("grass_block_side.png"), None),
            (LAYER_DIRT, Some("dirt.png"), None),
            (LAYER_STONE, Some("stone.png"), None),
            (LAYER_COBBLESTONE, Some("cobblestone.png"), None),
            (LAYER_OAK_PLANKS, Some("oak_planks.png"), None),
            (LAYER_OAK_LOG_SIDE, Some("oak_log.png"), None),
            (LAYER_OAK_LOG_TOP, Some("oak_log_top.png"), None),
            (
                LAYER_OAK_LEAVES,
                Some("oak_leaves.png"),
                Some([84, 162, 62, 255]),
            ),
            (LAYER_SAND, Some("sand.png"), None),
            (
                LAYER_WATER,
                Some("water_still.png"),
                Some([63, 118, 228, 200]),
            ),
            (LAYER_GLASS, Some("glass.png"), None),
            (LAYER_GLOWSTONE, Some("glowstone.png"), None),
            (LAYER_DEEPSLATE, Some("deepslate.png"), None),
            (LAYER_BEDROCK, Some("bedrock.png"), None),
            (LAYER_TINTED_SOLID, None, None),
        ];

        let mut pixels = vec![0u8; TILE_BYTES * (ATLAS_LAYER_COUNT as usize)];
        let mut loaded_mojang_pngs = 0usize;

        for (layer, maybe_png, grayscale_tint) in LAYER_SPECS {
            let dst_offset = usize::from(layer) * TILE_BYTES;
            let dst = &mut pixels[dst_offset..(dst_offset + TILE_BYTES)];

            if let Some(png_name) = maybe_png
                && let Some(raw_bytes) = self.read_block_png_bytes(png_name)
                && let Some(mut tile) = decode_png_16x16_rgba(&raw_bytes)
            {
                if let Some(tint) = grayscale_tint {
                    apply_biome_tint_if_grayscale(&mut tile, tint);
                }
                dst.copy_from_slice(&tile);
                loaded_mojang_pngs += 1;
                continue;
            }

            let proc_tile = generate_procedural_tile(layer);
            dst.copy_from_slice(&proc_tile);
        }

        BlockTextureAtlas {
            pixels,
            loaded_mojang_pngs,
        }
    }
}

/// Decodes the top `16x16` frame of a PNG image into `RGBA8` without panicking on malformed input.
pub fn decode_png_16x16_rgba(bytes: &[u8]) -> Option<[u8; TILE_BYTES]> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let buf_size = reader.output_buffer_size();
    if buf_size > 4 * 1024 * 1024 {
        return None;
    }
    let mut buf = vec![0u8; buf_size];
    let info = reader.next_frame(&mut buf).ok()?;
    if info.width < TILE_SIZE || info.height < TILE_SIZE {
        return None;
    }

    let mut out = [0u8; TILE_BYTES];
    let stride = info.line_size;
    for y in 0..16usize {
        let row = &buf[(y * stride)..((y + 1) * stride)];
        for x in 0..16usize {
            let o = (y * 16 + x) * 4;
            match info.color_type {
                png::ColorType::Rgba => {
                    let s = x * 4;
                    out[o..o + 4].copy_from_slice(&row[s..s + 4]);
                }
                png::ColorType::Rgb => {
                    let s = x * 3;
                    out[o] = row[s];
                    out[o + 1] = row[s + 1];
                    out[o + 2] = row[s + 2];
                    out[o + 3] = 255;
                }
                png::ColorType::GrayscaleAlpha => {
                    let s = x * 2;
                    out[o] = row[s];
                    out[o + 1] = row[s];
                    out[o + 2] = row[s];
                    out[o + 3] = row[s + 1];
                }
                png::ColorType::Grayscale => {
                    let v = row[x];
                    out[o] = v;
                    out[o + 1] = v;
                    out[o + 2] = v;
                    out[o + 3] = 255;
                }
                png::ColorType::Indexed => return None,
            }
        }
    }
    Some(out)
}

fn apply_biome_tint_if_grayscale(tile: &mut [u8; TILE_BYTES], tint: [u8; 4]) {
    let is_gray = (0..TILE_BYTES).step_by(4).all(|i| {
        tile[i + 3] == 0
            || (tile[i].abs_diff(tile[i + 1]) <= 4 && tile[i + 1].abs_diff(tile[i + 2]) <= 4)
    });
    if !is_gray {
        return;
    }
    for i in (0..TILE_BYTES).step_by(4) {
        tile[i] = ((u16::from(tile[i]) * u16::from(tint[0])) / 255) as u8;
        tile[i + 1] = ((u16::from(tile[i + 1]) * u16::from(tint[1])) / 255) as u8;
        tile[i + 2] = ((u16::from(tile[i + 2]) * u16::from(tint[2])) / 255) as u8;
        tile[i + 3] = ((u16::from(tile[i + 3]) * u16::from(tint[3])) / 255) as u8;
    }
}

#[inline]
fn pixel_hash(seed: u32, x: usize, y: usize) -> i16 {
    let mut h = seed
        .wrapping_add((x as u32).wrapping_mul(374761393))
        .wrapping_add((y as u32).wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    (((h >> 16) & 0x1F) as i16) - 16
}

#[inline]
fn clamp_u8(v: i16) -> u8 {
    v.clamp(0, 255) as u8
}

/// Clean-room procedural `16x16` pixel-art voxel texture generator.
pub fn generate_procedural_tile(layer: u8) -> [u8; TILE_BYTES] {
    let mut out = [0u8; TILE_BYTES];
    for y in 0..16usize {
        for x in 0..16usize {
            let n = pixel_hash(u32::from(layer) * 977 + 17, x, y);
            let rgba: [u8; 4] = match layer {
                LAYER_GRASS_TOP => [
                    clamp_u8(98 + n / 2),
                    clamp_u8(172 + n),
                    clamp_u8(70 + n / 2),
                    255,
                ],
                LAYER_GRASS_SIDE => {
                    let overhang = 3 + ((x * 7 + 3) % 3);
                    if y < overhang {
                        [
                            clamp_u8(98 + n / 2),
                            clamp_u8(172 + n),
                            clamp_u8(70 + n / 2),
                            255,
                        ]
                    } else {
                        [
                            clamp_u8(134 + n),
                            clamp_u8(96 + (n * 2) / 3),
                            clamp_u8(67 + n / 2),
                            255,
                        ]
                    }
                }
                LAYER_DIRT => [
                    clamp_u8(134 + n),
                    clamp_u8(96 + (n * 2) / 3),
                    clamp_u8(67 + n / 2),
                    255,
                ],
                LAYER_STONE => {
                    let strata = if (y + x / 4).is_multiple_of(4) { -8 } else { 0 };
                    let v = clamp_u8(128 + n / 2 + strata);
                    [v, v, clamp_u8(i16::from(v) + 3), 255]
                }
                LAYER_COBBLESTONE => {
                    let mortar = (x + (y / 4) * 3).is_multiple_of(4) || y.is_multiple_of(4);
                    let base = if mortar { 82 } else { 122 + n };
                    let v = clamp_u8(base);
                    [v, v, clamp_u8(i16::from(v) + 2), 255]
                }
                LAYER_OAK_PLANKS => {
                    let seam = y.is_multiple_of(4) || (x == ((y / 4) * 7 + 5) % 16);
                    let d = if seam { -22 } else { n / 2 };
                    [clamp_u8(166 + d), clamp_u8(133 + d), clamp_u8(80 + d), 255]
                }
                LAYER_OAK_LOG_SIDE => {
                    let ridge = if x.is_multiple_of(3) { -14 } else { 6 };
                    [
                        clamp_u8(106 + ridge + n / 2),
                        clamp_u8(82 + ridge + n / 2),
                        clamp_u8(48 + ridge / 2),
                        255,
                    ]
                }
                LAYER_OAK_LOG_TOP => {
                    let border = x <= 1 || x >= 14 || y <= 1 || y >= 14;
                    if border {
                        [102, 79, 46, 255]
                    } else {
                        let ring = ((x.abs_diff(7).max(y.abs_diff(7))) % 2) as i16 * -10;
                        [
                            clamp_u8(176 + ring),
                            clamp_u8(142 + ring),
                            clamp_u8(88 + ring),
                            255,
                        ]
                    }
                }
                LAYER_OAK_LEAVES => {
                    let d = if (x + y * 3).is_multiple_of(5) {
                        -20
                    } else {
                        n
                    };
                    [
                        clamp_u8(68 + d / 2),
                        clamp_u8(148 + d),
                        clamp_u8(50 + d / 2),
                        255,
                    ]
                }
                LAYER_SAND => {
                    let wave = if (y + x / 3).is_multiple_of(5) { -6 } else { 4 };
                    [
                        clamp_u8(222 + n / 2 + wave),
                        clamp_u8(210 + n / 2 + wave),
                        clamp_u8(164 + n / 2 + wave),
                        255,
                    ]
                }
                LAYER_WATER => {
                    let crest = if (y * 2 + x / 2).is_multiple_of(7) {
                        18
                    } else {
                        n / 2
                    };
                    [
                        clamp_u8(48 + crest / 2),
                        clamp_u8(116 + crest),
                        clamp_u8(216 + crest),
                        204,
                    ]
                }
                LAYER_GLASS => {
                    let frame = x == 0 || x == 15 || y == 0 || y == 15;
                    let streak = (x + y == 5) || (x + y == 6 && x > 1 && y > 1);
                    if frame {
                        [215, 240, 250, 220]
                    } else if streak {
                        [240, 252, 255, 180]
                    } else {
                        [190, 225, 240, 48]
                    }
                }
                LAYER_GLOWSTONE => {
                    let crystal = (((x * 5) ^ (y * 3)) & 3) == 0;
                    if crystal {
                        [255, 246, 188, 255]
                    } else {
                        [clamp_u8(218 + n), clamp_u8(172 + n), clamp_u8(84 + n), 255]
                    }
                }
                LAYER_DEEPSLATE => {
                    let v = clamp_u8(68 + n / 2);
                    [v, v, clamp_u8(i16::from(v) + 6), 255]
                }
                LAYER_BEDROCK => {
                    let dark = ((x * 13 + y * 7) & 3) == 0;
                    let v = if dark { 34 } else { clamp_u8(78 + n) };
                    [v, v, v, 255]
                }
                _ => {
                    let border = x == 0 || x == 15 || y == 0 || y == 15;
                    let v = if border { 236 } else { clamp_u8(248 + n / 4) };
                    [v, v, v, 255]
                }
            };
            let o = (y * 16 + x) * 4;
            out[o..o + 4].copy_from_slice(&rgba);
        }
    }
    out
}

/// Maps `(state_id, is_top_face, is_bottom_face)` to `(atlas_layer, vertex_tint_rgba)`.
pub fn block_face_material(
    state_id: u32,
    is_top_face: bool,
    is_bottom_face: bool,
) -> (u8, [u8; 4]) {
    const WHITE: [u8; 4] = [255, 255, 255, 255];
    let Some(block) = generated_blocks::block_by_state_id(state_id) else {
        return (LAYER_TINTED_SOLID, [200, 80, 200, 255]);
    };
    match block.name {
        "minecraft:grass_block" => {
            if is_top_face {
                (LAYER_GRASS_TOP, WHITE)
            } else if is_bottom_face {
                (LAYER_DIRT, WHITE)
            } else {
                (LAYER_GRASS_SIDE, WHITE)
            }
        }
        "minecraft:dirt" | "minecraft:coarse_dirt" => (LAYER_DIRT, WHITE),
        "minecraft:stone" | "minecraft:andesite" => (LAYER_STONE, WHITE),
        "minecraft:cobblestone" => (LAYER_COBBLESTONE, WHITE),
        "minecraft:oak_planks" => (LAYER_OAK_PLANKS, WHITE),
        "minecraft:oak_log" => {
            if is_top_face || is_bottom_face {
                (LAYER_OAK_LOG_TOP, WHITE)
            } else {
                (LAYER_OAK_LOG_SIDE, WHITE)
            }
        }
        "minecraft:oak_leaves" => (LAYER_OAK_LEAVES, WHITE),
        "minecraft:sand" => (LAYER_SAND, WHITE),
        "minecraft:water" => (LAYER_WATER, WHITE),
        "minecraft:glass" => (LAYER_GLASS, WHITE),
        "minecraft:glowstone" => (LAYER_GLOWSTONE, WHITE),
        "minecraft:deepslate" => (LAYER_DEEPSLATE, WHITE),
        "minecraft:bedrock" => (LAYER_BEDROCK, WHITE),
        _ => (
            LAYER_TINTED_SOLID,
            procedural_block_color(state_id, is_top_face),
        ),
    }
}

/// Clean-room procedural RGBA color (`[r, g, b, a]`) used for HUD icons, minimaps, and fallback blocks.
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
        "minecraft:oak_leaves" => [76, 153, 56, 255],
        "minecraft:oak_planks" => [162, 130, 78, 255],
        "minecraft:glass" => [210, 235, 245, 140],
        "minecraft:glowstone" => [248, 212, 120, 255],
        other => {
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

    #[test]
    fn procedural_atlas_and_png_decoder_produce_exact_16x16_rgba_tiles() {
        let atlas = BlockTextureAtlas::default();
        assert_eq!(atlas.pixels.len(), TILE_BYTES * ATLAS_LAYER_COUNT as usize);
        assert_eq!(atlas.loaded_mojang_pngs, 0);

        // Encode a synthetic 16x16 RGBA PNG in memory and verify lossless round-trip decoding
        let mut raw_png = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut raw_png, 16, 16);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().expect("png header");
            let mut img = [0u8; TILE_BYTES];
            img[0..4].copy_from_slice(&[12, 34, 56, 255]);
            writer.write_image_data(&img).expect("png data");
        }
        let decoded = decode_png_16x16_rgba(&raw_png).expect("decode synthetic 16x16 png");
        assert_eq!(&decoded[0..4], &[12, 34, 56, 255]);
    }
}
