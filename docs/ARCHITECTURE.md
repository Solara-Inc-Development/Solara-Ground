# Solara-Ground Architecture Overview

**Solara-Ground** is a clean-room native Rust client (`wgpu` + `winit`, Vulkan / DirectX 12) for **Minecraft: Java Edition 26.3** (protocol **`777`**).

## Workspace Crate Map

| Crate | Responsibility |
| :--- | :--- |
| `crates/ground-data` | Generated protocol registry tables (`1,658` items, `161` entity types + hitboxes, `1,991` sound events, `128` particles, `40` mob effects) |
| `crates/ground-protocol` | Online-mode RSA + Mojang session join, AES-128-CFB8 encryption, zlib framing, Configuration sync, and Play/Survival packet codecs |
| `crates/ground-world` | Bit-packed paletted chunks, `SectionLight` compression, `rayon` multi-core chunk decoder, 2D Greedy Mesher (`8`-byte vertices), AABB physics, DDA raycaster, and `LocalMojangAssets` |
| `crates/ground-ui` | Clean-room `5x7` bitmap pixel font, F3-lite telemetry overlay, 10 Survival Hearts, 10 Hunger icons, 9-slot hotbar, chat feed, and `Esc` Pause/Settings menu |
| `crates/ground-audio` | Clean-room procedural 16-bit PCM SFX synthesizer, 3D spatial attenuation, and local Mojang `.ogg` asset key mapping |
| `crates/ground-render` | Single-VBO GPU Mega-Buffer + `multi_draw_indexed_indirect` renderer, `16x16` 2D texture array, 3D players & mobs, target block selection outline, and FPS mouse-look |
| `crates/ground-client` | Client binary, `SessionConfig` (Solara Launcher hook with redacted tokens), and async network ↔ render bridge |
| `tools/mc-codegen` | Data-report code generator importing official `--reports` JSON |

> NOT AN OFFICIAL MINECRAFT PRODUCT. NOT APPROVED BY OR ASSOCIATED WITH MOJANG OR MICROSOFT.
