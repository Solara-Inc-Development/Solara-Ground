# Solara-Ground

An open-source **Minecraft: Java Edition** client written in native Rust (`wgpu` + `winit`), focused on maximum FPS, low input latency, compact memory usage, and zero garbage-collection stutters.

> **Status:** Active development (Phases 0–7 playable survival client complete: MC `26.3` online-mode protocol, `rayon` parallel chunk decoding + 2D greedy voxel mesher, single-VBO GPU Mega-Buffer + `multi_draw_indexed_indirect`, `16x16` 2D texture array, dynamic Day/Sunset/Night sky & fog shader, 3D players/mobs/item drops, `E` Inventory & Quick-Crafting GUI, `T`/`/` live Chat & Command bar, 3D first-person hand, and WAV/OS spatial audio).
> **Target Minecraft Version:** Java Edition **26.3** (*Wilderness Bound*, protocol **`777`**)
> **Supported Platforms:** Windows & Linux (Vulkan / DirectX 12)

## Goals

- **100% Standard Java Server Compatibility:** Connects to any standard Java server — Vanilla, Paper, Purpur, Velocity/BungeeCord proxies, or `Solara-Sky`.
- **Online-Mode Security:** Supports official Mojang/Microsoft online-mode sessions with RSA + AES-128-CFB8 stream encryption (designed for integration with **Solara Launcher**).
- **Voxel-First `wgpu` Renderer:** Background multithreaded chunk meshing (`rayon`), face culling, tightly packed 8-byte vertices, and camera frustum culling.
- **Fast Version Updates:** Built-in `tools/mc-codegen` imports official Mojang data reports (`--reports`) for packet IDs, block states, and registries.

## Workspace Layout

- `crates/ground-data` — Generated MC `26.3` (`777`) items (`1,658`), entity types (`161`) + hitboxes, sound events (`1,991`), particles (`128`), and mob effects (`40`)
- `crates/ground-protocol` — MC `26.3` (`777`) protocol primitives, online-mode session join & AES-128-CFB8 encryption, and Play/Survival client codecs
- `crates/ground-world` — Bit-packed paletted chunks, `SectionLight`, `rayon` multi-core 2D greedy voxel mesher (`8`-byte vertices), AABB physics, DDA raycaster & `LocalMojangAssets`
- `crates/ground-ui` — Clean-room `5x7` bitmap pixel font, F3-lite telemetry, 10 Survival Hearts, 10 Hunger icons, 9-slot hotbar, chat feed & `Esc` Pause/Settings menu
- `crates/ground-audio` — Clean-room procedural 16-bit PCM SFX synthesizer & 3D spatial sound attenuation
- `crates/ground-render` — `wgpu` + `winit` single-VBO Mega-Buffer + `multi_draw_indexed_indirect` renderer, `16x16` 2D texture array, 3D players & mobs, block selection outline & FPS mouse-look
- `crates/ground-client` — `solara-ground` client binary (`SessionConfig` with redacted token handling + live multiplayer bridge)
- `docs/` & `packaging/` — Architecture, GPU pipeline, EULA compliance docs & desktop entry
- `tools/mc-codegen` — Code generator from Mojang official `--reports` JSON

## Building & Testing

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run --bin solara-ground -- --ping 127.0.0.1:25565
```

## Assets & Mojang EULA Compliance

`Solara-Ground` is a clean-room native Rust client and **does not bundle, host, or redistribute any Mojang code or game assets**. Official Minecraft assets (textures, sounds, models) are downloaded directly from Mojang's official servers onto the player's machine by **Solara Launcher** after authenticating a legitimate paid Microsoft/Minecraft account, while only Solara's own Rust client files are delivered from Solara's servers / GitHub Releases.

## License & Legal Notice

Licensed under the **GNU General Public License v3.0** — see [LICENSE](./LICENSE).

> **NOT AN OFFICIAL MINECRAFT PRODUCT. NOT APPROVED BY OR ASSOCIATED WITH MOJANG OR MICROSOFT.**
