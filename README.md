# Solara-Ground

An open-source **Minecraft: Java Edition** client written in native Rust (`wgpu` + `winit`), focused on maximum FPS, low input latency, compact memory usage, and zero garbage-collection stutters.

> **Status:** Early active development (Phase 0 / Phase 1 protocol, voxel mesher, and renderer foundation). Not yet a playable client.
> **Target Minecraft Version:** Java Edition **26.3** (*Wilderness Bound*, protocol **`777`**)
> **Supported Platforms:** Windows & Linux (Vulkan / DirectX 12)

## Goals

- **100% Standard Java Server Compatibility:** Connects to any standard Java server — Vanilla, Paper, Purpur, Velocity/BungeeCord proxies, or `Solara-Sky`.
- **Online-Mode Security:** Supports official Mojang/Microsoft online-mode sessions with RSA + AES-128-CFB8 stream encryption (designed for integration with **Solara Launcher**).
- **Voxel-First `wgpu` Renderer:** Background multithreaded chunk meshing (`rayon`), face culling, tightly packed 8-byte vertices, and camera frustum culling.
- **Fast Version Updates:** Built-in `tools/mc-codegen` imports official Mojang data reports (`--reports`) for packet IDs, block states, and registries.

## Workspace Layout

- `crates/ground-protocol` — MC `26.3` (`777`) protocol primitives, online-mode session join & AES-128-CFB8 encryption, and async Server List Ping client
- `crates/ground-world` — Bit-packed paletted chunks, generated MC `26.3` block metadata, and compact 8-byte face-culling voxel chunk mesher
- `crates/ground-render` — `wgpu` + `winit` renderer configuration, camera projection, and 6-plane frustum culling
- `crates/ground-client` — `solara-ground` client binary (`SessionConfig` with redacted token handling + live `--ping <host:port>` mode)
- `tools/mc-codegen` — Code generator from Mojang official `--reports` JSON

## Building & Testing

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run --bin solara-ground -- --ping 127.0.0.1:25565
```

## License

Licensed under the **GNU General Public License v3.0** — see [LICENSE](./LICENSE).
