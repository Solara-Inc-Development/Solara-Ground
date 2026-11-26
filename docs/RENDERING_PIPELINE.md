# Solara-Ground GPU Voxel Rendering Pipeline

1. **8-Byte `PackedVoxelVertex`:** Encodes section-local `(x, y, z)` (`0..=16`), face normal, corner AO, 4-bit Sky & Block light, `16x16` 2D texture array layer, and RGBA8 tint in just **8 bytes per vertex**.
2. **Multi-Core 2D Greedy Meshing (`rayon`):** Coplanar adjacent faces sharing identical material, AO, and lighting merge into single `(W x H)` quads (`576 -> 6` quads on flat sections), repeating `16x16` block tiles via hardware `AddressMode::Repeat`.
3. **Single-VBO Mega-Buffer + `multi_draw_indexed_indirect`:** All visible world sections reside in a single contiguous GPU Vertex Buffer, single Index Buffer, and single read-only Storage Buffer of section origins, rendered in **1 indirect GPU draw call** after 6-plane Gribb-Hartmann frustum culling.

> NOT AN OFFICIAL MINECRAFT PRODUCT. NOT APPROVED BY OR ASSOCIATED WITH MOJANG OR MICROSOFT.
