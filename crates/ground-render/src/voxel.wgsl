struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct SectionUniform {
    origin: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u_camera: CameraUniform;

@group(0) @binding(1)
var t_blocks: texture_2d_array<f32>;

@group(0) @binding(2)
var s_blocks: sampler;

@group(1) @binding(0)
var<uniform> u_section: SectionUniform;

struct VertexInput {
    @location(0) packed: vec2<u32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) tex_layer: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let w0 = in.packed.x;
    let w1 = in.packed.y;

    let lx = f32(w0 & 0x1Fu);
    let ly = f32((w0 >> 5u) & 0x1Fu);
    let lz = f32((w0 >> 10u) & 0x1Fu);
    let face = (w0 >> 15u) & 0x07u;
    let ao_raw = f32((w0 >> 18u) & 0x03u);
    let blk_light = f32((w0 >> 20u) & 0x0Fu) / 15.0;
    let sky_light = f32((w0 >> 24u) & 0x0Fu) / 15.0;
    let tex_layer = (w0 >> 28u) & 0x0Fu;

    let r = f32(w1 & 0xFFu) / 255.0;
    let g = f32((w1 >> 8u) & 0xFFu) / 255.0;
    let b = f32((w1 >> 16u) & 0xFFu) / 255.0;
    let a = f32((w1 >> 24u) & 0xFFu) / 255.0;

    // Classic crisp directional Minecraft face shading
    var shade = 1.0;
    if (face == 2u) {
        shade = 0.52; // -Y bottom
    } else if (face == 0u || face == 1u) {
        shade = 0.82; // +/-X east/west
    } else if (face == 4u || face == 5u) {
        shade = 0.70; // +/-Z north/south
    }
    let ao = 0.62 + 0.1266 * ao_raw;
    let voxel_light = clamp(max(sky_light, blk_light * 0.96), 0.14, 1.0);

    // World/section-space tiling UVs so greedy-merged (W x H) quads repeat 16x16 block tiles seamlessly!
    var tile_uv = vec2<f32>(lx, lz);
    if (face == 0u || face == 1u) {
        tile_uv = vec2<f32>(lz, 16.0 - ly);
    } else if (face == 4u || face == 5u) {
        tile_uv = vec2<f32>(lx, 16.0 - ly);
    }

    let world_pos = u_section.origin.xyz + vec3<f32>(lx, ly, lz);
    var out: VertexOutput;
    out.clip_pos = u_camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = vec4<f32>(vec3<f32>(r, g, b) * shade * ao * voxel_light, a);
    out.uv = tile_uv;
    out.tex_layer = tex_layer;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(t_blocks, s_blocks, in.uv, i32(in.tex_layer));
    let out_alpha = texel.a * in.color.a;
    if (out_alpha < 0.05) {
        discard;
    }
    return vec4<f32>(texel.rgb * in.color.rgb, out_alpha);
}
