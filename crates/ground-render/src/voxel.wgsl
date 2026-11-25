struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct SectionUniform {
    origin: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u_camera: CameraUniform;

@group(1) @binding(0)
var<uniform> u_section: SectionUniform;

struct VertexInput {
    @location(0) packed: vec2<u32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
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
    let corner = (w0 >> 28u) & 0x03u;

    let r = f32(w1 & 0xFFu) / 255.0;
    let g = f32((w1 >> 8u) & 0xFFu) / 255.0;
    let b = f32((w1 >> 16u) & 0xFFu) / 255.0;
    let a = f32((w1 >> 24u) & 0xFFu) / 255.0;

    // Directional face sunlight matching classic Minecraft crisp voxel contrast
    var shade = 1.0;
    if (face == 2u) {
        shade = 0.52; // -Y bottom
    } else if (face == 0u || face == 1u) {
        shade = 0.82; // +/-X east/west
    } else if (face == 4u || face == 5u) {
        shade = 0.70; // +/-Z north/south
    }
    let ao = 0.55 + 0.15 * ao_raw;

    var corner_uv = vec2<f32>(0.0, 0.0);
    if (corner == 1u) {
        corner_uv = vec2<f32>(1.0, 0.0);
    } else if (corner == 2u) {
        corner_uv = vec2<f32>(1.0, 1.0);
    } else if (corner == 3u) {
        corner_uv = vec2<f32>(0.0, 1.0);
    }

    let world_pos = u_section.origin.xyz + vec3<f32>(lx, ly, lz);
    var out: VertexOutput;
    out.clip_pos = u_camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = vec4<f32>(vec3<f32>(r, g, b) * shade * ao, a);
    out.uv = corner_uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Subtle crisp block border accent so adjacent identical blocks remain readable pre-atlas
    let edge_dist = min(min(in.uv.x, 1.0 - in.uv.x), min(in.uv.y, 1.0 - in.uv.y));
    let border = smoothstep(0.0, 0.04, edge_dist);
    let rgb = in.color.rgb * (0.93 + 0.07 * border);
    return vec4<f32>(rgb, in.color.a);
}
