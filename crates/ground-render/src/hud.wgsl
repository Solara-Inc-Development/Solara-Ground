struct HudVertexInput {
    @location(0) pos_ndc: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct HudVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_hud(in: HudVertexInput) -> HudVertexOutput {
    var out: HudVertexOutput;
    out.clip_pos = vec4<f32>(in.pos_ndc, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_hud(in: HudVertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
