struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> u_camera: CameraUniform;

struct EntityVertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct EntityVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_entity(in: EntityVertexInput) -> EntityVertexOutput {
    var out: EntityVertexOutput;
    out.clip_pos = u_camera.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_entity(in: EntityVertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
