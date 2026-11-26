//! Clean-room 2D UI layout, 5x7 bitmap pixel font, Survival Hearts/Hunger bars,
//! live Chat feed, and Esc Pause/Settings overlay for Solara-Ground.

pub mod font;
pub mod hud_layout;

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct UiVertex {
    pub pos_ndc: [f32; 2],
    pub color: [u8; 4],
}

pub use hud_layout::{HudModel, build_full_ui_vertices};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_font_and_full_survival_hud_emit_valid_triangles() {
        let model = HudModel {
            width: 1280,
            height: 720,
            selected_slot: 2,
            hotbar_top_colors: [[128, 128, 128, 255]; 9],
            hotbar_side_colors: [[100, 100, 100, 255]; 9],
            health: 15.0,
            food: 18,
            survival_physics: true,
            paused: true,
            fps: 280,
            frame_ms: 3.5,
            camera_pos: [8.0, -56.0, 8.0],
            chunk_count: 81,
            entity_count: 5,
            chat_lines: vec!["Welcome to Solara-Sky!".into()],
        };
        let verts = build_full_ui_vertices(&model);
        assert!(verts.len() > 600);
        assert_eq!(verts.len() % 6, 0);
    }
}
