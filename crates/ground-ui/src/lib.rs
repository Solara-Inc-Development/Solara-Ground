//! Clean-room 2D UI layout, 5x7 bitmap pixel font, Survival Hearts/Hunger bars,
//! Inventory & Quick-Crafting overlay, Interactive Chat Bar, and Esc Pause menu for Solara-Ground.

pub mod font;
pub mod hud_layout;

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct UiVertex {
    pub pos_ndc: [f32; 2],
    pub color: [u8; 4],
}

pub use hud_layout::{HudModel, QUICK_CRAFT_RECIPES, UiSlotDisplay, build_full_ui_vertices};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_font_inventory_chat_and_survival_hud_emit_valid_triangles() {
        let model = HudModel {
            selected_slot: 2,
            health: 15.0,
            food: 18,
            inventory_open: true,
            chat_open: true,
            chat_input: "/craft wooden_sword".into(),
            tab_hint: "wooden_sword".into(),
            fps: 280,
            frame_ms: 3.5,
            chunk_count: 81,
            entity_count: 5,
            chat_lines: vec!["Welcome to Solara-Sky!".into()],
            ..Default::default()
        };
        let verts = build_full_ui_vertices(&model);
        assert!(verts.len() > 1200);
        assert_eq!(verts.len() % 6, 0);
    }
}
