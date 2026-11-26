use crate::UiVertex;
use crate::font::{push_rect_px, push_text_px};

#[derive(Debug, Clone)]
pub struct HudModel {
    pub width: u32,
    pub height: u32,
    pub selected_slot: usize,
    pub hotbar_top_colors: [[u8; 4]; 9],
    pub hotbar_side_colors: [[u8; 4]; 9],
    pub health: f32,
    pub food: i32,
    pub survival_physics: bool,
    pub paused: bool,
    pub fps: u32,
    pub frame_ms: f32,
    pub camera_pos: [f32; 3],
    pub chunk_count: usize,
    pub entity_count: usize,
    pub chat_lines: Vec<String>,
}

/// Builds all 2D UI triangles (Telemetry + Crosshair + 10 Hearts + 10 Hunger + 9-Slot Hotbar + Chat + Pause Menu).
pub fn build_full_ui_vertices(m: &HudModel) -> Vec<UiVertex> {
    let sw = m.width.max(1) as f32;
    let sh = m.height.max(1) as f32;
    let mut out = Vec::with_capacity(2048);

    // 1. Top-Left Crisp F3-Lite Telemetry Overlay
    push_rect_px(
        &mut out,
        sw,
        sh,
        [10.0, 10.0, 440.0, 44.0],
        [14, 16, 22, 168],
    );
    push_text_px(
        &mut out,
        (sw, sh),
        (18.0, 16.0),
        2.0,
        &format!("SOLARA-GROUND 26.3 | {} FPS ({:.1} MS)", m.fps, m.frame_ms),
        [110, 235, 165, 255],
    );
    push_text_px(
        &mut out,
        (sw, sh),
        (18.0, 34.0),
        2.0,
        &format!(
            "XYZ {:.0} {:.0} {:.0} | CHUNKS {} | ENT {} | {}",
            m.camera_pos[0],
            m.camera_pos[1],
            m.camera_pos[2],
            m.chunk_count,
            m.entity_count,
            if m.survival_physics {
                "SURVIVAL"
            } else {
                "FLY"
            }
        ),
        [228, 235, 248, 255],
    );

    // 2. Center Crosshair (when not paused)
    let cx = (sw * 0.5).floor();
    let cy = (sh * 0.5).floor();
    if !m.paused {
        push_rect_px(
            &mut out,
            sw,
            sh,
            [cx - 9.0, cy - 2.0, 18.0, 4.0],
            [20, 20, 24, 180],
        );
        push_rect_px(
            &mut out,
            sw,
            sh,
            [cx - 2.0, cy - 9.0, 4.0, 18.0],
            [20, 20, 24, 180],
        );
        push_rect_px(
            &mut out,
            sw,
            sh,
            [cx - 8.0, cy - 1.0, 16.0, 2.0],
            [245, 248, 255, 235],
        );
        push_rect_px(
            &mut out,
            sw,
            sh,
            [cx - 1.0, cy - 8.0, 2.0, 16.0],
            [245, 248, 255, 235],
        );
    }

    // 3. Bottom 9-Slot Hotbar + 10 Survival Hearts + 10 Hunger Icons
    let slot_px = 44.0f32;
    let gap_px = 4.0f32;
    let total_w = 9.0 * slot_px + 8.0 * gap_px + 12.0;
    let bar_x = ((sw - total_w) * 0.5).floor();
    let bar_y = (sh - slot_px - 22.0).max(64.0);

    // 10 Hearts row (left side above hotbar, 20 HP total)
    let hp_half_units = m.health.clamp(0.0, 20.0).ceil() as i32;
    let hearts_y = bar_y - 22.0;
    for h in 0..10i32 {
        let hx = bar_x + (h as f32) * 17.0;
        push_rect_px(
            &mut out,
            sw,
            sh,
            [hx, hearts_y, 14.0, 13.0],
            [28, 18, 22, 220],
        );
        let full_thresh = (h + 1) * 2;
        if hp_half_units >= full_thresh {
            push_rect_px(
                &mut out,
                sw,
                sh,
                [hx + 2.0, hearts_y + 2.0, 10.0, 9.0],
                [232, 46, 62, 255],
            );
            push_rect_px(
                &mut out,
                sw,
                sh,
                [hx + 3.0, hearts_y + 3.0, 3.0, 3.0],
                [255, 160, 170, 255],
            );
        } else if hp_half_units == full_thresh - 1 {
            push_rect_px(
                &mut out,
                sw,
                sh,
                [hx + 2.0, hearts_y + 2.0, 5.0, 9.0],
                [232, 46, 62, 255],
            );
        }
    }

    // 10 Hunger icons row (right side above hotbar, 20 Food total)
    let food_units = m.food.clamp(0, 20);
    for f in 0..10i32 {
        let fx = (bar_x + total_w - 14.0) - (f as f32) * 17.0;
        push_rect_px(
            &mut out,
            sw,
            sh,
            [fx, hearts_y, 14.0, 13.0],
            [28, 22, 16, 220],
        );
        let full_thresh = (f + 1) * 2;
        if food_units >= full_thresh {
            push_rect_px(
                &mut out,
                sw,
                sh,
                [fx + 2.0, hearts_y + 2.0, 10.0, 9.0],
                [212, 136, 56, 255],
            );
        } else if food_units == full_thresh - 1 {
            push_rect_px(
                &mut out,
                sw,
                sh,
                [fx + 2.0, hearts_y + 2.0, 5.0, 9.0],
                [212, 136, 56, 255],
            );
        }
    }

    // Hotbar backdrop & 9 slots
    push_rect_px(
        &mut out,
        sw,
        sh,
        [bar_x, bar_y - 6.0, total_w, slot_px + 12.0],
        [18, 20, 26, 196],
    );

    let sel = m.selected_slot.min(8);
    for i in 0..9usize {
        let sx = bar_x + 6.0 + (i as f32) * (slot_px + gap_px);
        let sy = bar_y;

        if i == sel {
            push_rect_px(
                &mut out,
                sw,
                sh,
                [sx - 3.0, sy - 3.0, slot_px + 6.0, slot_px + 6.0],
                [248, 250, 255, 250],
            );
        }
        push_rect_px(
            &mut out,
            sw,
            sh,
            [sx, sy, slot_px, slot_px],
            [36, 40, 48, 230],
        );

        let top_c = m.hotbar_top_colors[i];
        let side_c = m.hotbar_side_colors[i];
        let shaded_side = [
            ((u16::from(side_c[0]) * 4) / 5) as u8,
            ((u16::from(side_c[1]) * 4) / 5) as u8,
            ((u16::from(side_c[2]) * 4) / 5) as u8,
            255,
        ];
        push_rect_px(
            &mut out,
            sw,
            sh,
            [sx + 8.0, sy + 8.0, slot_px - 16.0, 10.0],
            [top_c[0], top_c[1], top_c[2], 255],
        );
        push_rect_px(
            &mut out,
            sw,
            sh,
            [sx + 8.0, sy + 18.0, slot_px - 16.0, slot_px - 26.0],
            shaded_side,
        );
    }

    // 4. Recent Chat Feed (bottom-left above hotbar)
    if !m.chat_lines.is_empty() {
        let count = m.chat_lines.len().min(5);
        let box_h = (count as f32) * 18.0 + 10.0;
        let box_y = (bar_y - 36.0 - box_h).max(64.0);
        push_rect_px(
            &mut out,
            sw,
            sh,
            [12.0, box_y, 460.0, box_h],
            [12, 14, 20, 150],
        );
        for (idx, line) in m.chat_lines.iter().rev().take(5).rev().enumerate() {
            let clipped: String = line.chars().take(58).collect();
            push_text_px(
                &mut out,
                (sw, sh),
                (20.0, box_y + 6.0 + (idx as f32) * 18.0),
                1.8,
                &clipped,
                [240, 244, 255, 245],
            );
        }
    }

    // 5. Esc Pause & Settings Overlay Menu
    if m.paused {
        push_rect_px(&mut out, sw, sh, [0.0, 0.0, sw, sh], [8, 10, 16, 168]);
        let mw = 380.0f32;
        let mh = 230.0f32;
        let mx = cx - mw * 0.5;
        let my = cy - mh * 0.5;
        push_rect_px(
            &mut out,
            sw,
            sh,
            [mx - 3.0, my - 3.0, mw + 6.0, mh + 6.0],
            [96, 196, 142, 240],
        );
        push_rect_px(&mut out, sw, sh, [mx, my, mw, mh], [22, 26, 34, 245]);

        push_text_px(
            &mut out,
            (sw, sh),
            (mx + 96.0, my + 20.0),
            2.6,
            "SOLARA PAUSED",
            [110, 235, 165, 255],
        );

        let items = [
            "[ESC] RESUME GAME",
            if m.survival_physics {
                "[F]   MODE: SURVIVAL GRAVITY"
            } else {
                "[F]   MODE: CREATIVE FLY"
            },
            "[R]   RESPAWN PLAYER",
            "[Q]   DISCONNECT AND EXIT",
        ];
        for (idx, label) in items.iter().enumerate() {
            let iy = my + 64.0 + (idx as f32) * 36.0;
            push_rect_px(
                &mut out,
                sw,
                sh,
                [mx + 28.0, iy, mw - 56.0, 28.0],
                [38, 44, 58, 240],
            );
            push_text_px(
                &mut out,
                (sw, sh),
                (mx + 42.0, iy + 7.0),
                1.8,
                label,
                [238, 242, 252, 255],
            );
        }
    }

    out
}
