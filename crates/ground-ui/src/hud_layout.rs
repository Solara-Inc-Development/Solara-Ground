use crate::UiVertex;
use crate::font::{push_rect_px, push_text_px};

/// Quick-craftable recipes displayed inside the `E` Inventory & Crafting screen (`[1]..[9]`).
pub static QUICK_CRAFT_RECIPES: &[(&str, &str)] = &[
    ("oak_planks", "[1] OAK PLANKS X4   (1 LOG)"),
    ("stick", "[2] STICK X4        (2 PLANKS)"),
    ("crafting_table", "[3] CRAFTING TABLE  (4 PLANKS)"),
    ("wooden_pickaxe", "[4] WOOD PICKAXE    (3 PLK + 2 STK)"),
    ("wooden_sword", "[5] WOOD SWORD      (2 PLK + 1 STK)"),
    ("stone_pickaxe", "[6] STONE PICKAXE   (3 COB + 2 STK)"),
    ("stone_sword", "[7] STONE SWORD     (2 COB + 1 STK)"),
    ("torch", "[8] TORCH X4        (1 COAL + 1 STK)"),
    ("cooked_beef", "[9] COOKED BEEF     (1 BEEF + 1 COAL)"),
];

#[derive(Debug, Clone, Copy, Default)]
pub struct UiSlotDisplay {
    pub count: u8,
    pub label: &'static str,
    pub top_color: [u8; 4],
    pub side_color: [u8; 4],
}

#[derive(Debug, Clone)]
pub struct HudModel {
    pub width: u32,
    pub height: u32,
    pub selected_slot: usize,
    pub hotbar_top_colors: [[u8; 4]; 9],
    pub hotbar_side_colors: [[u8; 4]; 9],
    pub inventory_slots: [UiSlotDisplay; 36],
    pub health: f32,
    pub food: i32,
    pub armor_points: u8,
    pub xp_level: i32,
    pub xp_progress: f32,
    pub day_time_ticks: i64,
    pub hand_swing: f32,
    pub survival_physics: bool,
    pub paused: bool,
    pub inventory_open: bool,
    pub chat_open: bool,
    pub chat_input: String,
    pub tab_hint: String,
    pub fps: u32,
    pub frame_ms: f32,
    pub camera_pos: [f32; 3],
    pub chunk_count: usize,
    pub entity_count: usize,
    pub effect_badges: Vec<(String, [u8; 4])>,
    pub chat_lines: Vec<String>,
}

impl Default for HudModel {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            selected_slot: 0,
            hotbar_top_colors: [[128, 128, 128, 255]; 9],
            hotbar_side_colors: [[100, 100, 100, 255]; 9],
            inventory_slots: [UiSlotDisplay::default(); 36],
            health: 20.0,
            food: 20,
            armor_points: 0,
            xp_level: 0,
            xp_progress: 0.0,
            day_time_ticks: 1000,
            hand_swing: 0.0,
            survival_physics: true,
            paused: false,
            inventory_open: false,
            chat_open: false,
            chat_input: String::new(),
            tab_hint: String::new(),
            fps: 0,
            frame_ms: 0.0,
            camera_pos: [8.0, -56.0, 8.0],
            chunk_count: 0,
            entity_count: 0,
            effect_badges: Vec::new(),
            chat_lines: Vec::new(),
        }
    }
}

fn day_phase_label(day_ticks: i64) -> &'static str {
    let t = day_ticks.rem_euclid(24_000);
    if t < 4_500 {
        "MORNING"
    } else if t < 11_500 {
        "NOON"
    } else if t < 13_800 {
        "SUNSET"
    } else if t < 22_200 {
        "NIGHT"
    } else {
        "DAWN"
    }
}

/// Builds all 2D UI triangles (Telemetry + Crosshair + Hearts + Hunger + Hotbar w/ Stack Counts +
/// Held Item Banner + Chat Feed + Interactive `T` Chat Bar + `E` Inventory/Crafting + Death/Pause Menus).
pub fn build_full_ui_vertices(m: &HudModel) -> Vec<UiVertex> {
    let sw = m.width.max(1) as f32;
    let sh = m.height.max(1) as f32;
    let mut out = Vec::with_capacity(4096);

    // 1. Top-Left Crisp F3-Lite Telemetry Overlay + Day/Night Clock
    push_rect_px(
        &mut out,
        sw,
        sh,
        [10.0, 10.0, 520.0, 60.0],
        [14, 16, 22, 172],
    );
    push_text_px(
        &mut out,
        (sw, sh),
        (18.0, 16.0),
        2.0,
        &format!(
            "SOLARA 26.3 | {} FPS ({:.1} MS) | {}",
            m.fps,
            m.frame_ms,
            day_phase_label(m.day_time_ticks)
        ),
        [110, 235, 165, 255],
    );
    push_text_px(
        &mut out,
        (sw, sh),
        (18.0, 34.0),
        1.8,
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
    push_text_px(
        &mut out,
        (sw, sh),
        (18.0, 52.0),
        1.5,
        "[E] INV/CRAFT  [T] CHAT  [/] CMD  [F] WALK/FLY  [SCROLL/1-9] ITEM",
        [176, 192, 218, 235],
    );

    // 1b. Top-Right Active Status Potion Effect Badges
    for (idx, (label, accent)) in m.effect_badges.iter().take(6).enumerate() {
        let bx = (sw - 176.0).max(540.0);
        let by = 12.0 + (idx as f32) * 26.0;
        push_rect_px(&mut out, sw, sh, [bx, by, 164.0, 22.0], [16, 20, 28, 205]);
        push_rect_px(&mut out, sw, sh, [bx, by, 5.0, 22.0], *accent);
        push_text_px(
            &mut out,
            (sw, sh),
            (bx + 12.0, by + 5.0),
            1.6,
            label,
            [242, 246, 255, 250],
        );
    }

    // 2. Center Crosshair (when playing actively)
    let cx = (sw * 0.5).floor();
    let cy = (sh * 0.5).floor();
    let modal_open = m.paused || m.inventory_open || m.chat_open || m.health <= 0.0;
    if !modal_open {
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

        // First-Person Right-Hand / Held Block-Item View (never clips into walls!)
        let swing_arc = (m.hand_swing * std::f32::consts::PI).sin();
        let hx = sw - 185.0 - swing_arc * 52.0;
        let hy = sh - 148.0 + swing_arc * 34.0;
        let sel_disp = m.inventory_slots[m.selected_slot.min(8)];
        if sel_disp.count > 0 {
            push_rect_px(
                &mut out,
                sw,
                sh,
                [hx, hy, 108.0, 28.0],
                [
                    sel_disp.top_color[0],
                    sel_disp.top_color[1],
                    sel_disp.top_color[2],
                    255,
                ],
            );
            push_rect_px(
                &mut out,
                sw,
                sh,
                [hx, hy + 28.0, 54.0, 96.0],
                [
                    sel_disp.side_color[0],
                    sel_disp.side_color[1],
                    sel_disp.side_color[2],
                    255,
                ],
            );
            let dark = [
                ((u16::from(sel_disp.side_color[0]) * 3) / 4) as u8,
                ((u16::from(sel_disp.side_color[1]) * 3) / 4) as u8,
                ((u16::from(sel_disp.side_color[2]) * 3) / 4) as u8,
                255,
            ];
            push_rect_px(&mut out, sw, sh, [hx + 54.0, hy + 28.0, 54.0, 96.0], dark);
        } else {
            // Bare right arm
            push_rect_px(
                &mut out,
                sw,
                sh,
                [hx + 24.0, hy + 10.0, 68.0, 22.0],
                [228, 182, 148, 255],
            );
            push_rect_px(
                &mut out,
                sw,
                sh,
                [hx + 24.0, hy + 32.0, 68.0, 118.0],
                [198, 152, 122, 255],
            );
        }
    }

    // 3. Bottom 9-Slot Hotbar + 10 Survival Hearts + 10 Hunger Icons + Stack Counts
    let slot_px = 44.0f32;
    let gap_px = 4.0f32;
    let total_w = 9.0 * slot_px + 8.0 * gap_px + 12.0;
    let bar_x = ((sw - total_w) * 0.5).floor();
    let bar_y = (sh - slot_px - 22.0).max(64.0);

    let hp_half_units = m.health.clamp(0.0, 20.0).ceil() as i32;
    let hearts_y = bar_y - 28.0;

    // 10 Silver Armor Chestplate Icons (when wearing armor!)
    if m.armor_points > 0 {
        let armor_y = hearts_y - 16.0;
        let ap = i32::from(m.armor_points.min(20));
        for a in 0..10i32 {
            let ax = bar_x + (a as f32) * 17.0;
            push_rect_px(
                &mut out,
                sw,
                sh,
                [ax, armor_y, 14.0, 13.0],
                [22, 26, 34, 220],
            );
            let thresh = (a + 1) * 2;
            if ap >= thresh {
                push_rect_px(
                    &mut out,
                    sw,
                    sh,
                    [ax + 2.0, armor_y + 2.0, 10.0, 9.0],
                    [208, 220, 236, 255],
                );
            } else if ap == thresh - 1 {
                push_rect_px(
                    &mut out,
                    sw,
                    sh,
                    [ax + 2.0, armor_y + 2.0, 5.0, 9.0],
                    [208, 220, 236, 255],
                );
            }
        }
    }

    // Green Experience Progress Bar + XP Level Counter right above Hotbar
    let xp_bar_y = bar_y - 12.0;
    push_rect_px(
        &mut out,
        sw,
        sh,
        [bar_x, xp_bar_y, total_w, 5.0],
        [20, 32, 22, 215],
    );
    let fill_w = (total_w * m.xp_progress.clamp(0.0, 1.0)).floor();
    if fill_w > 0.0 {
        push_rect_px(
            &mut out,
            sw,
            sh,
            [bar_x, xp_bar_y, fill_w, 5.0],
            [118, 246, 92, 255],
        );
    }
    if m.xp_level > 0 {
        let lvl_str = m.xp_level.to_string();
        let lx = cx - (lvl_str.len() as f32) * 5.5;
        push_text_px(
            &mut out,
            (sw, sh),
            (lx, xp_bar_y - 11.0),
            1.7,
            &lvl_str,
            [138, 255, 108, 255],
        );
    }
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

    let sel = m.selected_slot.min(8);
    let held_label = m.inventory_slots[sel].label;
    if !held_label.is_empty() {
        let label_w = (held_label.len() as f32) * 9.6;
        push_text_px(
            &mut out,
            (sw, sh),
            (cx - label_w * 0.5, hearts_y - 18.0),
            1.6,
            held_label,
            [248, 244, 186, 245],
        );
    }

    push_rect_px(
        &mut out,
        sw,
        sh,
        [bar_x, bar_y - 6.0, total_w, slot_px + 12.0],
        [18, 20, 26, 196],
    );

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

        let slot_disp = m.inventory_slots[i];
        let (top_c, side_c, draw_icon) = if slot_disp.count > 0 {
            (slot_disp.top_color, slot_disp.side_color, true)
        } else if !m.survival_physics {
            (m.hotbar_top_colors[i], m.hotbar_side_colors[i], true)
        } else {
            ([0; 4], [0; 4], false)
        };

        if draw_icon {
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
            if slot_disp.count > 1 {
                let cnt_txt = slot_disp.count.to_string();
                let tx = sx + slot_px - (cnt_txt.len() as f32) * 9.0 - 3.0;
                push_text_px(
                    &mut out,
                    (sw, sh),
                    (tx, sy + slot_px - 13.0),
                    1.5,
                    &cnt_txt,
                    [255, 255, 255, 255],
                );
            }
        }
    }

    // 4. Recent Chat Feed (bottom-left above hotbar)
    if !m.chat_lines.is_empty() {
        let max_lines = if m.chat_open { 8 } else { 5 };
        let count = m.chat_lines.len().min(max_lines);
        let box_h = (count as f32) * 18.0 + 10.0;
        let box_y = (bar_y - 44.0 - box_h).max(76.0);
        push_rect_px(
            &mut out,
            sw,
            sh,
            [12.0, box_y, 500.0, box_h],
            [12, 14, 20, if m.chat_open { 215 } else { 150 }],
        );
        for (idx, line) in m.chat_lines.iter().rev().take(max_lines).rev().enumerate() {
            let clipped: String = line.chars().take(62).collect();
            push_text_px(
                &mut out,
                (sw, sh),
                (20.0, box_y + 6.0 + (idx as f32) * 18.0),
                1.7,
                &clipped,
                [240, 244, 255, 245],
            );
        }
    }

    // 5. Live Interactive Chat / Command Input Bar (`T` or `/`)
    if m.chat_open {
        let iy = sh - 22.0;
        push_rect_px(
            &mut out,
            sw,
            sh,
            [12.0, iy - 8.0, sw - 24.0, 24.0],
            [10, 12, 18, 235],
        );
        let shown: String = m.chat_input.chars().take(78).collect();
        push_text_px(
            &mut out,
            (sw, sh),
            (20.0, iy - 3.0),
            1.8,
            &format!("> {shown}_"),
            [120, 245, 178, 255],
        );
        if !m.tab_hint.is_empty() {
            push_text_px(
                &mut out,
                (sw, sh),
                (sw - 320.0, iy - 3.0),
                1.6,
                &format!("TAB: {}", m.tab_hint),
                [250, 224, 130, 245],
            );
        }
    }

    // 6. Interactive `E` Inventory & Quick-Crafting Overlay
    if m.inventory_open {
        push_rect_px(&mut out, sw, sh, [0.0, 0.0, sw, sh], [8, 10, 16, 178]);
        let pw = 740.0f32;
        let ph = 340.0f32;
        let px = (cx - pw * 0.5).max(12.0);
        let py = (cy - ph * 0.5 - 16.0).max(20.0);

        push_rect_px(
            &mut out,
            sw,
            sh,
            [px - 3.0, py - 3.0, pw + 6.0, ph + 6.0],
            [110, 215, 158, 240],
        );
        push_rect_px(&mut out, sw, sh, [px, py, pw, ph], [22, 26, 34, 246]);

        push_text_px(
            &mut out,
            (sw, sh),
            (px + 20.0, py + 14.0),
            2.1,
            "INVENTORY & SURVIVAL CRAFTING",
            [110, 235, 165, 255],
        );
        push_text_px(
            &mut out,
            (sw, sh),
            (px + 20.0, py + 38.0),
            1.5,
            "[X] EQUIP NEXT BACKPACK ITEM TO HOTBAR   [E / ESC] CLOSE",
            [188, 202, 226, 240],
        );

        // 3x9 Backpack Grid (slots 9..36)
        let cell = 38.0f32;
        let cgap = 4.0f32;
        let gx = px + 20.0;
        let gy = py + 66.0;
        for row in 0..3usize {
            for col in 0..9usize {
                let idx = 9 + row * 9 + col;
                let sx = gx + (col as f32) * (cell + cgap);
                let sy = gy + (row as f32) * (cell + cgap);
                push_rect_px(&mut out, sw, sh, [sx, sy, cell, cell], [38, 44, 56, 240]);
                let s = m.inventory_slots[idx];
                if s.count > 0 {
                    push_rect_px(
                        &mut out,
                        sw,
                        sh,
                        [sx + 7.0, sy + 7.0, cell - 14.0, 9.0],
                        [s.top_color[0], s.top_color[1], s.top_color[2], 255],
                    );
                    push_rect_px(
                        &mut out,
                        sw,
                        sh,
                        [sx + 7.0, sy + 16.0, cell - 14.0, cell - 23.0],
                        [s.side_color[0], s.side_color[1], s.side_color[2], 255],
                    );
                    let cstr = s.count.to_string();
                    push_text_px(
                        &mut out,
                        (sw, sh),
                        (
                            sx + cell - (cstr.len() as f32) * 8.0 - 2.0,
                            sy + cell - 12.0,
                        ),
                        1.3,
                        &cstr,
                        [255, 255, 255, 255],
                    );
                }
            }
        }

        // Hotbar mirror row inside Inventory (slots 0..9)
        let hy = gy + 3.0 * (cell + cgap) + 14.0;
        for (col, &s) in m.inventory_slots.iter().enumerate().take(9) {
            let sx = gx + (col as f32) * (cell + cgap);
            if col == sel {
                push_rect_px(
                    &mut out,
                    sw,
                    sh,
                    [sx - 2.0, hy - 2.0, cell + 4.0, cell + 4.0],
                    [248, 250, 255, 245],
                );
            }
            push_rect_px(&mut out, sw, sh, [sx, hy, cell, cell], [38, 44, 56, 240]);
            if s.count > 0 {
                push_rect_px(
                    &mut out,
                    sw,
                    sh,
                    [sx + 7.0, hy + 7.0, cell - 14.0, 9.0],
                    [s.top_color[0], s.top_color[1], s.top_color[2], 255],
                );
                push_rect_px(
                    &mut out,
                    sw,
                    sh,
                    [sx + 7.0, hy + 16.0, cell - 14.0, cell - 23.0],
                    [s.side_color[0], s.side_color[1], s.side_color[2], 255],
                );
            }
        }

        // Right-hand Quick-Craft Panel (`[1]..[9]` keys craft instantly!)
        let rx = gx + 9.0 * (cell + cgap) + 22.0;
        push_text_px(
            &mut out,
            (sw, sh),
            (rx, py + 44.0),
            1.7,
            "QUICK CRAFT (PRESS 1-9):",
            [250, 224, 130, 255],
        );
        for (i, &(_key, desc)) in QUICK_CRAFT_RECIPES.iter().enumerate() {
            let ry = py + 68.0 + (i as f32) * 28.0;
            push_rect_px(
                &mut out,
                sw,
                sh,
                [rx - 4.0, ry - 4.0, pw - (rx - px) - 16.0, 23.0],
                [34, 40, 52, 235],
            );
            push_text_px(
                &mut out,
                (sw, sh),
                (rx + 2.0, ry + 2.0),
                1.4,
                desc,
                [235, 242, 255, 255],
            );
        }
    }

    // 7. Death Screen Overlay (`health <= 0.0`)
    if m.health <= 0.0 {
        push_rect_px(&mut out, sw, sh, [0.0, 0.0, sw, sh], [120, 12, 18, 155]);
        push_text_px(
            &mut out,
            (sw, sh),
            (cx - 108.0, cy - 36.0),
            3.6,
            "YOU DIED!",
            [255, 236, 236, 255],
        );
        push_text_px(
            &mut out,
            (sw, sh),
            (cx - 162.0, cy + 12.0),
            2.0,
            "PRESS [R] TO RESPAWN AT SPAWN",
            [255, 224, 150, 255],
        );
    } else if m.paused {
        // 8. Esc Pause & Settings Overlay Menu
        push_rect_px(&mut out, sw, sh, [0.0, 0.0, sw, sh], [8, 10, 16, 168]);
        let mw = 390.0f32;
        let mh = 240.0f32;
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
            (mx + 100.0, my + 20.0),
            2.6,
            "SOLARA PAUSED",
            [110, 235, 165, 255],
        );

        let items = [
            "[ESC] RESUME GAME",
            if m.survival_physics {
                "[F]   MODE: SURVIVAL WALKING"
            } else {
                "[F]   MODE: CREATIVE FLY"
            },
            "[E]   OPEN INVENTORY & CRAFT",
            "[Q]   DISCONNECT AND EXIT",
        ];
        for (idx, label) in items.iter().enumerate() {
            let iy = my + 64.0 + (idx as f32) * 38.0;
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
