//! Client-side active status potion effect state (`Speed`, `Strength`, `Regeneration`,
//! `Resistance`, `Night Vision`) for HUD badges, movement speed scaling, and night-vision lighting.

use ground_data::generated_effects::id as effect_id;

#[derive(Debug, Clone, Copy)]
pub struct ClientActiveEffect {
    pub effect_id: u32,
    pub amplifier: u8,
    pub remaining_secs: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ClientEffects {
    active: Vec<ClientActiveEffect>,
}

impl ClientEffects {
    pub fn new() -> Self {
        Self { active: Vec::new() }
    }

    pub fn apply_or_refresh(&mut self, effect_id: u32, amplifier: u8, duration_ticks: i32) {
        let secs = (duration_ticks.max(0) as f32) * 0.05;
        if secs <= 0.0 {
            return;
        }
        if let Some(ex) = self.active.iter_mut().find(|e| e.effect_id == effect_id) {
            ex.amplifier = amplifier;
            ex.remaining_secs = secs;
        } else {
            self.active.push(ClientActiveEffect {
                effect_id,
                amplifier,
                remaining_secs: secs,
            });
        }
    }

    pub fn remove(&mut self, effect_id: u32) {
        self.active.retain(|e| e.effect_id != effect_id);
    }

    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Steps countdown timers; returns `true` when integer displayed seconds or active set changed.
    pub fn step(&mut self, dt_secs: f32) -> bool {
        if self.active.is_empty() {
            return false;
        }
        let dt = dt_secs.clamp(0.0, 0.25);
        let mut ui_changed = false;
        let mut i = 0;
        while i < self.active.len() {
            let before = self.active[i].remaining_secs.ceil() as i32;
            self.active[i].remaining_secs -= dt;
            if self.active[i].remaining_secs <= 0.0 {
                self.active.swap_remove(i);
                ui_changed = true;
            } else {
                let after = self.active[i].remaining_secs.ceil() as i32;
                if after != before {
                    ui_changed = true;
                }
                i += 1;
            }
        }
        ui_changed
    }

    pub fn speed_multiplier(&self) -> f32 {
        self.active
            .iter()
            .find(|e| e.effect_id == effect_id::SPEED)
            .map_or(1.0, |e| 1.0 + 0.20 * f32::from(e.amplifier + 1))
    }

    pub fn has_night_vision(&self) -> bool {
        self.active
            .iter()
            .any(|e| e.effect_id == effect_id::NIGHT_VISION)
    }

    pub fn hud_badges(&self) -> Vec<(String, [u8; 4])> {
        self.active
            .iter()
            .map(|e| {
                let (tag, color) = match e.effect_id {
                    effect_id::SPEED => ("SPEED", [92, 212, 250, 245]),
                    effect_id::STRENGTH => ("STRENGTH", [242, 88, 88, 245]),
                    effect_id::REGENERATION => ("REGEN", [248, 126, 198, 245]),
                    effect_id::RESISTANCE => ("RESIST", [176, 188, 208, 245]),
                    effect_id::NIGHT_VISION => ("NIGHT VIS", [112, 242, 148, 245]),
                    effect_id::HASTE => ("HASTE", [246, 208, 84, 245]),
                    effect_id::JUMP_BOOST => ("JUMP", [132, 246, 118, 245]),
                    _ => ("EFFECT", [190, 168, 248, 245]),
                };
                let secs = e.remaining_secs.ceil().max(1.0) as u32;
                (format!("{tag} {secs}S"), color)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_night_vision_and_hud_badges_update_cleanly() {
        let mut fx = ClientEffects::new();
        fx.apply_or_refresh(effect_id::SPEED, 0, 200);
        fx.apply_or_refresh(effect_id::NIGHT_VISION, 0, 40);
        assert!((fx.speed_multiplier() - 1.20).abs() < 1e-5);
        assert!(fx.has_night_vision());
        assert_eq!(fx.hud_badges().len(), 2);

        for _ in 0..10 {
            let _ = fx.step(0.25);
        }
        assert!(!fx.has_night_vision());
        assert_eq!(fx.hud_badges().len(), 1);
    }
}
