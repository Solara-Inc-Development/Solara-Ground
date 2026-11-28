//! Client-side player progression, XP bar state, and equipped armor defense calculator
//! for Solara-Ground.

use ground_data::generated_items::id as item_id;

/// Returns armor defense points (`0..=8`) granted by an item ID when worn in an armor slot.
pub const fn armor_points_for_item(item_id: u32) -> u8 {
    match item_id {
        item_id::LEATHER_BOOTS => 1,
        item_id::LEATHER_LEGGINGS => 2,
        item_id::LEATHER_CHESTPLATE => 3,
        item_id::LEATHER_HELMET => 1,
        item_id::IRON_BOOTS => 2,
        item_id::IRON_LEGGINGS => 5,
        item_id::IRON_CHESTPLATE => 6,
        item_id::IRON_HELMET => 2,
        item_id::DIAMOND_BOOTS => 3,
        item_id::DIAMOND_LEGGINGS => 6,
        item_id::DIAMOND_CHESTPLATE => 8,
        item_id::DIAMOND_HELMET => 3,
        _ => 0,
    }
}

/// Live client-side armor & experience state synchronized from `SET_PLAYER_INVENTORY` and `SET_EXPERIENCE`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProgressionState {
    pub armor_items: [u32; 4],
    pub xp_progress: f32,
    pub xp_level: i32,
    pub total_xp: i32,
}

impl ProgressionState {
    pub fn set_armor_slot(&mut self, armor_idx: usize, item_id: u32) {
        if armor_idx < self.armor_items.len() {
            self.armor_items[armor_idx] = item_id;
        }
    }

    pub fn total_armor_points(&self) -> u8 {
        self.armor_items
            .iter()
            .copied()
            .map(armor_points_for_item)
            .sum::<u8>()
            .min(20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_full_iron_and_diamond_armor_points() {
        let mut p = ProgressionState::default();
        p.set_armor_slot(0, item_id::IRON_BOOTS);
        p.set_armor_slot(1, item_id::IRON_LEGGINGS);
        p.set_armor_slot(2, item_id::IRON_CHESTPLATE);
        p.set_armor_slot(3, item_id::IRON_HELMET);
        assert_eq!(p.total_armor_points(), 15);
    }
}
