//! Client-side clean-room generated Minecraft: Java Edition 26.3 (protocol 777) tables:
//! items (`1,658`), entity types (`161`) + hitboxes, sound events (`1,991`), particles (`128`), and effects (`40`).

pub mod generated_effects;
pub mod generated_entities;
pub mod generated_items;
pub mod generated_particles;
pub mod generated_sounds;

const _: () = {
    assert!(generated_items::TOTAL_ITEMS >= 1600);
    assert!(generated_sounds::TOTAL_SOUND_EVENTS >= 1900);
};

/// Returns `(width, height)` bounding-box dimensions in blocks for `entity_type`.
pub const fn entity_hitbox(entity_type: u32) -> (f32, f32) {
    match entity_type {
        generated_entities::id::PLAYER => (0.6, 1.8),
        generated_entities::id::ZOMBIE | generated_entities::id::SKELETON => (0.6, 1.95),
        generated_entities::id::CREEPER => (0.6, 1.7),
        generated_entities::id::PIG => (0.9, 0.9),
        generated_entities::id::COW => (0.9, 1.4),
        generated_entities::id::SHEEP => (0.9, 1.3),
        generated_entities::id::CHICKEN => (0.4, 0.7),
        _ => (0.8, 1.5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_client_items_entities_and_hitboxes() {
        assert!(generated_items::by_key("diamond_sword").is_some());
        assert_eq!(entity_hitbox(generated_entities::id::PLAYER), (0.6, 1.8));
        assert_eq!(entity_hitbox(generated_entities::id::PIG), (0.9, 0.9));
    }
}
