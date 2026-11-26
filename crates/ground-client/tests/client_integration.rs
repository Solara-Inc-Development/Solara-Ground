use ground_audio::{SfxEvent, SoundTriggerQueue};
use ground_data::{entity_hitbox, generated_entities, generated_items};
use ground_ui::{HudModel, build_full_ui_vertices};
use ground_world::ChunkSection;
use ground_world::mesh::mesh_section;

#[test]
fn cross_crate_client_data_ui_audio_and_greedy_mesher_integration() {
    assert!(generated_items::by_key("netherite_pickaxe").is_some());
    assert_eq!(entity_hitbox(generated_entities::id::ZOMBIE), (0.6, 1.95));

    let mut sec = ChunkSection::new_air();
    for z in 0..16 {
        for x in 0..16 {
            sec.set(
                x,
                0,
                z,
                ground_world::generated_blocks::default_state::GRASS_BLOCK,
            );
        }
    }
    assert_eq!(mesh_section(&sec).quad_count(), 6);

    let mut sfx = SoundTriggerQueue::new("./assets");
    assert!(sfx.trigger_spatial(SfxEvent::Footstep, [0.0; 3], [1.0, 0.0, 0.0]) > 1000);

    let ui = build_full_ui_vertices(&HudModel {
        width: 1920,
        height: 1080,
        selected_slot: 4,
        hotbar_top_colors: [[120, 180, 90, 255]; 9],
        hotbar_side_colors: [[110, 85, 60, 255]; 9],
        health: 20.0,
        food: 20,
        survival_physics: true,
        paused: false,
        fps: 320,
        frame_ms: 3.1,
        camera_pos: [0.0, -58.0, 0.0],
        chunk_count: 81,
        entity_count: 4,
        chat_lines: vec!["Solara-Ground integration test".into()],
    });
    assert!(ui.len() > 400);
}
