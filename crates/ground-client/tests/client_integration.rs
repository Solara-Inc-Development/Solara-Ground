use ground_audio::{
    SAMPLE_RATE_HZ, SfxEvent, SoundTriggerQueue, encode_wav_16bit_mono, synthesize_clean_room_pcm,
};
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

    let pcm = synthesize_clean_room_pcm(SfxEvent::ItemPickup, 0.9);
    let wav = encode_wav_16bit_mono(&pcm, SAMPLE_RATE_HZ);
    assert_eq!(&wav[..4], b"RIFF");

    let mut sfx = SoundTriggerQueue::new("./assets");
    assert!(sfx.trigger_spatial(SfxEvent::Footstep, [0.0; 3], [1.0, 0.0, 0.0]) > 900);

    let ui = build_full_ui_vertices(&HudModel {
        width: 1920,
        height: 1080,
        selected_slot: 4,
        health: 20.0,
        food: 20,
        inventory_open: true,
        chat_open: true,
        chat_input: "/craft wooden_pickaxe".into(),
        fps: 320,
        frame_ms: 3.1,
        camera_pos: [0.0, -58.0, 0.0],
        chunk_count: 81,
        entity_count: 4,
        chat_lines: vec!["Solara-Ground playable integration test".into()],
        ..Default::default()
    });
    assert!(ui.len() > 1000);
}
