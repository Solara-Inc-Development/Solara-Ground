pub mod session;

use crate::session::{RedactedToken, SessionConfig};
use anyhow::{Context, Result, bail};
use clap::Parser;
use glam::Vec3;
use ground_audio::{SfxEvent, SoundTriggerQueue};
use ground_data::entity_hitbox;
use ground_protocol::client::{ClientConnection, login_online, ping_server};
use ground_protocol::config_play::{
    PlayServerEvent, bootstrap_play_session, complete_configuration, encode_attack_entity,
    encode_chat_command, encode_chat_message, encode_move_player_pos_rot, encode_perform_respawn,
    encode_request_command_suggestions, encode_set_carried_item, encode_start_destroy_block,
    encode_stop_destroy_block, encode_swing_arm, encode_use_item_air, encode_use_item_on,
    read_next_play_event,
};
use ground_protocol::generated_packets;
use ground_render::{
    ClientWindowAction, Frustum, LiveRenderBridge, RemoteEntityVisual, RenderSceneUpdate,
    VoxelCamera, WindowLaunchConfig, WorldSectionMesh, build_frustum_culled_indirect_cmds,
    pack_mega_mesh_batch, run_voxel_window,
};
use ground_world::assets::LocalMojangAssets;
use ground_world::mesh::mesh_chunk_column;
use ground_world::physics::{ClientWorldMap, PlayerBody, raycast_blocks};
use ground_world::{
    ChunkPos, OVERWORLD_MIN_Y, OVERWORLD_SECTIONS, decode_and_mesh_chunks_parallel,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use tracing::info;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(
    name = "solara-ground",
    about = "High-performance Rust Minecraft: Java Edition 26.3 client (Windows & Linux)"
)]
struct Cli {
    /// Ping a Minecraft 26.3 server (e.g. `127.0.0.1:25565`) and exit
    #[arg(long)]
    ping: Option<String>,

    /// Connect to a Minecraft 26.3 server (e.g. `127.0.0.1:25565`) and stream/render the world
    #[arg(long)]
    connect: Option<String>,

    /// Run protocol + chunk decoding + greedy meshing + Mega-Buffer indirect culling without a window
    #[arg(long)]
    headless_smoke: bool,

    /// Run gameplay smoke (AABB physics, DDA raycast, block break/place, /save Anvil command)
    #[arg(long)]
    gameplay_smoke: bool,

    /// Run full Phase 4–6 multiplayer, mobs, PvP/survival health, Tab auto-complete, and Mega-Buffer E2E test
    #[cfg(feature = "dev-offline")]
    #[arg(long)]
    multiplayer_smoke: bool,

    /// Authenticated player username (provided by solara-launcher in online mode)
    #[arg(long)]
    username: Option<String>,

    /// Authenticated player UUID (provided by solara-launcher in online mode)
    #[arg(long)]
    uuid: Option<Uuid>,

    /// Authenticated Minecraft access token (provided by solara-launcher; never logged)
    #[arg(long)]
    access_token: Option<String>,

    /// Path to local Mojang assets directory downloaded from Mojang's official CDN
    #[arg(long, default_value = "./assets")]
    assets_dir: PathBuf,
}

fn parse_host_port(addr: &str) -> Result<(&str, u16)> {
    if let Some((host, port_str)) = addr.rsplit_once(':') {
        let port: u16 = port_str
            .parse()
            .with_context(|| format!("invalid port in '{addr}'"))?;
        Ok((host, port))
    } else {
        Ok((addr, 25565))
    }
}

fn flatten_chunk_meshes(
    chunk_meshes: &HashMap<ChunkPos, Vec<WorldSectionMesh>>,
) -> Vec<WorldSectionMesh> {
    let mut flat = Vec::new();
    for list in chunk_meshes.values() {
        flat.extend(list.iter().cloned());
    }
    flat
}

/// Ray-AABB intersection against visible entities within `max_reach` blocks (for melee combat!).
fn raycast_entity_target(
    entities: &HashMap<i32, RemoteEntityVisual>,
    origin: [f64; 3],
    dir: [f64; 3],
    max_reach: f64,
) -> Option<i32> {
    let mut best: Option<(f64, i32)> = None;
    for (&eid, ent) in entities {
        let (w, h) = entity_hitbox(ent.entity_type);
        let half = f64::from(w) * 0.5;
        let min_b = [
            f64::from(ent.feet_pos.x) - half,
            f64::from(ent.feet_pos.y),
            f64::from(ent.feet_pos.z) - half,
        ];
        let max_b = [
            f64::from(ent.feet_pos.x) + half,
            f64::from(ent.feet_pos.y + h),
            f64::from(ent.feet_pos.z) + half,
        ];

        let mut tmin = 0.0f64;
        let mut tmax = max_reach;
        let mut hit = true;
        for axis in 0..3 {
            if dir[axis].abs() < 1e-9 {
                if origin[axis] < min_b[axis] || origin[axis] > max_b[axis] {
                    hit = false;
                    break;
                }
            } else {
                let inv = 1.0 / dir[axis];
                let mut t1 = (min_b[axis] - origin[axis]) * inv;
                let mut t2 = (max_b[axis] - origin[axis]) * inv;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                tmin = tmin.max(t1);
                tmax = tmax.min(t2);
                if tmin > tmax {
                    hit = false;
                    break;
                }
            }
        }
        if hit && best.is_none_or(|(best_t, _)| tmin < best_t) {
            best = Some((tmin, eid));
        }
    }
    best.map(|(_, id)| id)
}

fn spawn_live_network_bridge(
    mut conn: ClientConnection,
    mut world_map: ClientWorldMap,
    mut chunk_meshes: HashMap<ChunkPos, Vec<WorldSectionMesh>>,
    initial_entities: Vec<RemoteEntityVisual>,
    spawn_feet: [f64; 3],
    assets_dir: PathBuf,
) -> LiveRenderBridge {
    use generated_packets::play::serverbound as c2s;

    let (scene_tx, scene_rx) = mpsc::channel::<RenderSceneUpdate>();
    let (action_tx, action_rx) = mpsc::channel::<ClientWindowAction>();

    let mut entities: HashMap<i32, RemoteEntityVisual> = initial_entities
        .into_iter()
        .map(|e| (e.entity_id, e))
        .collect();

    std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
        else {
            return;
        };

        rt.block_on(async move {
            let mut body = PlayerBody::new(spawn_feet);
            let mut sfx = SoundTriggerQueue::new(assets_dir);
            let mut next_seq = 1i32;
            let mut next_tab_tx = 100i32;
            let mut net_accum = 0.0f64;
            let mut step_accum = 0.0f64;
            let mut drain_tick = tokio::time::interval(std::time::Duration::from_millis(16));
            drain_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = drain_tick.tick() => {
                        let mut disconnected = false;
                        loop {
                            match action_rx.try_recv() {
                                Ok(ClientWindowAction::TickCamera {
                                    eye_pos,
                                    look_dir,
                                    wish_xz,
                                    jump,
                                    survival_physics,
                                    yaw_deg,
                                    pitch_deg,
                                    dt,
                                }) => {
                                    let effective_eye = if survival_physics {
                                        body.step(&world_map, wish_xz, jump, dt);
                                        if body.on_ground && (wish_xz[0].abs() > 0.1 || wish_xz[1].abs() > 0.1) {
                                            step_accum += dt;
                                            if step_accum >= 0.38 {
                                                step_accum = 0.0;
                                                sfx.trigger_spatial(
                                                    SfxEvent::Footstep,
                                                    [body.feet_pos[0] as f32, body.feet_pos[1] as f32, body.feet_pos[2] as f32],
                                                    [body.feet_pos[0] as f32, body.feet_pos[1] as f32, body.feet_pos[2] as f32],
                                                );
                                            }
                                        }
                                        let eye = body.eye_pos();
                                        let _ = scene_tx.send(RenderSceneUpdate::TeleportCamera {
                                            position: Vec3::new(eye[0] as f32, eye[1] as f32, eye[2] as f32),
                                            yaw_rad: yaw_deg.to_radians(),
                                            pitch_rad: pitch_deg.to_radians(),
                                        });
                                        eye
                                    } else {
                                        body.feet_pos = [eye_pos[0], eye_pos[1] - 1.62, eye_pos[2]];
                                        body.velocity = [0.0, 0.0, 0.0];
                                        eye_pos
                                    };

                                    let target = raycast_blocks(&world_map, effective_eye, look_dir, 6.0)
                                        .map(|h| [h.block_pos.0, h.block_pos.1, h.block_pos.2]);
                                    let _ = scene_tx.send(RenderSceneUpdate::TargetBlock(target));

                                    net_accum += dt;
                                    if net_accum >= 0.05 {
                                        net_accum = 0.0;
                                        let _ = conn.send_packet(
                                            c2s::MOVE_PLAYER_POS_ROT,
                                            &encode_move_player_pos_rot(
                                                body.feet_pos,
                                                yaw_deg,
                                                pitch_deg,
                                                if survival_physics { body.on_ground } else { false },
                                            ),
                                        ).await;
                                    }
                                }
                                Ok(ClientWindowAction::SelectHotbarSlot(slot)) => {
                                    sfx.trigger_spatial(SfxEvent::UiClick, [0.0; 3], [0.0; 3]);
                                    let _ = conn.send_packet(
                                        c2s::SET_CARRIED_ITEM,
                                        &encode_set_carried_item(slot),
                                    ).await;
                                }
                                Ok(ClientWindowAction::PrimaryAttackOrBreak { eye_pos, look_dir }) => {
                                    let _ = conn.send_packet(c2s::PUNCH, &encode_swing_arm()).await;
                                    if let Some(target_entity) = raycast_entity_target(&entities, eye_pos, look_dir, 3.8) {
                                        sfx.trigger_spatial(SfxEvent::Hurt, [eye_pos[0] as f32, eye_pos[1] as f32, eye_pos[2] as f32], [eye_pos[0] as f32, eye_pos[1] as f32, eye_pos[2] as f32]);
                                        let _ = conn.send_packet(
                                            c2s::ATTACK,
                                            &encode_attack_entity(target_entity),
                                        ).await;
                                    } else if let Some(hit) = raycast_blocks(&world_map, eye_pos, look_dir, 6.0) {
                                        sfx.trigger_spatial(
                                            SfxEvent::BlockBreak,
                                            [eye_pos[0] as f32, eye_pos[1] as f32, eye_pos[2] as f32],
                                            [hit.block_pos.0 as f32, hit.block_pos.1 as f32, hit.block_pos.2 as f32],
                                        );
                                        let seq_start = next_seq;
                                        let seq_stop = next_seq + 1;
                                        next_seq += 2;
                                        let _ = conn.send_packet(
                                            c2s::PLAYER_ACTION,
                                            &encode_start_destroy_block(hit.block_pos, hit.face_direction as u8, seq_start),
                                        ).await;
                                        let _ = conn.send_packet(
                                            c2s::PLAYER_ACTION,
                                            &encode_stop_destroy_block(hit.block_pos, hit.face_direction as u8, seq_stop),
                                        ).await;
                                    }
                                }
                                Ok(ClientWindowAction::PlaceLook {
                                    eye_pos,
                                    look_dir,
                                    yaw_deg,
                                    pitch_deg,
                                }) => {
                                    if let Some(hit) = raycast_blocks(&world_map, eye_pos, look_dir, 6.0) {
                                        sfx.trigger_spatial(
                                            SfxEvent::BlockPlace,
                                            [eye_pos[0] as f32, eye_pos[1] as f32, eye_pos[2] as f32],
                                            [hit.adjacent_pos.0 as f32, hit.adjacent_pos.1 as f32, hit.adjacent_pos.2 as f32],
                                        );
                                        let seq = next_seq;
                                        next_seq += 1;
                                        let _ = conn.send_packet(
                                            c2s::USE_ITEM_ON,
                                            &encode_use_item_on(hit.block_pos, hit.face_direction, seq),
                                        ).await;
                                    } else {
                                        sfx.trigger_spatial(SfxEvent::EatFood, [0.0; 3], [0.0; 3]);
                                        let seq = next_seq;
                                        next_seq += 1;
                                        let _ = conn.send_packet(
                                            c2s::USE_ITEM,
                                            &encode_use_item_air(seq, yaw_deg, pitch_deg),
                                        ).await;
                                    }
                                }
                                Ok(ClientWindowAction::SendChatOrCommand(msg)) => {
                                    let trimmed = msg.trim();
                                    if let Some(cmd) = trimmed.strip_prefix('/') {
                                        let _ = conn.send_packet(c2s::CHAT_COMMAND, &encode_chat_command(cmd)).await;
                                    } else if !trimmed.is_empty() {
                                        let _ = conn.send_packet(c2s::CHAT, &encode_chat_message(trimmed)).await;
                                    }
                                }
                                Ok(ClientWindowAction::RequestTabComplete(input)) => {
                                    let tx = next_tab_tx;
                                    next_tab_tx += 1;
                                    let _ = conn.send_packet(
                                        c2s::COMMAND_SUGGESTION,
                                        &encode_request_command_suggestions(tx, &input),
                                    ).await;
                                }
                                Ok(ClientWindowAction::Respawn) => {
                                    let _ = conn.send_packet(c2s::CLIENT_COMMAND, &encode_perform_respawn()).await;
                                }
                                Err(mpsc::TryRecvError::Empty) => break,
                                Err(mpsc::TryRecvError::Disconnected) => {
                                    disconnected = true;
                                    break;
                                }
                            }
                        }
                        if disconnected {
                            break;
                        }
                    }
                    ev_res = read_next_play_event(&mut conn) => {
                        let Ok(ev) = ev_res else {
                            break;
                        };
                        match ev {
                            PlayServerEvent::LevelChunkWithLight(raw) => {
                                let batch = vec![raw];
                                if let Ok(Ok(cols)) = tokio::task::spawn_blocking(move || {
                                    decode_and_mesh_chunks_parallel(&batch, OVERWORLD_SECTIONS)
                                }).await {
                                    for col in cols {
                                        let mut ws = Vec::new();
                                        for (sec_idx, mesh) in col.meshes.into_iter().enumerate() {
                                            if !mesh.is_empty() {
                                                let world_y = OVERWORLD_MIN_Y + (sec_idx as i32) * 16;
                                                ws.push(WorldSectionMesh {
                                                    origin: Vec3::new(
                                                        (col.pos.x * 16) as f32,
                                                        world_y as f32,
                                                        (col.pos.z * 16) as f32,
                                                    ),
                                                    mesh,
                                                });
                                            }
                                        }
                                        chunk_meshes.insert(col.pos, ws);
                                        world_map.insert_chunk(col.pos, col.sections);
                                    }
                                    if scene_tx.send(RenderSceneUpdate::ReplaceSections {
                                        sections: flatten_chunk_meshes(&chunk_meshes),
                                        chunk_count: world_map.chunk_count(),
                                    }).is_err() {
                                        break;
                                    }
                                }
                            }
                            PlayServerEvent::BlockUpdate { pos, state_id } => {
                                if let Some((cpos, _)) = world_map.set_block(pos.0, pos.1, pos.2, state_id)
                                    && let Some((_, sections)) = world_map.chunks_iter().find(|&(p, _)| *p == cpos)
                                {
                                    let meshes = mesh_chunk_column(sections);
                                    let mut ws = Vec::new();
                                    for (sec_idx, mesh) in meshes.into_iter().enumerate() {
                                        if !mesh.is_empty() {
                                            let world_y = OVERWORLD_MIN_Y + (sec_idx as i32) * 16;
                                            ws.push(WorldSectionMesh {
                                                origin: Vec3::new(
                                                    (cpos.x * 16) as f32,
                                                    world_y as f32,
                                                    (cpos.z * 16) as f32,
                                                ),
                                                mesh,
                                            });
                                        }
                                    }
                                    chunk_meshes.insert(cpos, ws);
                                    let _ = scene_tx.send(RenderSceneUpdate::ReplaceSections {
                                        sections: flatten_chunk_meshes(&chunk_meshes),
                                        chunk_count: world_map.chunk_count(),
                                    });
                                }
                            }
                            PlayServerEvent::AddEntity(ent) => {
                                entities.insert(
                                    ent.entity_id,
                                    RemoteEntityVisual {
                                        entity_id: ent.entity_id,
                                        entity_type: ent.entity_type.max(0) as u32,
                                        feet_pos: Vec3::new(ent.position[0] as f32, ent.position[1] as f32, ent.position[2] as f32),
                                        yaw_deg: ent.yaw_deg,
                                    },
                                );
                                let _ = scene_tx.send(RenderSceneUpdate::Entities(entities.values().copied().collect()));
                            }
                            PlayServerEvent::TeleportEntity(tp) => {
                                if let Some(e) = entities.get_mut(&tp.entity_id) {
                                    e.feet_pos = Vec3::new(tp.position[0] as f32, tp.position[1] as f32, tp.position[2] as f32);
                                    e.yaw_deg = tp.yaw_deg;
                                    let _ = scene_tx.send(RenderSceneUpdate::Entities(entities.values().copied().collect()));
                                }
                            }
                            PlayServerEvent::RemoveEntities(ids) => {
                                let mut changed = false;
                                for id in ids {
                                    changed |= entities.remove(&id).is_some();
                                }
                                if changed {
                                    let _ = scene_tx.send(RenderSceneUpdate::Entities(entities.values().copied().collect()));
                                }
                            }
                            PlayServerEvent::SetHealth { health, food, .. } => {
                                let _ = scene_tx.send(RenderSceneUpdate::Vitals { health, food });
                            }
                            PlayServerEvent::SetTime { day_time, .. } => {
                                let _ = scene_tx.send(RenderSceneUpdate::DayTime(day_time));
                            }
                            PlayServerEvent::SetPlayerInventory { slot, item_id, count } => {
                                let _ = scene_tx.send(RenderSceneUpdate::InventorySlot { slot, item_id, count });
                            }
                            PlayServerEvent::TakeItemEntity { .. } => {
                                sfx.trigger_spatial(SfxEvent::ItemPickup, [0.0; 3], [0.0; 3]);
                            }
                            PlayServerEvent::CommandSuggestions { matches, .. } => {
                                let first = matches.into_iter().next().unwrap_or_default();
                                let _ = scene_tx.send(RenderSceneUpdate::TabHint(first));
                            }
                            PlayServerEvent::SystemChat { text, .. } => {
                                let _ = scene_tx.send(RenderSceneUpdate::ChatLine(text));
                            }
                            PlayServerEvent::Teleport(tp) => {
                                body.feet_pos = tp.position;
                                body.velocity = [0.0, 0.0, 0.0];
                                let _ = scene_tx.send(RenderSceneUpdate::TeleportCamera {
                                    position: Vec3::new(tp.position[0] as f32, tp.position[1] as f32 + 1.62, tp.position[2] as f32),
                                    yaw_rad: tp.yaw_deg.to_radians(),
                                    pitch_rad: tp.pitch_deg.to_radians(),
                                });
                            }
                            _ => {}
                        }
                    }
                }
            }
        });
    });

    LiveRenderBridge {
        scene_rx,
        action_tx,
    }
}

async fn run_gameplay_smoke_session(
    conn: &mut ClientConnection,
    world: &mut ClientWorldMap,
    spawn_pos: [f64; 3],
) -> Result<()> {
    use generated_packets::play::serverbound as c2s;

    let mut body = PlayerBody::new(spawn_pos);
    for _ in 0..30 {
        body.step(world, [0.0, 0.0], false, 0.05);
    }
    if !body.on_ground {
        bail!(
            "expected player physics to land on ground, feet={:?}",
            body.feet_pos
        );
    }
    conn.send_packet(
        c2s::MOVE_PLAYER_POS_ROT,
        &encode_move_player_pos_rot(body.feet_pos, 0.0, 90.0, body.on_ground),
    )
    .await?;

    let hit = raycast_blocks(world, body.eye_pos(), [0.0, -1.0, 0.0], 5.0)
        .context("DDA raycast downward missed floor")?;

    let target_break = (6, -59, 6);
    conn.send_packet(
        c2s::PLAYER_ACTION,
        &encode_start_destroy_block(target_break, 1, 1),
    )
    .await?;
    conn.send_packet(
        c2s::PLAYER_ACTION,
        &encode_stop_destroy_block(target_break, 1, 2),
    )
    .await?;

    let mut saw_break = false;
    for _ in 0..1024 {
        if let PlayServerEvent::BlockUpdate { pos, state_id } = read_next_play_event(conn).await?
            && pos == target_break
        {
            world.set_block(pos.0, pos.1, pos.2, state_id);
            assert_eq!(state_id, ground_world::generated_blocks::default_state::AIR);
            saw_break = true;
            break;
        }
    }
    if !saw_break {
        bail!("did not receive live BLOCK_UPDATE for broken block");
    }

    conn.send_packet(c2s::SET_CARRIED_ITEM, &encode_set_carried_item(7))
        .await?;
    let place_base = (7, -59, 7);
    conn.send_packet(c2s::USE_ITEM_ON, &encode_use_item_on(place_base, 1, 2))
        .await?;

    let mut placed_state = None;
    for _ in 0..1024 {
        if let PlayServerEvent::BlockUpdate { pos, state_id } = read_next_play_event(conn).await?
            && pos == (7, -58, 7)
        {
            world.set_block(pos.0, pos.1, pos.2, state_id);
            placed_state = Some(state_id);
            break;
        }
    }
    if placed_state != Some(ground_world::generated_blocks::default_state::GLASS) {
        bail!("expected placed GLASS state, got {placed_state:?}");
    }

    conn.send_packet(c2s::CHAT_COMMAND, &encode_chat_command("save"))
        .await?;
    let mut save_msg = String::new();
    for _ in 0..1024 {
        if let PlayServerEvent::SystemChat { text, .. } = read_next_play_event(conn).await?
            && text.contains("Anvil .mca")
        {
            save_msg = text;
            break;
        }
    }
    if save_msg.is_empty() {
        bail!("did not receive /save Anvil confirmation message");
    }

    println!(
        "Gameplay Smoke OK | LandedFeetY: {:.2} | RaycastHit: {:?} | Broke: {:?} | PlacedGlassAt: (7, -58, 7) | ServerSave: {}",
        body.feet_pos[1], hit.block_pos, target_break, save_msg
    );
    Ok(())
}

#[cfg(feature = "dev-offline")]
async fn run_multiplayer_smoke(host: &str, port: u16) -> Result<()> {
    use generated_packets::play::serverbound as c2s;
    use ground_protocol::config_play::ENTITY_TYPE_PLAYER;
    use ground_render::build_entities_and_outline_mesh;
    use ground_ui::{HudModel, build_full_ui_vertices};

    let (mut conn_a, fin_a) =
        ground_protocol::client::login_dev_offline(host, port, "SolaraAlpha").await?;
    complete_configuration(&mut conn_a, 4).await?;
    let boot_a = bootstrap_play_session(&mut conn_a).await?;

    // Verify starter Overworld mobs (Pig, Cow, Sheep, Zombie) arrived on join!
    let starter_mobs = boot_a
        .initial_entities
        .iter()
        .filter(|e| e.entity_type != ENTITY_TYPE_PLAYER)
        .count();
    if starter_mobs < 4 {
        bail!("expected >= 4 starter Overworld mobs on join, got {starter_mobs}");
    }

    let initial_raw = boot_a.raw_chunk_packets.clone();
    let decoded_initial = tokio::task::spawn_blocking(move || {
        decode_and_mesh_chunks_parallel(&initial_raw, OVERWORLD_SECTIONS)
    })
    .await
    .context("join multi-core initial chunk decode/mesh")??;

    let mut initial_sections = Vec::new();
    let mut initial_quads = 0usize;
    for col in &decoded_initial {
        for (sec_idx, m) in col.meshes.iter().enumerate() {
            if !m.is_empty() {
                initial_quads += m.quad_count();
                initial_sections.push(WorldSectionMesh {
                    origin: Vec3::new(
                        (col.pos.x * 16) as f32,
                        (OVERWORLD_MIN_Y + (sec_idx as i32) * 16) as f32,
                        (col.pos.z * 16) as f32,
                    ),
                    mesh: m.clone(),
                });
            }
        }
    }
    let mega = pack_mega_mesh_batch(&initial_sections);

    let (mut conn_b, _fin_b) =
        ground_protocol::client::login_dev_offline(host, port, "SolaraBravo").await?;
    complete_configuration(&mut conn_b, 4).await?;
    let boot_b = bootstrap_play_session(&mut conn_b).await?;

    let b_saw_a_tab = boot_b
        .initial_players
        .iter()
        .any(|p| p.uuid == fin_a.uuid && p.username.as_deref() == Some("SolaraAlpha"));
    let b_saw_a_entity = boot_b
        .initial_entities
        .iter()
        .any(|e| e.entity_id == boot_a.join_info.entity_id && e.entity_type == ENTITY_TYPE_PLAYER);
    if !b_saw_a_tab || !b_saw_a_entity {
        bail!("Client B did not receive Client A's tab/entity on join");
    }

    let mut a_saw_b_entity = None;
    for _ in 0..1024 {
        if let PlayServerEvent::AddEntity(ent) = read_next_play_event(&mut conn_a).await?
            && ent.entity_id == boot_b.join_info.entity_id
        {
            a_saw_b_entity = Some(ent);
            break;
        }
    }
    if a_saw_b_entity.is_none() {
        bail!("Client A did not receive live spawn for Client B");
    }

    // Verify Tab command auto-completion (/sp -> spawnmob)
    conn_a
        .send_packet(
            c2s::COMMAND_SUGGESTION,
            &encode_request_command_suggestions(42, "/sp"),
        )
        .await?;
    let mut saw_spawnmob_tab = false;
    for _ in 0..1024 {
        if let PlayServerEvent::CommandSuggestions { tx_id, matches, .. } =
            read_next_play_event(&mut conn_a).await?
            && tx_id == 42
            && matches.iter().any(|m| m == "spawnmob")
        {
            saw_spawnmob_tab = true;
            break;
        }
    }
    if !saw_spawnmob_tab {
        bail!("did not receive Tab auto-completion for /sp -> spawnmob");
    }

    // Switch Client B to Survival mode, then Client A melee-attacks Client B (c2s::ATTACK)!
    conn_b
        .send_packet(c2s::CHAT_COMMAND, &encode_chat_command("gamemode survival"))
        .await?;
    for _ in 0..1024 {
        if let PlayServerEvent::SystemChat { text, .. } = read_next_play_event(&mut conn_b).await?
            && text.contains("Survival")
        {
            break;
        }
    }

    conn_a
        .send_packet(
            c2s::ATTACK,
            &encode_attack_entity(boot_b.join_info.entity_id),
        )
        .await?;

    let mut b_health_after_hit = None;
    for _ in 0..1024 {
        if let PlayServerEvent::SetHealth { health, .. } = read_next_play_event(&mut conn_b).await?
            && health < 20.0
        {
            b_health_after_hit = Some(health);
            break;
        }
    }
    if b_health_after_hit != Some(16.0) {
        bail!("expected Client B PvP health = 16.0 after melee hit, got {b_health_after_hit:?}");
    }

    // Verify /kill + PERFORM_RESPAWN restores Client B to 20.0 HP!
    conn_b
        .send_packet(c2s::CHAT_COMMAND, &encode_chat_command("kill"))
        .await?;
    conn_b
        .send_packet(c2s::CLIENT_COMMAND, &encode_perform_respawn())
        .await?;
    let mut respawned_full_hp = false;
    for _ in 0..1024 {
        if let PlayServerEvent::SetHealth { health, .. } = read_next_play_event(&mut conn_b).await?
            && (health - 20.0).abs() < 1e-4
        {
            respawned_full_hp = true;
            break;
        }
    }
    if !respawned_full_hp {
        bail!("expected Client B respawn to restore 20.0 HP");
    }

    // Phase 7 Playable Survival Checks:
    // 1. Lua plugin custom command (/rules)
    conn_a
        .send_packet(c2s::CHAT_COMMAND, &encode_chat_command("rules"))
        .await?;
    let mut saw_lua_rules = false;
    for _ in 0..1024 {
        if let PlayServerEvent::SystemChat { text, .. } = read_next_play_event(&mut conn_a).await?
            && text.contains("Solara-Sky Rules")
        {
            saw_lua_rules = true;
            break;
        }
    }
    if !saw_lua_rules {
        bail!("did not receive response from auto-loaded Lua plugin /rules command");
    }

    // 2. Survival Crafting (/craft crafting_table) + live SET_PLAYER_INVENTORY sync
    conn_a
        .send_packet(
            c2s::CHAT_COMMAND,
            &encode_chat_command("craft crafting_table"),
        )
        .await?;
    let mut saw_crafted_inv_slot = false;
    for _ in 0..1024 {
        if let PlayServerEvent::SetPlayerInventory { item_id, count, .. } =
            read_next_play_event(&mut conn_a).await?
            && item_id == ground_data::generated_items::id::CRAFTING_TABLE
            && count >= 1
        {
            saw_crafted_inv_slot = true;
            break;
        }
    }
    if !saw_crafted_inv_slot {
        bail!("did not receive SET_PLAYER_INVENTORY for crafted crafting_table");
    }

    // 3. Overworld Day/Night MC 26.3 WorldClock sync (/time sunset -> 12500 ticks)
    conn_a
        .send_packet(c2s::CHAT_COMMAND, &encode_chat_command("time sunset"))
        .await?;
    let mut saw_sunset_clock = false;
    for _ in 0..1024 {
        if let PlayServerEvent::SetTime { day_time, .. } = read_next_play_event(&mut conn_a).await?
            && (12_500..12_600).contains(&day_time)
        {
            saw_sunset_clock = true;
            break;
        }
    }
    if !saw_sunset_clock {
        bail!("did not receive MC 26.3 SET_TIME WorldClock update for /time sunset");
    }

    // 4. Survival Block Break -> Floating 3D ITEM drop + Auto-Pickup (TAKE_ITEM_ENTITY)
    conn_b
        .send_packet(
            c2s::PLAYER_ACTION,
            &encode_start_destroy_block((8, -59, 7), 1, 50),
        )
        .await?;
    conn_b
        .send_packet(
            c2s::PLAYER_ACTION,
            &encode_stop_destroy_block((8, -59, 7), 1, 51),
        )
        .await?;
    let mut saw_item_pickup = false;
    for _ in 0..1024 {
        if let PlayServerEvent::TakeItemEntity { amount, .. } =
            read_next_play_event(&mut conn_b).await?
            && amount >= 1
        {
            saw_item_pickup = true;
            break;
        }
    }
    if !saw_item_pickup {
        bail!("expected Client B to auto-pickup dropped block item (TAKE_ITEM_ENTITY)");
    }

    // Exercise 3D entities + block outline + full pixel-font Survival & Inventory UI
    let (_everts, eidx) = build_entities_and_outline_mesh(
        &[RemoteEntityVisual {
            entity_id: boot_b.join_info.entity_id,
            entity_type: ENTITY_TYPE_PLAYER as u32,
            feet_pos: Vec3::new(8.0, -58.0, 8.0),
            yaw_deg: 90.0,
        }],
        Some([8, -59, 8]),
    );
    let ui_verts = build_full_ui_vertices(&HudModel {
        width: 1280,
        height: 720,
        selected_slot: 0,
        health: 16.0,
        food: 20,
        day_time_ticks: 12_500,
        inventory_open: true,
        chat_open: true,
        chat_input: "/craft wooden_sword".into(),
        fps: 300,
        frame_ms: 3.3,
        camera_pos: [8.0, -56.0, 8.0],
        chunk_count: 81,
        entity_count: starter_mobs + 1,
        chat_lines: vec!["Playable Phase 7 verified!".into()],
        ..Default::default()
    });

    println!(
        "Phase 7 Playable Survival Smoke OK | Chunks: {} (GreedyQuads: {}, MegaVboKB: {}) | StarterMobs: {} | LuaPlugin(/rules): true | Crafting(crafting_table): true | WorldClock(sunset): true | ItemDropPickup: true | PvP+Respawn: true | EntityOutlineTris: {} | PlayableUiVerts: {}",
        boot_a.raw_chunk_packets.len(),
        initial_quads,
        (mega.vertices.len() * 8) / 1024,
        starter_mobs,
        eidx.len() / 3,
        ui_verts.len()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    if let Some(target) = &cli.ping {
        let (host, port) = parse_host_port(target)?;
        info!(%host, port, "pinging Minecraft server");
        let (status, rtt) = ping_server(host, port).await?;
        println!(
            "Server: {} (protocol {}) | Players: {}/{} | RTT: {:.2} ms | MOTD: {}",
            status.version.name,
            status.version.protocol,
            status.players.online,
            status.players.max,
            rtt.as_secs_f64() * 1000.0,
            status.description.as_plain_text()
        );
        return Ok(());
    }

    let assets = LocalMojangAssets::open(&cli.assets_dir, None)?;
    let atlas = assets.build_block_atlas();
    info!(
        mc_version = ground_protocol::MINECRAFT_VERSION,
        protocol = ground_protocol::PROTOCOL_VERSION,
        worker_threads = ground_world::worker_threads(),
        assets_dir = %assets.root().display(),
        mojang_png_tiles = atlas.loaded_mojang_pngs,
        total_blocks = ground_world::generated_blocks::TOTAL_BLOCKS,
        total_items = ground_data::generated_items::TOTAL_ITEMS,
        total_entities = ground_data::generated_entities::TOTAL_ENTITY_TYPES,
        "Solara-Ground initialized"
    );

    if let Some(target) = &cli.connect {
        let (host, port) = parse_host_port(target)?;

        #[cfg(feature = "dev-offline")]
        if cli.multiplayer_smoke {
            return run_multiplayer_smoke(host, port).await;
        }

        let (mut conn, finished) = match (&cli.username, cli.uuid, &cli.access_token) {
            (Some(username), Some(uuid), Some(token)) => {
                let session = SessionConfig {
                    username: username.clone(),
                    uuid,
                    access_token: RedactedToken::new(token.clone()),
                    assets_dir: cli.assets_dir.clone(),
                };
                info!(?session, %host, port, "connecting in online mode");
                login_online(
                    host,
                    port,
                    &session.username,
                    session.uuid,
                    session.access_token.expose_secret(),
                )
                .await?
            }
            #[cfg(feature = "dev-offline")]
            _ => {
                let dev_name = cli.username.as_deref().unwrap_or("SolaraDev");
                info!(username = %dev_name, %host, port, "connecting with dev-offline test feature");
                ground_protocol::client::login_dev_offline(host, port, dev_name).await?
            }
            #[cfg(not(feature = "dev-offline"))]
            _ => {
                bail!(
                    "Online-mode credentials (--username, --uuid, --access-token) are required to connect \
                     (supplied automatically by Solara Launcher)."
                );
            }
        };

        info!(
            username = %finished.username,
            uuid = %finished.uuid,
            "Login state succeeded, entering Configuration state"
        );
        let cfg_summary = complete_configuration(&mut conn, 6).await?;
        info!(
            registries = cfg_summary.registries_received,
            entries = cfg_summary.registry_entries_received,
            tag_groups = cfg_summary.tag_groups_received,
            "Configuration state complete, bootstrapping Play world"
        );

        let bootstrap = bootstrap_play_session(&mut conn).await?;

        let raw_packets = bootstrap.raw_chunk_packets;
        let decoded_columns = tokio::task::spawn_blocking(move || {
            decode_and_mesh_chunks_parallel(&raw_packets, OVERWORLD_SECTIONS)
        })
        .await
        .context("join multi-core chunk decode & greedy mesh")??;

        let mut world_map = ClientWorldMap::new();
        let mut chunk_meshes: HashMap<ChunkPos, Vec<WorldSectionMesh>> = HashMap::new();
        let mut world_sections = Vec::new();
        let mut total_quads = 0usize;
        for col in decoded_columns {
            let mut col_ws = Vec::new();
            for (sec_idx, mesh) in col.meshes.into_iter().enumerate() {
                if !mesh.is_empty() {
                    total_quads += mesh.quad_count();
                    let world_y = OVERWORLD_MIN_Y + (sec_idx as i32) * 16;
                    let wsm = WorldSectionMesh {
                        origin: Vec3::new(
                            (col.pos.x * 16) as f32,
                            world_y as f32,
                            (col.pos.z * 16) as f32,
                        ),
                        mesh,
                    };
                    col_ws.push(wsm.clone());
                    world_sections.push(wsm);
                }
            }
            chunk_meshes.insert(col.pos, col_ws);
            world_map.insert_chunk(col.pos, col.sections);
        }

        let initial_entities: Vec<RemoteEntityVisual> = bootstrap
            .initial_entities
            .iter()
            .map(|e| RemoteEntityVisual {
                entity_id: e.entity_id,
                entity_type: e.entity_type.max(0) as u32,
                feet_pos: Vec3::new(
                    e.position[0] as f32,
                    e.position[1] as f32,
                    e.position[2] as f32,
                ),
                yaw_deg: e.yaw_deg,
            })
            .collect();

        if cli.gameplay_smoke {
            return run_gameplay_smoke_session(&mut conn, &mut world_map, bootstrap.spawn.position)
                .await;
        }

        let cam = VoxelCamera {
            position: Vec3::new(
                bootstrap.spawn.position[0] as f32,
                bootstrap.spawn.position[1] as f32 + 1.62,
                bootstrap.spawn.position[2] as f32,
            ),
            yaw_rad: bootstrap.spawn.yaw_deg.to_radians(),
            pitch_rad: -0.20,
            ..Default::default()
        };

        let chunk_count = world_map.chunk_count();
        if cli.headless_smoke {
            let frustum = Frustum::from_view_proj(cam.view_proj_matrix());
            let mega = pack_mega_mesh_batch(&world_sections);
            let mut indirect_cmds = Vec::new();
            build_frustum_culled_indirect_cmds(&mega.slots, &frustum, &mut indirect_cmds);
            println!(
                "Play Smoke OK | Player: {} | Chunks: {} | MegaSlots: {} | IndirectDraws: {} | GreedyQuads: {} | Entities: {}",
                finished.username,
                chunk_count,
                mega.slots.len(),
                indirect_cmds.len(),
                total_quads,
                initial_entities.len()
            );
            return Ok(());
        }

        let bridge = spawn_live_network_bridge(
            conn,
            world_map,
            chunk_meshes,
            initial_entities.clone(),
            bootstrap.spawn.position,
            cli.assets_dir.clone(),
        );
        return run_voxel_window(WindowLaunchConfig {
            title: format!(
                "Solara-Ground {} — {}",
                ground_protocol::MINECRAFT_VERSION,
                target
            ),
            atlas,
            camera: cam,
            sections: world_sections,
            entities: initial_entities,
            initial_health: bootstrap.initial_health,
            initial_day_time: bootstrap.initial_day_time,
            initial_inventory: bootstrap.initial_inventory,
            initial_survival: bootstrap.join_info.game_mode == 0,
            initial_chunks: chunk_count,
            bridge: Some(bridge),
        });
    }

    bail!(
        "Usage: `solara-ground --ping <host:port>` or launch via Solara Launcher with `--connect <host:port>`."
    )
}
