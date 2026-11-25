pub mod session;

use crate::session::{RedactedToken, SessionConfig};
use anyhow::{Context, Result, bail};
use clap::Parser;
use glam::Vec3;
use ground_protocol::client::{ClientConnection, login_online, ping_server};
use ground_protocol::config_play::{
    ENTITY_TYPE_PLAYER, PlayServerEvent, bootstrap_play_session, complete_configuration,
    encode_chat_command, encode_move_player_pos_rot, encode_set_carried_item,
    encode_start_destroy_block, encode_use_item_on, read_next_play_event,
};
use ground_protocol::generated_packets;
use ground_render::{
    ClientWindowAction, Frustum, LiveRenderBridge, RemotePlayerVisual, RenderSceneUpdate,
    VoxelCamera, WorldSectionMesh, build_frustum_culled_indirect_cmds, pack_mega_mesh_batch,
    run_voxel_window,
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

    /// Run Phase 4+5 multiplayer, dynamic chunk streaming, Folia /tps & /plugins, and Mega-Buffer E2E test
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

fn spawn_live_network_bridge(
    mut conn: ClientConnection,
    mut world_map: ClientWorldMap,
    mut chunk_meshes: HashMap<ChunkPos, Vec<WorldSectionMesh>>,
    initial_players: Vec<RemotePlayerVisual>,
) -> LiveRenderBridge {
    use generated_packets::play::serverbound as c2s;

    let (scene_tx, scene_rx) = mpsc::channel::<RenderSceneUpdate>();
    let (action_tx, action_rx) = mpsc::channel::<ClientWindowAction>();

    let mut remote_players: HashMap<i32, RemotePlayerVisual> = initial_players
        .into_iter()
        .map(|p| (p.entity_id, p))
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
            let mut next_seq = 1i32;
            let mut drain_tick = tokio::time::interval(std::time::Duration::from_millis(16));
            drain_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = drain_tick.tick() => {
                        let mut disconnected = false;
                        loop {
                            match action_rx.try_recv() {
                                Ok(ClientWindowAction::Move { feet_pos, yaw_deg, pitch_deg, on_ground }) => {
                                    let _ = conn.send_packet(
                                        c2s::MOVE_PLAYER_POS_ROT,
                                        &encode_move_player_pos_rot(feet_pos, yaw_deg, pitch_deg, on_ground),
                                    ).await;
                                }
                                Ok(ClientWindowAction::SelectHotbarSlot(slot)) => {
                                    let _ = conn.send_packet(
                                        c2s::SET_CARRIED_ITEM,
                                        &encode_set_carried_item(slot),
                                    ).await;
                                }
                                Ok(ClientWindowAction::BreakLook { eye_pos, look_dir }) => {
                                    if let Some(hit) = raycast_blocks(&world_map, eye_pos, look_dir, 6.0) {
                                        let seq = next_seq;
                                        next_seq += 1;
                                        let _ = conn.send_packet(
                                            c2s::PLAYER_ACTION,
                                            &encode_start_destroy_block(hit.block_pos, hit.face_direction as u8, seq),
                                        ).await;
                                    }
                                }
                                Ok(ClientWindowAction::PlaceLook { eye_pos, look_dir }) => {
                                    if let Some(hit) = raycast_blocks(&world_map, eye_pos, look_dir, 6.0) {
                                        let seq = next_seq;
                                        next_seq += 1;
                                        let _ = conn.send_packet(
                                            c2s::USE_ITEM_ON,
                                            &encode_use_item_on(hit.block_pos, hit.face_direction, seq),
                                        ).await;
                                    }
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
                                    if scene_tx.send(RenderSceneUpdate::ReplaceSections(flatten_chunk_meshes(&chunk_meshes))).is_err() {
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
                                    let _ = scene_tx.send(RenderSceneUpdate::ReplaceSections(flatten_chunk_meshes(&chunk_meshes)));
                                }
                            }
                            PlayServerEvent::AddEntity(ent) if ent.entity_type == ENTITY_TYPE_PLAYER => {
                                remote_players.insert(
                                    ent.entity_id,
                                    RemotePlayerVisual {
                                        entity_id: ent.entity_id,
                                        feet_pos: Vec3::new(ent.position[0] as f32, ent.position[1] as f32, ent.position[2] as f32),
                                        yaw_deg: ent.yaw_deg,
                                    },
                                );
                                let _ = scene_tx.send(RenderSceneUpdate::RemotePlayers(remote_players.values().copied().collect()));
                            }
                            PlayServerEvent::TeleportEntity(tp) => {
                                if let Some(p) = remote_players.get_mut(&tp.entity_id) {
                                    p.feet_pos = Vec3::new(tp.position[0] as f32, tp.position[1] as f32, tp.position[2] as f32);
                                    p.yaw_deg = tp.yaw_deg;
                                    let _ = scene_tx.send(RenderSceneUpdate::RemotePlayers(remote_players.values().copied().collect()));
                                }
                            }
                            PlayServerEvent::RemoveEntities(ids) => {
                                let mut changed = false;
                                for id in ids {
                                    changed |= remote_players.remove(&id).is_some();
                                }
                                if changed {
                                    let _ = scene_tx.send(RenderSceneUpdate::RemotePlayers(remote_players.values().copied().collect()));
                                }
                            }
                            PlayServerEvent::Teleport(tp) => {
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

    let mut saw_break = false;
    for _ in 0..24 {
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
    for _ in 0..24 {
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
    for _ in 0..24 {
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
    use ground_render::{build_hud_vertices, build_remote_players_mesh};

    let (mut conn_a, fin_a) =
        ground_protocol::client::login_dev_offline(host, port, "SolaraAlpha").await?;
    complete_configuration(&mut conn_a, 4).await?;
    let boot_a = bootstrap_play_session(&mut conn_a).await?;

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
        bail!(
            "Client B did not receive Client A's tab/entity on join (tab={b_saw_a_tab}, entity={b_saw_a_entity})"
        );
    }

    let mut a_saw_b_tab = false;
    let mut a_saw_b_entity = None;
    for _ in 0..24 {
        match read_next_play_event(&mut conn_a).await? {
            PlayServerEvent::PlayerInfoUpdate(list) => {
                if list
                    .iter()
                    .any(|p| p.username.as_deref() == Some("SolaraBravo"))
                {
                    a_saw_b_tab = true;
                }
            }
            PlayServerEvent::AddEntity(ent)
                if ent.entity_id == boot_b.join_info.entity_id
                    && ent.entity_type == ENTITY_TYPE_PLAYER =>
            {
                a_saw_b_entity = Some(ent);
            }
            _ => {}
        }
        if a_saw_b_tab && a_saw_b_entity.is_some() {
            break;
        }
    }
    if !a_saw_b_tab || a_saw_b_entity.is_none() {
        bail!(
            "Client A did not receive live spawn for Client B (tab={a_saw_b_tab}, entity={a_saw_b_entity:?})"
        );
    }

    conn_b
        .send_packet(
            c2s::MOVE_PLAYER_POS_ROT,
            &encode_move_player_pos_rot([42.5, -48.0, 8.5], 90.0, 0.0, true),
        )
        .await?;

    let mut streamed_raw_packets = Vec::new();
    for _ in 0..32 {
        let ev = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            read_next_play_event(&mut conn_b),
        )
        .await;
        match ev {
            Ok(Ok(PlayServerEvent::LevelChunkWithLight(raw))) => {
                streamed_raw_packets.push(raw);
                if streamed_raw_packets.len() >= 9 {
                    break;
                }
            }
            Ok(Ok(_)) => {}
            _ => break,
        }
    }
    if streamed_raw_packets.is_empty() {
        bail!("expected dynamic procedural chunks streamed upon crossing chunk boundary");
    }
    let streamed_new_chunks = streamed_raw_packets.len();
    let streamed_cols = tokio::task::spawn_blocking(move || {
        decode_and_mesh_chunks_parallel(&streamed_raw_packets, OVERWORLD_SECTIONS)
    })
    .await
    .context("join multi-core dynamic chunk decode/mesh")??;
    let streamed_quads: usize = streamed_cols
        .iter()
        .flat_map(|col| &col.meshes)
        .map(|m| m.quad_count())
        .sum();

    let mut tracked_x = 0.0f64;
    for _ in 0..24 {
        if let PlayServerEvent::TeleportEntity(tp) = read_next_play_event(&mut conn_a).await?
            && tp.entity_id == boot_b.join_info.entity_id
        {
            tracked_x = tp.position[0];
            if tracked_x >= 40.0 {
                break;
            }
        }
    }
    if tracked_x < 40.0 {
        bail!("Client A did not receive TeleportEntity movement for Client B (got x={tracked_x})");
    }

    // Verify Phase 5 Folia region TPS profiler (/tps) & SkyPlugin command (/plugins & /solara)
    conn_a
        .send_packet(c2s::CHAT_COMMAND, &encode_chat_command("tps"))
        .await?;
    conn_a
        .send_packet(c2s::CHAT_COMMAND, &encode_chat_command("plugins"))
        .await?;
    conn_a
        .send_packet(c2s::CHAT_COMMAND, &encode_chat_command("solara"))
        .await?;

    let mut tps_line = String::new();
    let mut plugins_line = String::new();
    let mut custom_plugin_line = String::new();
    for _ in 0..32 {
        if let PlayServerEvent::SystemChat { text, .. } = read_next_play_event(&mut conn_a).await? {
            if text.contains("Solara-Sky TPS") {
                tps_line = text;
            } else if text.contains("SolaraCore") {
                plugins_line = text;
            } else if text.contains("Folia-style") {
                custom_plugin_line = text;
            }
        }
        if !tps_line.is_empty() && !plugins_line.is_empty() && !custom_plugin_line.is_empty() {
            break;
        }
    }
    if tps_line.is_empty() || plugins_line.is_empty() || custom_plugin_line.is_empty() {
        bail!(
            "missing Phase 5 command output (tps={tps_line:?}, plugins={plugins_line:?}, solara={custom_plugin_line:?})"
        );
    }

    let (_p_verts, p_idx) = build_remote_players_mesh(&[RemotePlayerVisual {
        entity_id: boot_b.join_info.entity_id,
        feet_pos: Vec3::new(tracked_x as f32, -48.0, 8.5),
        yaw_deg: 90.0,
    }]);
    let hud_verts = build_hud_vertices(1280, 720, 0);

    println!(
        "Multiplayer+Phase5 Smoke OK | Chunks: {} (GreedyQuads: {}, MegaVboKB: {}) | DynamicChunks: {} (Quads: {}) | RemoteX: {:.1} | PlayerTris: {} | HudVerts: {}\n  -> {}\n  -> {}\n  -> {}",
        boot_a.raw_chunk_packets.len(),
        initial_quads,
        (mega.vertices.len() * 8) / 1024,
        streamed_new_chunks,
        streamed_quads,
        tracked_x,
        p_idx.len() / 3,
        hud_verts.len(),
        tps_line,
        plugins_line,
        custom_plugin_line
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
        total_block_states = ground_world::generated_blocks::TOTAL_BLOCK_STATES,
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

        let remote_players: Vec<RemotePlayerVisual> = bootstrap
            .initial_entities
            .iter()
            .filter(|e| e.entity_type == ENTITY_TYPE_PLAYER)
            .map(|e| RemotePlayerVisual {
                entity_id: e.entity_id,
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

        if cli.headless_smoke {
            let frustum = Frustum::from_view_proj(cam.view_proj_matrix());
            let mega = pack_mega_mesh_batch(&world_sections);
            let mut indirect_cmds = Vec::new();
            build_frustum_culled_indirect_cmds(&mega.slots, &frustum, &mut indirect_cmds);
            println!(
                "Play Smoke OK | Player: {} | Chunks: {} | MegaSlots: {} | IndirectDraws: {} | GreedyQuads: {} | MegaVboKB: {}",
                finished.username,
                world_map.chunk_count(),
                mega.slots.len(),
                indirect_cmds.len(),
                total_quads,
                (mega.vertices.len() * 8) / 1024
            );
            return Ok(());
        }

        let bridge =
            spawn_live_network_bridge(conn, world_map, chunk_meshes, remote_players.clone());
        return run_voxel_window(
            format!(
                "Solara-Ground {} — {}",
                ground_protocol::MINECRAFT_VERSION,
                target
            ),
            atlas,
            cam,
            world_sections,
            remote_players,
            Some(bridge),
        );
    }

    bail!(
        "Usage: `solara-ground --ping <host:port>` or launch via Solara Launcher with `--connect <host:port>`."
    )
}
