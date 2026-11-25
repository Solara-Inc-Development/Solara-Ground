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
use ground_render::{Frustum, RemotePlayerVisual, VoxelCamera, WorldSectionMesh, run_voxel_window};
use ground_world::assets::LocalMojangAssets;
use ground_world::physics::{ClientWorldMap, PlayerBody, raycast_blocks};
use ground_world::{OVERWORLD_MIN_Y, OVERWORLD_SECTIONS, decode_and_mesh_chunks_parallel};
use std::path::PathBuf;
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

    /// Run protocol + chunk decoding + greedy meshing + frustum culling without opening an OS window
    #[arg(long)]
    headless_smoke: bool,

    /// Run Phase 3 gameplay smoke (AABB physics, DDA raycast, block break/place, /save Anvil command)
    #[arg(long)]
    gameplay_smoke: bool,

    /// Run Phase 4 multiplayer + dynamic procedural chunk streaming + entity sync E2E test
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

async fn run_gameplay_smoke_session(
    conn: &mut ClientConnection,
    world: &mut ClientWorldMap,
    spawn_pos: [f64; 3],
) -> Result<()> {
    use generated_packets::play::serverbound as c2s;

    // 1. Run player gravity + AABB collision until landing on the spawn plaza
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

    // 2. Raycast straight down from eye position to hit the plaza block beneath feet
    let hit = raycast_blocks(world, body.eye_pos(), [0.0, -1.0, 0.0], 5.0)
        .context("DDA raycast downward missed floor")?;

    // Break a nearby block at (6, -59, 6) and wait for live BLOCK_UPDATE -> AIR
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

    // 3. Select hotbar slot 7 (GLASS) and place it on top of hit.block_pos -> adjacent_pos
    conn.send_packet(c2s::SET_CARRIED_ITEM, &encode_set_carried_item(7))
        .await?;
    let place_base = (7, -59, 7);
    conn.send_packet(
        c2s::USE_ITEM_ON,
        &encode_use_item_on(place_base, 1, 2), // Direction::Up (1) -> places at (7, -58, 7)
    )
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

    // 4. Trigger `/save` command to persist modified chunks to Anvil .mca region files!
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

    // 1. Connect Client A ("SolaraAlpha")
    let (mut conn_a, fin_a) =
        ground_protocol::client::login_dev_offline(host, port, "SolaraAlpha").await?;
    complete_configuration(&mut conn_a, 4).await?;
    let boot_a = bootstrap_play_session(&mut conn_a).await?;

    // Multi-core decode + 2D greedy-mesh Client A's initial procedural terrain chunks across all CPU cores!
    let initial_raw = boot_a.raw_chunk_packets.clone();
    let decoded_initial = tokio::task::spawn_blocking(move || {
        decode_and_mesh_chunks_parallel(&initial_raw, OVERWORLD_SECTIONS)
    })
    .await
    .context("join multi-core initial chunk decode/mesh")??;
    let initial_quads: usize = decoded_initial
        .iter()
        .flat_map(|col| &col.meshes)
        .map(|m| m.quad_count())
        .sum();

    // 2. Connect Client B ("SolaraBravo") while Client A is online!
    let (mut conn_b, _fin_b) =
        ground_protocol::client::login_dev_offline(host, port, "SolaraBravo").await?;
    complete_configuration(&mut conn_b, 4).await?;
    let boot_b = bootstrap_play_session(&mut conn_b).await?;

    // Verify Client B saw Client A in its initial tab-list & visible 3D player entities
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

    // Verify Client A receives live PlayerInfoUpdate + AddEntity for Client B
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

    // 3. Client B walks East across chunk boundaries (x = 8 -> 42, chunk 0 -> chunk 2)
    conn_b
        .send_packet(
            c2s::MOVE_PLAYER_POS_ROT,
            &encode_move_player_pos_rot([42.5, -48.0, 8.5], 90.0, 0.0, true),
        )
        .await?;

    // Verify Client B dynamically receives newly streamed procedural chunks from Solara-Sky
    // and processes them in a multi-core rayon batch off the async network thread!
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

    // Verify Client A receives live TeleportEntity tracking Client B's movement to x = 42.5!
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

    // Exercise 3D remote player mesh & 2D HUD builder
    let (p_verts, p_idx) = build_remote_players_mesh(&[RemotePlayerVisual {
        entity_id: boot_b.join_info.entity_id,
        feet_pos: Vec3::new(tracked_x as f32, -48.0, 8.5),
        yaw_deg: 90.0,
    }]);
    let hud_verts = build_hud_vertices(1280, 720, 0);

    println!(
        "Multiplayer Smoke OK | InitialChunks: {} (GreedyQuads: {}) | DynamicStreamedChunks: {} (Quads: {}) | RemoteTrackedX: {:.1} | PlayerMeshVerts: {} (Tris: {}) | HudVerts: {}",
        boot_a.raw_chunk_packets.len(),
        initial_quads,
        streamed_new_chunks,
        streamed_quads,
        tracked_x,
        p_verts.len(),
        p_idx.len() / 3,
        hud_verts.len()
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
        let mut world_sections = Vec::new();
        let mut total_quads = 0usize;
        for col in decoded_columns {
            for (sec_idx, mesh) in col.meshes.into_iter().enumerate() {
                if !mesh.is_empty() {
                    total_quads += mesh.quad_count();
                    let world_y = OVERWORLD_MIN_Y + (sec_idx as i32) * 16;
                    world_sections.push(WorldSectionMesh {
                        origin: Vec3::new(
                            (col.pos.x * 16) as f32,
                            world_y as f32,
                            (col.pos.z * 16) as f32,
                        ),
                        mesh,
                    });
                }
            }
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
            let visible = world_sections
                .iter()
                .filter(|s| frustum.intersects_aabb(s.origin, s.origin + Vec3::splat(16.0)))
                .count();
            println!(
                "Play Smoke OK | Player: {} | Chunks: {} | MeshedSections: {} | VisibleInFrustum: {} | GreedyQuads: {} | RemotePlayers: {}",
                finished.username,
                world_map.chunk_count(),
                world_sections.len(),
                visible,
                total_quads,
                remote_players.len()
            );
            return Ok(());
        }

        drop(conn);
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
        );
    }

    bail!(
        "Usage: `solara-ground --ping <host:port>` or launch via Solara Launcher with `--connect <host:port>`."
    )
}
