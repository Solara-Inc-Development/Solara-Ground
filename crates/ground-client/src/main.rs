pub mod session;

use crate::session::{RedactedToken, SessionConfig};
use anyhow::{Context, Result, bail};
use clap::Parser;
use glam::Vec3;
use ground_protocol::client::{ClientConnection, login_online, ping_server};
use ground_protocol::config_play::{
    PlayServerEvent, bootstrap_play_session, complete_configuration, encode_chat_command,
    encode_move_player_pos_rot, encode_set_carried_item, encode_start_destroy_block,
    encode_use_item_on, read_next_play_event,
};
use ground_protocol::generated_packets;
use ground_render::{Frustum, VoxelCamera, WorldSectionMesh, run_voxel_window};
use ground_world::assets::LocalMojangAssets;
use ground_world::mesh::mesh_sections_parallel;
use ground_world::physics::{ClientWorldMap, PlayerBody, raycast_blocks};
use ground_world::{OVERWORLD_MIN_Y, OVERWORLD_SECTIONS, decode_level_chunk_with_light};
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

    /// Run protocol + chunk decoding + meshing + frustum culling without opening an OS window
    #[arg(long)]
    headless_smoke: bool,

    /// Run Phase 3 gameplay smoke (AABB physics, DDA raycast, block break/place, /save Anvil command)
    #[arg(long)]
    gameplay_smoke: bool,

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
    for _ in 0..16 {
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
    for _ in 0..16 {
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
    for _ in 0..16 {
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
    info!(
        mc_version = ground_protocol::MINECRAFT_VERSION,
        protocol = ground_protocol::PROTOCOL_VERSION,
        assets_dir = %assets.root().display(),
        total_blocks = ground_world::generated_blocks::TOTAL_BLOCKS,
        total_block_states = ground_world::generated_blocks::TOTAL_BLOCK_STATES,
        "Solara-Ground initialized"
    );

    if let Some(target) = &cli.connect {
        let (host, port) = parse_host_port(target)?;

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

        let mut world_map = ClientWorldMap::new();
        let mut world_sections = Vec::new();
        let mut total_quads = 0usize;
        for raw_pkt in &bootstrap.raw_chunk_packets {
            let (cpos, sections) = decode_level_chunk_with_light(raw_pkt, OVERWORLD_SECTIONS)?;
            let meshes = mesh_sections_parallel(&sections);
            for (sec_idx, mesh) in meshes.into_iter().enumerate() {
                if !mesh.is_empty() {
                    total_quads += mesh.quad_count();
                    let world_y = OVERWORLD_MIN_Y + (sec_idx as i32) * 16;
                    world_sections.push(WorldSectionMesh {
                        origin: Vec3::new(
                            (cpos.x * 16) as f32,
                            world_y as f32,
                            (cpos.z * 16) as f32,
                        ),
                        mesh,
                    });
                }
            }
            world_map.insert_chunk(cpos, sections);
        }

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
            let persisted_glass = world_map.get_block(7, -58, 7)
                == ground_world::generated_blocks::default_state::GLASS;
            println!(
                "Play Smoke OK | Player: {} | Chunks: {} | MeshedSections: {} | VisibleInFrustum: {} | Quads: {} | PersistedGlassAt(7,-58,7): {}",
                finished.username,
                world_map.chunk_count(),
                world_sections.len(),
                visible,
                total_quads,
                persisted_glass
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
            cam,
            world_sections,
        );
    }

    bail!(
        "Usage: `solara-ground --ping <host:port>` or launch via Solara Launcher with `--connect <host:port>`."
    )
}
