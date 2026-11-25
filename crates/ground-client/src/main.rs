pub mod session;

use crate::session::{RedactedToken, SessionConfig};
use anyhow::{Context, Result, bail};
use clap::Parser;
use glam::Vec3;
use ground_protocol::client::{login_online, ping_server};
use ground_protocol::config_play::{bootstrap_play_session, complete_configuration};
use ground_render::{Frustum, VoxelCamera, WorldSectionMesh, run_voxel_window};
use ground_world::assets::LocalMojangAssets;
use ground_world::mesh::mesh_sections_parallel;
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

        // Decode all incoming MC 26.3 chunks and mesh non-empty sections in parallel on rayon
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
                "Play Smoke OK | Player: {} | Entity: {} | Dim: {} | Chunks: {} | MeshedSections: {} | VisibleInFrustum: {} | Quads: {}",
                finished.username,
                bootstrap.join_info.entity_id,
                bootstrap.join_info.dimension_name,
                bootstrap.raw_chunk_packets.len(),
                world_sections.len(),
                visible,
                total_quads
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
