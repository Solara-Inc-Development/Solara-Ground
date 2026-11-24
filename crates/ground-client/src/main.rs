pub mod session;

use crate::session::{RedactedToken, SessionConfig};
use anyhow::{Context, Result, bail};
use clap::Parser;
use ground_protocol::client::ping_server;
use std::path::PathBuf;
use tracing::info;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(
    name = "solara-ground",
    about = "High-performance Rust Minecraft: Java Edition 26.3 client (Windows & Linux)"
)]
struct Cli {
    /// Ping a Minecraft 26.3 server (e.g. `127.0.0.1:25565` or `127.0.0.1`) and exit
    #[arg(long)]
    ping: Option<String>,

    /// Authenticated player username (provided by solara-launcher in online mode)
    #[arg(long)]
    username: Option<String>,

    /// Authenticated player UUID (provided by solara-launcher in online mode)
    #[arg(long)]
    uuid: Option<Uuid>,

    /// Authenticated Minecraft access token (provided by solara-launcher; never logged)
    #[arg(long)]
    access_token: Option<String>,

    /// Path to Minecraft assets directory
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

    info!(
        mc_version = ground_protocol::MINECRAFT_VERSION,
        protocol = ground_protocol::PROTOCOL_VERSION,
        total_blocks = ground_world::generated_blocks::TOTAL_BLOCKS,
        total_block_states = ground_world::generated_blocks::TOTAL_BLOCK_STATES,
        vertex_stride = ground_render::packed_voxel_vertex_layout().array_stride,
        "Solara-Ground initialized"
    );

    match (cli.username, cli.uuid, cli.access_token) {
        (Some(username), Some(uuid), Some(token)) => {
            let session = SessionConfig {
                username,
                uuid,
                access_token: RedactedToken::new(token),
                assets_dir: cli.assets_dir,
            };
            info!(
                ?session,
                "verified online-mode session parameters (window loop arrives in Phase 2)"
            );
            Ok(())
        }
        _ => {
            bail!(
                "Solara-Ground requires an authenticated online-mode session (to be launched via Solara Launcher). \
                 Tip: run `solara-ground --ping <host:port>` to test Server List Ping right now!"
            );
        }
    }
}
