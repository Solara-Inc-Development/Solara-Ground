//! Client-side Configuration and Play state handlers for Minecraft 26.3 (protocol 777).

use crate::MINECRAFT_VERSION;
use crate::buf::{PacketReader, PacketWriter};
use crate::client::ClientConnection;
use crate::error::{ProtocolError, ProtocolResult};
use crate::generated_packets;
use crate::nbt::read_network_nbt;

#[derive(Debug, Clone, Default)]
pub struct ConfigurationSummary {
    pub registries_received: usize,
    pub registry_entries_received: usize,
    pub tag_groups_received: usize,
}

fn encode_client_information(view_distance: u8) -> Vec<u8> {
    let mut w = PacketWriter::new();
    w.write_string("en_us"); // locale
    w.write_i8(view_distance as i8);
    w.write_varint(0); // chatVisibility = FULL
    w.write_bool(true); // chatColors
    w.write_u8(0x7F); // modelCustomisation = all skin layers
    w.write_varint(1); // mainHand = RIGHT
    w.write_bool(false); // textFilteringEnabled
    w.write_bool(true); // allowsListing
    w.write_varint(0); // particleStatus = ALL
    w.into_inner()
}

fn encode_brand_payload(brand: &str) -> Vec<u8> {
    let mut w = PacketWriter::new();
    w.write_string("minecraft:brand");
    w.write_string(brand);
    w.into_inner()
}

/// Runs the client-side `Configuration` phase until `FinishConfiguration` completes.
pub async fn complete_configuration(
    conn: &mut ClientConnection,
    view_distance: u8,
) -> ProtocolResult<ConfigurationSummary> {
    use generated_packets::configuration::{clientbound as s2c, serverbound as c2s};

    conn.send_packet(c2s::CUSTOM_PAYLOAD, &encode_brand_payload("solara-ground"))
        .await?;
    conn.send_packet(
        c2s::CLIENT_INFORMATION,
        &encode_client_information(view_distance),
    )
    .await?;

    let mut summary = ConfigurationSummary::default();

    loop {
        let (pkt_id, payload) = conn.read_packet().await?;
        match pkt_id {
            s2c::DISCONNECT => {
                let mut r = PacketReader::new(&payload);
                let tag = read_network_nbt(&mut r)?;
                return Err(ProtocolError::ServerDisconnect(format!("{tag:?}")));
            }
            s2c::KEEP_ALIVE => {
                conn.send_packet(c2s::KEEP_ALIVE, &payload).await?;
            }
            s2c::PING => {
                conn.send_packet(c2s::PONG, &payload).await?;
            }
            s2c::SELECT_KNOWN_PACKS => {
                // Advertise vanilla minecraft:core 26.3
                let mut w = PacketWriter::new();
                w.write_varint(1);
                w.write_string("minecraft");
                w.write_string("core");
                w.write_string(MINECRAFT_VERSION);
                conn.send_packet(c2s::SELECT_KNOWN_PACKS, w.as_slice())
                    .await?;
            }
            s2c::REGISTRY_DATA => {
                let mut r = PacketReader::new(&payload);
                let _reg_id = r.read_string(32767)?;
                let count = r.read_usize_bounded(65536)?;
                for _ in 0..count {
                    let _entry_id = r.read_string(32767)?;
                    if r.read_bool()? {
                        let _nbt = read_network_nbt(&mut r)?;
                    }
                }
                summary.registries_received += 1;
                summary.registry_entries_received += count;
            }
            s2c::UPDATE_TAGS => {
                let mut r = PacketReader::new(&payload);
                summary.tag_groups_received = r.read_usize_bounded(1024)?;
            }
            s2c::FINISH_CONFIGURATION => {
                conn.send_packet(c2s::FINISH_CONFIGURATION, &[]).await?;
                return Ok(summary);
            }
            _ => {
                // Custom payloads, feature flags, server links, etc.
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlayJoinInfo {
    pub entity_id: i32,
    pub dimension_name: String,
    pub view_distance: i32,
    pub game_mode: u8,
    pub sea_level: i32,
    pub online_mode: bool,
}

impl PlayJoinInfo {
    pub fn decode(r: &mut PacketReader<'_>) -> ProtocolResult<Self> {
        let entity_id = r.read_i32()?;
        let _hardcore = r.read_bool()?;
        let dim_count = r.read_usize_bounded(256)?;
        for _ in 0..dim_count {
            let _ = r.read_string(32767)?;
        }
        let _max_players = r.read_varint()?;
        let view_distance = r.read_varint()?;
        let _sim_distance = r.read_varint()?;
        let _reduced_debug = r.read_bool()?;
        let _show_death = r.read_bool()?;
        let _limited_crafting = r.read_bool()?;

        // CommonPlayerSpawnInfo
        let _dim_type_id = r.read_varint()?;
        let dimension_name = r.read_string(32767)?;
        let _seed = r.read_i64()?;
        let game_mode = r.read_u8()?;
        let _prev_game_mode = r.read_i8()?;
        let _is_debug = r.read_bool()?;
        let _is_flat = r.read_bool()?;
        if r.read_bool()? {
            let _death_dim = r.read_string(32767)?;
            let _death_pos = r.read_block_pos()?;
        }
        let _portal_cooldown = r.read_varint()?;
        let sea_level = r.read_varint()?;
        let online_mode = r.read_bool()?;
        let _enforces_secure = r.read_bool()?;

        Ok(Self {
            entity_id,
            dimension_name,
            view_distance,
            game_mode,
            sea_level,
            online_mode,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlayerSpawnTransform {
    pub teleport_id: i32,
    pub position: [f64; 3],
    pub yaw_deg: f32,
    pub pitch_deg: f32,
}

impl PlayerSpawnTransform {
    pub fn decode(r: &mut PacketReader<'_>) -> ProtocolResult<Self> {
        let teleport_id = r.read_varint()?;
        let x = r.read_f64()?;
        let y = r.read_f64()?;
        let z = r.read_f64()?;
        let _dx = r.read_f64()?;
        let _dy = r.read_f64()?;
        let _dz = r.read_f64()?;
        let yaw_deg = r.read_f32()?;
        let pitch_deg = r.read_f32()?;
        let _relatives = r.read_i32()?;
        Ok(Self {
            teleport_id,
            position: [x, y, z],
            yaw_deg,
            pitch_deg,
        })
    }

    pub fn encode_accept(&self) -> Vec<u8> {
        let mut w = PacketWriter::with_capacity(36);
        w.write_varint(self.teleport_id);
        w.write_f64(self.position[0]);
        w.write_f64(self.position[1]);
        w.write_f64(self.position[2]);
        w.write_f32(self.yaw_deg);
        w.write_f32(self.pitch_deg);
        w.into_inner()
    }
}

/// Snapshot of initial `Play` state data received upon joining a server (`Login` + initial chunks + spawn position).
#[derive(Debug)]
pub struct InitialPlayBootstrap {
    pub join_info: PlayJoinInfo,
    pub spawn: PlayerSpawnTransform,
    pub raw_chunk_packets: Vec<Vec<u8>>,
}

/// Collects the initial Play state login, streamed spawn chunks, and player spawn position,
/// replying to `KeepAlive` and acknowledging the initial teleport.
pub async fn bootstrap_play_session(
    conn: &mut ClientConnection,
) -> ProtocolResult<InitialPlayBootstrap> {
    use generated_packets::play::{clientbound as s2c, serverbound as c2s};

    let mut join_info: Option<PlayJoinInfo> = None;
    let mut raw_chunk_packets = Vec::new();

    loop {
        let (pkt_id, payload) = conn.read_packet().await?;
        match pkt_id {
            s2c::DISCONNECT => {
                let mut r = PacketReader::new(&payload);
                let tag = read_network_nbt(&mut r)?;
                return Err(ProtocolError::ServerDisconnect(format!("{tag:?}")));
            }
            s2c::KEEP_ALIVE => {
                conn.send_packet(c2s::KEEP_ALIVE, &payload).await?;
            }
            s2c::PING => {
                conn.send_packet(c2s::PONG, &payload).await?;
            }
            s2c::LOGIN => {
                let mut r = PacketReader::new(&payload);
                join_info = Some(PlayJoinInfo::decode(&mut r)?);
            }
            s2c::LEVEL_CHUNK_WITH_LIGHT => {
                raw_chunk_packets.push(payload);
            }
            s2c::CHUNK_BATCH_FINISHED => {
                // Acknowledge chunk batch with desired chunks/tick = 25.0
                let mut w = PacketWriter::with_capacity(4);
                w.write_f32(25.0);
                conn.send_packet(c2s::CHUNK_BATCH_RECEIVED, w.as_slice())
                    .await?;
            }
            s2c::PLAYER_POSITION => {
                let mut r = PacketReader::new(&payload);
                let spawn = PlayerSpawnTransform::decode(&mut r)?;
                conn.send_packet(c2s::ACCEPT_TELEPORTATION, &spawn.encode_accept())
                    .await?;
                conn.send_packet(c2s::PLAYER_LOADED, &[]).await?;
                let info = join_info.ok_or_else(|| {
                    ProtocolError::Auth("received PlayerPosition before Play Login".into())
                })?;
                return Ok(InitialPlayBootstrap {
                    join_info: info,
                    spawn,
                    raw_chunk_packets,
                });
            }
            _ => {}
        }
    }
}
