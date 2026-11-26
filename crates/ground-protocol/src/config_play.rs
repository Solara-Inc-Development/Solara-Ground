//! Client-side Configuration and Play state handlers for Minecraft 26.3 (protocol 777).

use crate::MINECRAFT_VERSION;
use crate::buf::{PacketReader, PacketWriter};
use crate::client::ClientConnection;
use crate::error::{ProtocolError, ProtocolResult};
use crate::generated_packets;
use crate::nbt::read_network_nbt;
use uuid::Uuid;

/// Official `minecraft:player` protocol ID in `minecraft:entity_type` for MC 26.3 (`777`).
pub const ENTITY_TYPE_PLAYER: i32 = 159;

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
            _ => {}
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

#[derive(Debug, Clone)]
pub struct RemotePlayerInfo {
    pub uuid: Uuid,
    pub username: Option<String>,
    pub game_mode: Option<u8>,
    pub latency_ms: Option<i32>,
}

#[derive(Debug, Clone, Copy)]
pub struct RemoteEntitySpawn {
    pub entity_id: i32,
    pub uuid: Uuid,
    pub entity_type: i32,
    pub position: [f64; 3],
    pub yaw_deg: f32,
    pub pitch_deg: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct RemoteEntityTeleport {
    pub entity_id: i32,
    pub position: [f64; 3],
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub on_ground: bool,
}

fn skip_lp_vec3(r: &mut PacketReader<'_>) -> ProtocolResult<()> {
    let b0 = r.read_u8()?;
    if b0 == 0 {
        return Ok(());
    }
    let _b1 = r.read_u8()?;
    let _b2 = r.read_i32()?;
    if (b0 & 0x04) != 0 {
        let _scale = r.read_varlong()?;
    }
    Ok(())
}

#[inline]
fn angle_byte_to_deg(b: i8) -> f32 {
    (f32::from(b as u8) * 360.0) / 256.0
}

pub fn decode_player_info_update(
    r: &mut PacketReader<'_>,
) -> ProtocolResult<Vec<RemotePlayerInfo>> {
    let actions = r.read_u8()?;
    let count = r.read_usize_bounded(4096)?;
    let mut out = Vec::with_capacity(count);

    for _ in 0..count {
        let uuid = r.read_uuid()?;
        let mut username = None;
        let mut game_mode = None;
        let mut latency_ms = None;

        // Bit 0: ADD_PLAYER
        if (actions & 0x01) != 0 {
            username = Some(r.read_string(16)?);
            let prop_count = r.read_usize_bounded(64)?;
            for _ in 0..prop_count {
                let _name = r.read_string(64)?;
                let _val = r.read_string(32767)?;
                if r.read_bool()? {
                    let _sig = r.read_string(1024)?;
                }
            }
        }
        // Bit 1: INITIALIZE_CHAT
        if (actions & 0x02) != 0 && r.read_bool()? {
            let _chat_session_id = r.read_uuid()?;
            let _expires_at = r.read_i64()?;
            let _pub_key = r.read_byte_array(512)?;
            let _key_sig = r.read_byte_array(4096)?;
        }
        // Bit 2: UPDATE_GAME_MODE
        if (actions & 0x04) != 0 {
            game_mode = Some(r.read_varint()? as u8);
        }
        // Bit 3: UPDATE_LISTED
        if (actions & 0x08) != 0 {
            let _listed = r.read_bool()?;
        }
        // Bit 4: UPDATE_LATENCY
        if (actions & 0x10) != 0 {
            latency_ms = Some(r.read_varint()?);
        }
        // Bit 5: UPDATE_DISPLAY_NAME
        if (actions & 0x20) != 0 && r.read_bool()? {
            let _display_nbt = read_network_nbt(r)?;
        }
        // Bit 6: UPDATE_LIST_ORDER
        if (actions & 0x40) != 0 {
            let _order = r.read_varint()?;
        }
        // Bit 7: UPDATE_HAT
        if (actions & 0x80) != 0 {
            let _hat = r.read_bool()?;
        }

        out.push(RemotePlayerInfo {
            uuid,
            username,
            game_mode,
            latency_ms,
        });
    }
    Ok(out)
}

pub fn decode_player_info_remove(r: &mut PacketReader<'_>) -> ProtocolResult<Vec<Uuid>> {
    let count = r.read_usize_bounded(4096)?;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(r.read_uuid()?);
    }
    Ok(out)
}

pub fn decode_add_entity(r: &mut PacketReader<'_>) -> ProtocolResult<RemoteEntitySpawn> {
    let entity_id = r.read_varint()?;
    let uuid = r.read_uuid()?;
    let entity_type = r.read_varint()?;
    let x = r.read_f64()?;
    let y = r.read_f64()?;
    let z = r.read_f64()?;
    let (pitch_deg, yaw_deg) = if r.remaining() >= 3 {
        skip_lp_vec3(r)?;
        let pitch = angle_byte_to_deg(r.read_i8()?);
        let yaw = angle_byte_to_deg(r.read_i8()?);
        (pitch, yaw)
    } else {
        (0.0, 0.0)
    };
    Ok(RemoteEntitySpawn {
        entity_id,
        uuid,
        entity_type,
        position: [x, y, z],
        yaw_deg,
        pitch_deg,
    })
}

pub fn decode_teleport_entity(r: &mut PacketReader<'_>) -> ProtocolResult<RemoteEntityTeleport> {
    let entity_id = r.read_varint()?;
    let x = r.read_f64()?;
    let y = r.read_f64()?;
    let z = r.read_f64()?;
    let _dx = r.read_f64()?;
    let _dy = r.read_f64()?;
    let _dz = r.read_f64()?;
    let yaw_deg = r.read_f32()?;
    let pitch_deg = r.read_f32()?;
    let _relatives = r.read_i32()?;
    let on_ground = r.read_bool()?;
    Ok(RemoteEntityTeleport {
        entity_id,
        position: [x, y, z],
        yaw_deg,
        pitch_deg,
        on_ground,
    })
}

pub fn decode_remove_entities(r: &mut PacketReader<'_>) -> ProtocolResult<Vec<i32>> {
    let count = r.read_usize_bounded(4096)?;
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        ids.push(r.read_varint()?);
    }
    Ok(ids)
}

/// Encodes `ServerboundMovePlayerPacket.PosRot` (`x, y, z, yaw, pitch, flags`).
pub fn encode_move_player_pos_rot(
    pos: [f64; 3],
    yaw_deg: f32,
    pitch_deg: f32,
    on_ground: bool,
) -> Vec<u8> {
    let mut w = PacketWriter::with_capacity(33);
    w.write_f64(pos[0]);
    w.write_f64(pos[1]);
    w.write_f64(pos[2]);
    w.write_f32(yaw_deg);
    w.write_f32(pitch_deg);
    w.write_u8(u8::from(on_ground));
    w.into_inner()
}

/// Encodes `ServerboundSetCarriedItemPacket` (`slot` in `0..=8`).
pub fn encode_set_carried_item(slot: u16) -> Vec<u8> {
    let mut w = PacketWriter::with_capacity(2);
    w.write_u16(slot);
    w.into_inner()
}

/// Encodes `ServerboundPlayerActionPacket` (`START_DESTROY_BLOCK = 0`).
pub fn encode_start_destroy_block(
    block_pos: (i32, i32, i32),
    face_direction: u8,
    sequence: i32,
) -> Vec<u8> {
    let mut w = PacketWriter::with_capacity(16);
    w.write_varint(0); // START_DESTROY_BLOCK
    w.write_block_pos(block_pos.0, block_pos.1, block_pos.2);
    w.write_u8(face_direction);
    w.write_varint(sequence);
    w.into_inner()
}

/// Encodes `ServerboundPlayerActionPacket` (`STOP_DESTROY_BLOCK = 2`, finishes Survival block break).
pub fn encode_stop_destroy_block(
    block_pos: (i32, i32, i32),
    face_direction: u8,
    sequence: i32,
) -> Vec<u8> {
    let mut w = PacketWriter::with_capacity(16);
    w.write_varint(2); // STOP_DESTROY_BLOCK
    w.write_block_pos(block_pos.0, block_pos.1, block_pos.2);
    w.write_u8(face_direction);
    w.write_varint(sequence);
    w.into_inner()
}

/// Encodes `ServerboundUseItemOnPacket` (main hand block placement against `block_pos` / `face_direction`).
pub fn encode_use_item_on(
    block_pos: (i32, i32, i32),
    face_direction: i32,
    sequence: i32,
) -> Vec<u8> {
    let mut w = PacketWriter::with_capacity(32);
    w.write_varint(0); // InteractionHand::MAIN_HAND
    w.write_block_pos(block_pos.0, block_pos.1, block_pos.2);
    w.write_varint(face_direction);
    w.write_f32(0.5);
    w.write_f32(0.5);
    w.write_f32(0.5);
    w.write_bool(false); // inside
    w.write_bool(false); // worldBorderHit
    w.write_varint(sequence);
    w.into_inner()
}

/// Encodes `ServerboundChatCommandPacket` (without leading `/`).
pub fn encode_chat_command(command: &str) -> Vec<u8> {
    let mut w = PacketWriter::with_capacity(4 + command.len());
    w.write_string(command.trim_start_matches('/'));
    w.into_inner()
}

/// Encodes `ServerboundAttackPacket` (`c2s::ATTACK = 0x01`, `entityId: VarInt`).
pub fn encode_attack_entity(entity_id: i32) -> Vec<u8> {
    let mut w = PacketWriter::with_capacity(5);
    w.write_varint(entity_id);
    w.into_inner()
}

/// Encodes `ServerboundClientCommandPacket` (`PERFORM_RESPAWN = 0`).
pub fn encode_perform_respawn() -> Vec<u8> {
    let mut w = PacketWriter::with_capacity(2);
    w.write_varint(0);
    w.into_inner()
}

/// Encodes `ServerboundCommandSuggestionPacket` (`id: VarInt, command: String`).
pub fn encode_request_command_suggestions(tx_id: i32, input: &str) -> Vec<u8> {
    let mut w = PacketWriter::with_capacity(8 + input.len());
    w.write_varint(tx_id);
    w.write_string(input);
    w.into_inner()
}

/// Decoded live Play-state server event processed during gameplay.
#[derive(Debug, Clone)]
pub enum PlayServerEvent {
    LevelChunkWithLight(Vec<u8>),
    BlockUpdate {
        pos: (i32, i32, i32),
        state_id: u32,
    },
    SystemChat {
        text: String,
        overlay: bool,
    },
    Teleport(PlayerSpawnTransform),
    BlockChangedAck(i32),
    PlayerInfoUpdate(Vec<RemotePlayerInfo>),
    PlayerInfoRemove(Vec<Uuid>),
    AddEntity(RemoteEntitySpawn),
    TeleportEntity(RemoteEntityTeleport),
    RemoveEntities(Vec<i32>),
    SetHealth {
        health: f32,
        food: i32,
        saturation: f32,
    },
    HurtAnimation {
        entity_id: i32,
        yaw_deg: f32,
    },
    Respawn,
    CommandSuggestions {
        tx_id: i32,
        start: usize,
        length: usize,
        matches: Vec<String>,
    },
    Other(i32),
}

/// Reads the next Play-state packet, automatically replying to `KeepAlive`, `Ping`, and `ChunkBatchFinished`.
pub async fn read_next_play_event(conn: &mut ClientConnection) -> ProtocolResult<PlayServerEvent> {
    use generated_packets::play::{clientbound as s2c, serverbound as c2s};

    loop {
        let (pkt_id, payload) = conn.read_packet().await?;
        match pkt_id {
            s2c::KEEP_ALIVE => {
                conn.send_packet(c2s::KEEP_ALIVE, &payload).await?;
            }
            s2c::PING => {
                conn.send_packet(c2s::PONG, &payload).await?;
            }
            s2c::CHUNK_BATCH_FINISHED => {
                let mut w = PacketWriter::with_capacity(4);
                w.write_f32(25.0);
                conn.send_packet(c2s::CHUNK_BATCH_RECEIVED, w.as_slice())
                    .await?;
            }
            s2c::LEVEL_CHUNK_WITH_LIGHT => {
                return Ok(PlayServerEvent::LevelChunkWithLight(payload));
            }
            s2c::BLOCK_UPDATE => {
                let mut r = PacketReader::new(&payload);
                let pos = r.read_block_pos()?;
                let state_id = r.read_varint()? as u32;
                return Ok(PlayServerEvent::BlockUpdate { pos, state_id });
            }
            s2c::SYSTEM_CHAT => {
                let mut r = PacketReader::new(&payload);
                let tag = read_network_nbt(&mut r)?;
                let overlay = r.read_bool()?;
                let text = match &tag {
                    crate::nbt::NbtTag::String(s) => s.clone(),
                    crate::nbt::NbtTag::Compound(entries) => entries
                        .iter()
                        .find(|(k, _)| k == "text")
                        .and_then(|(_, v)| match v {
                            crate::nbt::NbtTag::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_else(|| format!("{tag:?}")),
                    _ => format!("{tag:?}"),
                };
                return Ok(PlayServerEvent::SystemChat { text, overlay });
            }
            s2c::PLAYER_POSITION => {
                let mut r = PacketReader::new(&payload);
                let tp = PlayerSpawnTransform::decode(&mut r)?;
                conn.send_packet(c2s::ACCEPT_TELEPORTATION, &tp.encode_accept())
                    .await?;
                return Ok(PlayServerEvent::Teleport(tp));
            }
            s2c::BLOCK_CHANGED_ACK => {
                let mut r = PacketReader::new(&payload);
                return Ok(PlayServerEvent::BlockChangedAck(r.read_varint()?));
            }
            s2c::PLAYER_INFO_UPDATE => {
                let mut r = PacketReader::new(&payload);
                return Ok(PlayServerEvent::PlayerInfoUpdate(
                    decode_player_info_update(&mut r)?,
                ));
            }
            s2c::PLAYER_INFO_REMOVE => {
                let mut r = PacketReader::new(&payload);
                return Ok(PlayServerEvent::PlayerInfoRemove(
                    decode_player_info_remove(&mut r)?,
                ));
            }
            s2c::ADD_ENTITY => {
                let mut r = PacketReader::new(&payload);
                return Ok(PlayServerEvent::AddEntity(decode_add_entity(&mut r)?));
            }
            s2c::TELEPORT_ENTITY => {
                let mut r = PacketReader::new(&payload);
                return Ok(PlayServerEvent::TeleportEntity(decode_teleport_entity(
                    &mut r,
                )?));
            }
            s2c::REMOVE_ENTITIES => {
                let mut r = PacketReader::new(&payload);
                return Ok(PlayServerEvent::RemoveEntities(decode_remove_entities(
                    &mut r,
                )?));
            }
            s2c::SET_HEALTH => {
                let mut r = PacketReader::new(&payload);
                let health = r.read_f32()?;
                let food = r.read_varint()?;
                let saturation = r.read_f32()?;
                return Ok(PlayServerEvent::SetHealth {
                    health,
                    food,
                    saturation,
                });
            }
            s2c::HURT_ANIMATION => {
                let mut r = PacketReader::new(&payload);
                let entity_id = r.read_varint()?;
                let yaw_deg = r.read_f32()?;
                return Ok(PlayServerEvent::HurtAnimation { entity_id, yaw_deg });
            }
            s2c::RESPAWN => {
                return Ok(PlayServerEvent::Respawn);
            }
            s2c::COMMAND_SUGGESTIONS => {
                let mut r = PacketReader::new(&payload);
                let tx_id = r.read_varint()?;
                let start = r.read_usize_bounded(32500)?;
                let length = r.read_usize_bounded(32500)?;
                let count = r.read_usize_bounded(1024)?;
                let mut matches = Vec::with_capacity(count);
                for _ in 0..count {
                    let text = r.read_string(32500)?;
                    if r.read_bool()? {
                        let _tooltip = read_network_nbt(&mut r)?;
                    }
                    matches.push(text);
                }
                return Ok(PlayServerEvent::CommandSuggestions {
                    tx_id,
                    start,
                    length,
                    matches,
                });
            }
            other => return Ok(PlayServerEvent::Other(other)),
        }
    }
}

/// Snapshot of initial `Play` state data received upon joining a server
/// (`Login` + initial chunks + tab-list players + already-online visible entities + spawn position).
#[derive(Debug)]
pub struct InitialPlayBootstrap {
    pub join_info: PlayJoinInfo,
    pub spawn: PlayerSpawnTransform,
    pub raw_chunk_packets: Vec<Vec<u8>>,
    pub initial_players: Vec<RemotePlayerInfo>,
    pub initial_entities: Vec<RemoteEntitySpawn>,
    pub initial_health: (f32, i32),
}

/// Collects the initial Play state login, streamed spawn chunks, initial players/entities,
/// and player spawn position, replying to `KeepAlive` and acknowledging the initial teleport.
pub async fn bootstrap_play_session(
    conn: &mut ClientConnection,
) -> ProtocolResult<InitialPlayBootstrap> {
    use generated_packets::play::{clientbound as s2c, serverbound as c2s};

    let mut join_info: Option<PlayJoinInfo> = None;
    let mut raw_chunk_packets = Vec::new();
    let mut initial_players = Vec::new();
    let mut initial_entities = Vec::new();
    let mut initial_health = (20.0f32, 20i32);

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
            s2c::PLAYER_INFO_UPDATE => {
                let mut r = PacketReader::new(&payload);
                initial_players.extend(decode_player_info_update(&mut r)?);
            }
            s2c::ADD_ENTITY => {
                let mut r = PacketReader::new(&payload);
                initial_entities.push(decode_add_entity(&mut r)?);
            }
            s2c::SET_HEALTH => {
                let mut r = PacketReader::new(&payload);
                let hp = r.read_f32()?;
                let food = r.read_varint()?;
                initial_health = (hp, food);
            }
            s2c::CHUNK_BATCH_FINISHED => {
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
                    initial_players,
                    initial_entities,
                    initial_health,
                });
            }
            _ => {}
        }
    }
}
