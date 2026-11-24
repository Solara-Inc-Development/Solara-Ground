use crate::PROTOCOL_VERSION;
use crate::buf::{PacketReader, PacketWriter};
use crate::crypto::{MinecraftCipher, encrypt_login_key_response, mojang_server_hash};
use crate::error::{ProtocolError, ProtocolResult};
use crate::frame::{MAX_WIRE_FRAME_BYTES, decode_frame_body, encode_frame};
use crate::generated_packets;
use crate::packets::{
    ClientIntent, HandshakeC2s, LoginFinishedS2c, LoginHelloC2s, LoginHelloS2c, LoginKeyC2s,
    ServerStatusResponse, decode_disconnect_reason,
};
use rand::RngCore;
use serde::Serialize;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use uuid::Uuid;

/// Framed client-side Minecraft TCP connection supporting AES-128-CFB8 and zlib compression.
pub struct ClientConnection {
    stream: TcpStream,
    cipher: Option<MinecraftCipher>,
    compression_threshold: Option<usize>,
}

impl ClientConnection {
    pub async fn connect(host: &str, port: u16) -> ProtocolResult<Self> {
        let stream = TcpStream::connect((host, port)).await?;
        let _ = stream.set_nodelay(true);
        Ok(Self {
            stream,
            cipher: None,
            compression_threshold: None,
        })
    }

    pub fn enable_encryption(&mut self, secret: &[u8; 16]) {
        self.cipher = Some(MinecraftCipher::new(secret));
    }

    pub fn set_compression(&mut self, threshold: i32) {
        self.compression_threshold = (threshold >= 0).then_some(threshold as usize);
    }

    async fn read_byte(&mut self) -> ProtocolResult<u8> {
        let mut b = [0u8; 1];
        self.stream.read_exact(&mut b).await?;
        if let Some(c) = &mut self.cipher {
            c.decrypt_in_place(&mut b);
        }
        Ok(b[0])
    }

    pub async fn read_packet(&mut self) -> ProtocolResult<(i32, Vec<u8>)> {
        let mut num_read = 0usize;
        let mut frame_len_u32 = 0u32;
        loop {
            let byte = self.read_byte().await?;
            if num_read >= 5 {
                return Err(ProtocolError::VarIntTooLarge);
            }
            frame_len_u32 |= u32::from(byte & 0x7F) << (7 * num_read);
            num_read += 1;
            if (byte & 0x80) == 0 {
                break;
            }
        }
        let frame_len_i32 = frame_len_u32 as i32;
        if frame_len_i32 < 0 {
            return Err(ProtocolError::NegativeLength(frame_len_i32));
        }
        let frame_len = frame_len_i32 as usize;
        if frame_len > MAX_WIRE_FRAME_BYTES {
            return Err(ProtocolError::PacketTooLarge {
                len: frame_len,
                max: MAX_WIRE_FRAME_BYTES,
            });
        }
        let mut body = vec![0u8; frame_len];
        self.stream.read_exact(&mut body).await?;
        if let Some(c) = &mut self.cipher {
            c.decrypt_in_place(&mut body);
        }
        decode_frame_body(&body, self.compression_threshold.is_some())
    }

    pub async fn send_packet(&mut self, packet_id: i32, payload: &[u8]) -> ProtocolResult<()> {
        let mut wire = encode_frame(packet_id, payload, self.compression_threshold)?;
        if let Some(c) = &mut self.cipher {
            c.encrypt_in_place(&mut wire);
        }
        self.stream.write_all(&wire).await?;
        self.stream.flush().await?;
        Ok(())
    }
}

/// Performs a standard Minecraft 26.3 Server List Ping against `(host, port)`, returning
/// the parsed server status and measured ping RTT.
pub async fn ping_server(
    host: &str,
    port: u16,
) -> ProtocolResult<(ServerStatusResponse, Duration)> {
    let mut conn = ClientConnection::connect(host, port).await?;

    let hs = HandshakeC2s {
        protocol_version: PROTOCOL_VERSION,
        server_address: host.to_owned(),
        server_port: port,
        intent: ClientIntent::Status,
    };
    conn.send_packet(HandshakeC2s::PACKET_ID, &hs.encode())
        .await?;
    conn.send_packet(generated_packets::status::serverbound::STATUS_REQUEST, &[])
        .await?;

    let (status_id, status_payload) = conn.read_packet().await?;
    if status_id != generated_packets::status::clientbound::STATUS_RESPONSE {
        return Err(ProtocolError::UnexpectedPacketId(status_id));
    }
    let mut r = PacketReader::new(&status_payload);
    let status = ServerStatusResponse::decode(&mut r)?;

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    let mut ping_w = PacketWriter::with_capacity(8);
    ping_w.write_i64(now_ms);
    let start = Instant::now();
    conn.send_packet(
        generated_packets::status::serverbound::PING_REQUEST,
        ping_w.as_slice(),
    )
    .await?;

    let (pong_id, pong_payload) = conn.read_packet().await?;
    let rtt = start.elapsed();
    if pong_id != generated_packets::status::clientbound::PONG_RESPONSE {
        return Err(ProtocolError::UnexpectedPacketId(pong_id));
    }
    let mut pr = PacketReader::new(&pong_payload);
    let echoed = pr.read_i64()?;
    if echoed != now_ms {
        return Err(ProtocolError::Auth("ping pong timestamp mismatch".into()));
    }

    Ok((status, rtt))
}

#[derive(Serialize)]
struct MojangJoinRequest<'a> {
    #[serde(rename = "accessToken")]
    access_token: &'a str,
    #[serde(rename = "selectedProfile")]
    selected_profile: String,
    #[serde(rename = "serverId")]
    server_id: &'a str,
}

/// Calls Mojang's official `session/minecraft/join` endpoint for online-mode authentication.
pub async fn join_mojang_session(
    http: &reqwest::Client,
    access_token: &str,
    profile_uuid: Uuid,
    server_hash: &str,
) -> ProtocolResult<()> {
    let body = MojangJoinRequest {
        access_token,
        selected_profile: profile_uuid.simple().to_string(),
        server_id: server_hash,
    };
    let resp = http
        .post("https://sessionserver.mojang.com/session/minecraft/join")
        .json(&body)
        .send()
        .await
        .map_err(|e| ProtocolError::Auth(format!("Mojang session join request failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(ProtocolError::Auth(format!(
            "Mojang sessionserver rejected login (HTTP {})",
            resp.status()
        )));
    }
    Ok(())
}

/// Performs the online-mode Login handshake up to `ClientboundLoginFinishedPacket` + `LoginAcknowledged`.
pub async fn login_online(
    host: &str,
    port: u16,
    username: &str,
    profile_uuid: Uuid,
    access_token: &str,
) -> ProtocolResult<(ClientConnection, LoginFinishedS2c)> {
    let mut conn = ClientConnection::connect(host, port).await?;
    let hs = HandshakeC2s {
        protocol_version: PROTOCOL_VERSION,
        server_address: host.to_owned(),
        server_port: port,
        intent: ClientIntent::Login,
    };
    conn.send_packet(HandshakeC2s::PACKET_ID, &hs.encode())
        .await?;

    let hello = LoginHelloC2s {
        username: username.to_owned(),
        profile_id: profile_uuid,
    };
    conn.send_packet(LoginHelloC2s::PACKET_ID, &hello.encode())
        .await?;

    let http = reqwest::Client::builder()
        .user_agent("Solara-Ground/0.1")
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| ProtocolError::Auth(e.to_string()))?;

    loop {
        let (pkt_id, payload) = conn.read_packet().await?;
        match pkt_id {
            generated_packets::login::clientbound::LOGIN_DISCONNECT => {
                let mut r = PacketReader::new(&payload);
                return Err(ProtocolError::ServerDisconnect(decode_disconnect_reason(
                    &mut r,
                )?));
            }
            generated_packets::login::clientbound::HELLO => {
                let mut r = PacketReader::new(&payload);
                let enc_req = LoginHelloS2c::decode(&mut r)?;

                let mut shared_secret = [0u8; 16];
                rand::thread_rng().fill_bytes(&mut shared_secret);

                if enc_req.should_authenticate {
                    let server_hash = mojang_server_hash(
                        &enc_req.server_id,
                        &shared_secret,
                        &enc_req.public_key_der,
                    );
                    join_mojang_session(&http, access_token, profile_uuid, &server_hash).await?;
                }

                let (enc_secret, enc_token) = encrypt_login_key_response(
                    &enc_req.public_key_der,
                    &shared_secret,
                    &enc_req.verify_token,
                )?;
                let key_pkt = LoginKeyC2s {
                    shared_secret_encrypted: enc_secret,
                    verify_token_encrypted: enc_token,
                };
                conn.send_packet(LoginKeyC2s::PACKET_ID, &key_pkt.encode())
                    .await?;
                conn.enable_encryption(&shared_secret);
            }
            generated_packets::login::clientbound::LOGIN_COMPRESSION => {
                let mut r = PacketReader::new(&payload);
                let threshold = r.read_varint()?;
                conn.set_compression(threshold);
            }
            generated_packets::login::clientbound::CUSTOM_QUERY => {
                let mut r = PacketReader::new(&payload);
                let tx_id = r.read_varint()?;
                // Standard vanilla client replies `false` (not understood) to unknown login queries
                let mut w = PacketWriter::new();
                w.write_varint(tx_id);
                w.write_bool(false);
                conn.send_packet(
                    generated_packets::login::serverbound::CUSTOM_QUERY_ANSWER,
                    w.as_slice(),
                )
                .await?;
            }
            generated_packets::login::clientbound::LOGIN_FINISHED => {
                let mut r = PacketReader::new(&payload);
                let finished = LoginFinishedS2c::decode(&mut r)?;
                conn.send_packet(
                    generated_packets::login::serverbound::LOGIN_ACKNOWLEDGED,
                    &[],
                )
                .await?;
                return Ok((conn, finished));
            }
            other => return Err(ProtocolError::UnexpectedPacketId(other)),
        }
    }
}
