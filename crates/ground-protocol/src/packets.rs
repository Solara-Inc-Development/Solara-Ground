use crate::buf::{PacketReader, PacketWriter};
use crate::error::ProtocolResult;
use crate::generated_packets;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientIntent {
    Status = 1,
    Login = 2,
    Transfer = 3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandshakeC2s {
    pub protocol_version: i32,
    pub server_address: String,
    pub server_port: u16,
    pub intent: ClientIntent,
}

impl HandshakeC2s {
    pub const PACKET_ID: i32 = generated_packets::handshake::serverbound::INTENTION;

    pub fn encode(&self) -> Vec<u8> {
        let mut w = PacketWriter::new();
        w.write_varint(self.protocol_version);
        w.write_string(&self.server_address);
        w.write_u16(self.server_port);
        w.write_varint(self.intent as i32);
        w.into_inner()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatusVersion {
    pub name: String,
    pub protocol: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatusPlayers {
    pub max: usize,
    pub online: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum StatusDescription {
    Plain(String),
    Component {
        #[serde(default)]
        text: String,
    },
    Other(serde_json::Value),
}

impl StatusDescription {
    pub fn as_plain_text(&self) -> String {
        match self {
            Self::Plain(s) => s.clone(),
            Self::Component { text } => text.clone(),
            Self::Other(v) => v.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerStatusResponse {
    pub version: StatusVersion,
    pub players: StatusPlayers,
    pub description: StatusDescription,
    #[serde(default, rename = "enforcesSecureChat")]
    pub enforces_secure_chat: bool,
    #[serde(default)]
    pub favicon: Option<String>,
}

impl ServerStatusResponse {
    pub fn decode(r: &mut PacketReader<'_>) -> ProtocolResult<Self> {
        let raw_json = r.read_string(32767)?;
        Ok(serde_json::from_str(&raw_json)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginHelloC2s {
    pub username: String,
    pub profile_id: Uuid,
}

impl LoginHelloC2s {
    pub const PACKET_ID: i32 = generated_packets::login::serverbound::HELLO;

    pub fn encode(&self) -> Vec<u8> {
        let mut w = PacketWriter::new();
        w.write_string(&self.username);
        w.write_uuid(self.profile_id);
        w.into_inner()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginHelloS2c {
    pub server_id: String,
    pub public_key_der: Vec<u8>,
    pub verify_token: Vec<u8>,
    pub should_authenticate: bool,
}

impl LoginHelloS2c {
    pub const PACKET_ID: i32 = generated_packets::login::clientbound::HELLO;

    pub fn decode(r: &mut PacketReader<'_>) -> ProtocolResult<Self> {
        Ok(Self {
            server_id: r.read_string(20)?,
            public_key_der: r.read_byte_array(2048)?,
            verify_token: r.read_byte_array(512)?,
            should_authenticate: r.read_bool()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginKeyC2s {
    pub shared_secret_encrypted: Vec<u8>,
    pub verify_token_encrypted: Vec<u8>,
}

impl LoginKeyC2s {
    pub const PACKET_ID: i32 = generated_packets::login::serverbound::KEY;

    pub fn encode(&self) -> Vec<u8> {
        let mut w = PacketWriter::new();
        w.write_byte_array(&self.shared_secret_encrypted);
        w.write_byte_array(&self.verify_token_encrypted);
        w.into_inner()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileProperty {
    pub name: String,
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginFinishedS2c {
    pub uuid: Uuid,
    pub username: String,
    pub properties: Vec<ProfileProperty>,
    pub session_id: Uuid,
}

impl LoginFinishedS2c {
    pub const PACKET_ID: i32 = generated_packets::login::clientbound::LOGIN_FINISHED;

    pub fn decode(r: &mut PacketReader<'_>) -> ProtocolResult<Self> {
        let uuid = r.read_uuid()?;
        let username = r.read_string(16)?;
        let count = r.read_usize_bounded(64)?;
        let mut properties = Vec::with_capacity(count);
        for _ in 0..count {
            let name = r.read_string(32767)?;
            let value = r.read_string(32767)?;
            let signature = if r.read_bool()? {
                Some(r.read_string(32767)?)
            } else {
                None
            };
            properties.push(ProfileProperty {
                name,
                value,
                signature,
            });
        }
        let session_id = r.read_uuid()?;
        Ok(Self {
            uuid,
            username,
            properties,
            session_id,
        })
    }
}

pub fn decode_disconnect_reason(r: &mut PacketReader<'_>) -> ProtocolResult<String> {
    let raw = r.read_string(262144)?;
    if let Ok(desc) = serde_json::from_str::<StatusDescription>(&raw) {
        Ok(desc.as_plain_text())
    } else {
        Ok(raw)
    }
}
