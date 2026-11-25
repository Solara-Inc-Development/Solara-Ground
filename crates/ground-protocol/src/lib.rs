//! Minecraft: Java Edition 26.3 (protocol 777) client protocol primitives, online auth,
//! Configuration/Play state handlers, and Server List Ping.

pub mod buf;
pub mod client;
pub mod config_play;
pub mod crypto;
pub mod error;
pub mod frame;
pub mod generated_packets;
pub mod nbt;
pub mod packets;
pub mod varint;

pub use error::{ProtocolError, ProtocolResult};

/// Targeted Minecraft: Java Edition version name.
pub const MINECRAFT_VERSION: &str = "26.3";

/// Targeted Minecraft: Java Edition protocol number (26.3 "Wilderness Bound").
pub const PROTOCOL_VERSION: i32 = 777;
