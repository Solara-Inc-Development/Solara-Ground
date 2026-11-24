use crate::error::{ProtocolError, ProtocolResult};
use crate::varint::{read_varint, varint_len, write_varint};
use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use std::io::{Read, Write};

pub const MAX_WIRE_FRAME_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_UNCOMPRESSED_BYTES: usize = 8 * 1024 * 1024;

pub fn encode_frame(
    packet_id: i32,
    body: &[u8],
    compression_threshold: Option<usize>,
) -> ProtocolResult<Vec<u8>> {
    let mut uncompressed = Vec::with_capacity(varint_len(packet_id) + body.len());
    write_varint(packet_id, &mut uncompressed);
    uncompressed.extend_from_slice(body);

    match compression_threshold {
        None => {
            let mut out =
                Vec::with_capacity(varint_len(uncompressed.len() as i32) + uncompressed.len());
            write_varint(uncompressed.len() as i32, &mut out);
            out.extend_from_slice(&uncompressed);
            Ok(out)
        }
        Some(threshold) => {
            if uncompressed.len() < threshold {
                let data_len = 0i32;
                let total_len = varint_len(data_len) + uncompressed.len();
                let mut out = Vec::with_capacity(varint_len(total_len as i32) + total_len);
                write_varint(total_len as i32, &mut out);
                write_varint(data_len, &mut out);
                out.extend_from_slice(&uncompressed);
                Ok(out)
            } else {
                let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
                encoder
                    .write_all(&uncompressed)
                    .map_err(|e| ProtocolError::Compression(e.to_string()))?;
                let compressed = encoder
                    .finish()
                    .map_err(|e| ProtocolError::Compression(e.to_string()))?;
                let data_len = uncompressed.len() as i32;
                let total_len = varint_len(data_len) + compressed.len();
                let mut out = Vec::with_capacity(varint_len(total_len as i32) + total_len);
                write_varint(total_len as i32, &mut out);
                write_varint(data_len, &mut out);
                out.extend_from_slice(&compressed);
                Ok(out)
            }
        }
    }
}

pub fn decode_frame_body(
    frame_body: &[u8],
    compression_enabled: bool,
) -> ProtocolResult<(i32, Vec<u8>)> {
    if frame_body.len() > MAX_WIRE_FRAME_BYTES {
        return Err(ProtocolError::PacketTooLarge {
            len: frame_body.len(),
            max: MAX_WIRE_FRAME_BYTES,
        });
    }

    if !compression_enabled {
        let (packet_id, used) = read_varint(frame_body)?;
        return Ok((packet_id, frame_body[used..].to_vec()));
    }

    let (data_len_raw, header_used) = read_varint(frame_body)?;
    if data_len_raw < 0 {
        return Err(ProtocolError::NegativeLength(data_len_raw));
    }
    let data_len = data_len_raw as usize;
    let remaining = &frame_body[header_used..];

    if data_len == 0 {
        let (packet_id, used) = read_varint(remaining)?;
        return Ok((packet_id, remaining[used..].to_vec()));
    }

    if data_len > MAX_UNCOMPRESSED_BYTES {
        return Err(ProtocolError::PacketTooLarge {
            len: data_len,
            max: MAX_UNCOMPRESSED_BYTES,
        });
    }

    let decoder = ZlibDecoder::new(remaining);
    let mut limited = decoder.take((MAX_UNCOMPRESSED_BYTES + 1) as u64);
    let mut decompressed = Vec::with_capacity(data_len.min(64 * 1024));
    limited
        .read_to_end(&mut decompressed)
        .map_err(|e| ProtocolError::Compression(e.to_string()))?;

    if decompressed.len() != data_len {
        return Err(ProtocolError::Compression(format!(
            "decompressed size {} did not match declared {data_len}",
            decompressed.len()
        )));
    }

    let (packet_id, used) = read_varint(&decompressed)?;
    Ok((packet_id, decompressed[used..].to_vec()))
}
