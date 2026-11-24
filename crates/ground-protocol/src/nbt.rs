use crate::buf::{PacketReader, PacketWriter};
use crate::error::{ProtocolError, ProtocolResult};

pub const MAX_NBT_DEPTH: usize = 512;
pub const MAX_NBT_ELEMENTS: usize = 1_048_576;

const TAG_END: u8 = 0;
const TAG_BYTE: u8 = 1;
const TAG_SHORT: u8 = 2;
const TAG_INT: u8 = 3;
const TAG_LONG: u8 = 4;
const TAG_FLOAT: u8 = 5;
const TAG_DOUBLE: u8 = 6;
const TAG_BYTE_ARRAY: u8 = 7;
const TAG_STRING: u8 = 8;
const TAG_LIST: u8 = 9;
const TAG_COMPOUND: u8 = 10;
const TAG_INT_ARRAY: u8 = 11;
const TAG_LONG_ARRAY: u8 = 12;

#[derive(Debug, Clone, PartialEq)]
pub enum NbtTag {
    End,
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    ByteArray(Vec<i8>),
    String(String),
    List(Vec<NbtTag>),
    Compound(Vec<(String, NbtTag)>),
    IntArray(Vec<i32>),
    LongArray(Vec<i64>),
}

impl NbtTag {
    pub const fn tag_id(&self) -> u8 {
        match self {
            Self::End => TAG_END,
            Self::Byte(_) => TAG_BYTE,
            Self::Short(_) => TAG_SHORT,
            Self::Int(_) => TAG_INT,
            Self::Long(_) => TAG_LONG,
            Self::Float(_) => TAG_FLOAT,
            Self::Double(_) => TAG_DOUBLE,
            Self::ByteArray(_) => TAG_BYTE_ARRAY,
            Self::String(_) => TAG_STRING,
            Self::List(_) => TAG_LIST,
            Self::Compound(_) => TAG_COMPOUND,
            Self::IntArray(_) => TAG_INT_ARRAY,
            Self::LongArray(_) => TAG_LONG_ARRAY,
        }
    }
}

fn read_nbt_string(r: &mut PacketReader<'_>) -> ProtocolResult<String> {
    let len = usize::from(r.read_u16()?);
    let bytes = r.read_exact(len)?;
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| ProtocolError::InvalidUtf8)
}

fn write_nbt_string(w: &mut PacketWriter, s: &str) {
    let bytes = s.as_bytes();
    w.write_u16(bytes.len() as u16);
    w.write_slice(bytes);
}

fn check_array_len(raw: i32) -> ProtocolResult<usize> {
    if raw < 0 {
        return Err(ProtocolError::NegativeLength(raw));
    }
    let len = raw as usize;
    if len > MAX_NBT_ELEMENTS {
        return Err(ProtocolError::LengthTooLarge {
            len,
            max: MAX_NBT_ELEMENTS,
        });
    }
    Ok(len)
}

fn read_payload(r: &mut PacketReader<'_>, tag_id: u8, depth: usize) -> ProtocolResult<NbtTag> {
    if depth > MAX_NBT_DEPTH {
        return Err(ProtocolError::NbtDepthExceeded(MAX_NBT_DEPTH));
    }
    match tag_id {
        TAG_END => Ok(NbtTag::End),
        TAG_BYTE => Ok(NbtTag::Byte(r.read_i8()?)),
        TAG_SHORT => Ok(NbtTag::Short(r.read_i16()?)),
        TAG_INT => Ok(NbtTag::Int(r.read_i32()?)),
        TAG_LONG => Ok(NbtTag::Long(r.read_i64()?)),
        TAG_FLOAT => Ok(NbtTag::Float(r.read_f32()?)),
        TAG_DOUBLE => Ok(NbtTag::Double(r.read_f64()?)),
        TAG_BYTE_ARRAY => {
            let len = check_array_len(r.read_i32()?)?;
            let raw = r.read_exact(len)?;
            Ok(NbtTag::ByteArray(raw.iter().map(|&b| b as i8).collect()))
        }
        TAG_STRING => Ok(NbtTag::String(read_nbt_string(r)?)),
        TAG_LIST => {
            let elem_id = r.read_u8()?;
            let len = check_array_len(r.read_i32()?)?;
            if elem_id == TAG_END && len > 0 {
                return Err(ProtocolError::InvalidNbtTag(elem_id));
            }
            let mut items = Vec::with_capacity(len.min(1024));
            for _ in 0..len {
                items.push(read_payload(r, elem_id, depth + 1)?);
            }
            Ok(NbtTag::List(items))
        }
        TAG_COMPOUND => {
            let mut entries = Vec::new();
            loop {
                let child_id = r.read_u8()?;
                if child_id == TAG_END {
                    break;
                }
                let name = read_nbt_string(r)?;
                let val = read_payload(r, child_id, depth + 1)?;
                entries.push((name, val));
            }
            Ok(NbtTag::Compound(entries))
        }
        TAG_INT_ARRAY => {
            let len = check_array_len(r.read_i32()?)?;
            let mut vals = Vec::with_capacity(len.min(1024));
            for _ in 0..len {
                vals.push(r.read_i32()?);
            }
            Ok(NbtTag::IntArray(vals))
        }
        TAG_LONG_ARRAY => {
            let len = check_array_len(r.read_i32()?)?;
            let mut vals = Vec::with_capacity(len.min(1024));
            for _ in 0..len {
                vals.push(r.read_i64()?);
            }
            Ok(NbtTag::LongArray(vals))
        }
        other => Err(ProtocolError::InvalidNbtTag(other)),
    }
}

fn write_payload(w: &mut PacketWriter, tag: &NbtTag) {
    match tag {
        NbtTag::End => {}
        NbtTag::Byte(v) => w.write_i8(*v),
        NbtTag::Short(v) => w.write_i16(*v),
        NbtTag::Int(v) => w.write_i32(*v),
        NbtTag::Long(v) => w.write_i64(*v),
        NbtTag::Float(v) => w.write_f32(*v),
        NbtTag::Double(v) => w.write_f64(*v),
        NbtTag::ByteArray(vals) => {
            w.write_i32(vals.len() as i32);
            for &b in vals {
                w.write_i8(b);
            }
        }
        NbtTag::String(s) => write_nbt_string(w, s),
        NbtTag::List(items) => {
            let elem_id = items.first().map_or(TAG_END, NbtTag::tag_id);
            w.write_u8(elem_id);
            w.write_i32(items.len() as i32);
            for item in items {
                write_payload(w, item);
            }
        }
        NbtTag::Compound(entries) => {
            for (name, val) in entries {
                w.write_u8(val.tag_id());
                write_nbt_string(w, name);
                write_payload(w, val);
            }
            w.write_u8(TAG_END);
        }
        NbtTag::IntArray(vals) => {
            w.write_i32(vals.len() as i32);
            for &v in vals {
                w.write_i32(v);
            }
        }
        NbtTag::LongArray(vals) => {
            w.write_i32(vals.len() as i32);
            for &v in vals {
                w.write_i64(v);
            }
        }
    }
}

pub fn read_network_nbt(r: &mut PacketReader<'_>) -> ProtocolResult<NbtTag> {
    let tag_id = r.read_u8()?;
    read_payload(r, tag_id, 0)
}

pub fn write_network_nbt(w: &mut PacketWriter, tag: &NbtTag) {
    w.write_u8(tag.tag_id());
    write_payload(w, tag);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_network_nbt_roundtrip() {
        let nbt = NbtTag::Compound(vec![(
            "dimension".into(),
            NbtTag::String("overworld".into()),
        )]);
        let mut w = PacketWriter::new();
        write_network_nbt(&mut w, &nbt);
        let mut r = PacketReader::new(w.as_slice());
        assert_eq!(read_network_nbt(&mut r).expect("nbt"), nbt);
    }
}
