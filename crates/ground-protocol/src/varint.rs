use crate::error::{ProtocolError, ProtocolResult};

pub const fn varint_len(value: i32) -> usize {
    let mut v = value as u32;
    let mut len = 1;
    while (v & !0x7F) != 0 {
        v >>= 7;
        len += 1;
    }
    len
}

pub fn write_varint(value: i32, out: &mut Vec<u8>) {
    let mut v = value as u32;
    while (v & !0x7F) != 0 {
        out.push(((v & 0x7F) as u8) | 0x80);
        v >>= 7;
    }
    out.push((v & 0x7F) as u8);
}

pub fn read_varint(slice: &[u8]) -> ProtocolResult<(i32, usize)> {
    let mut num_read = 0usize;
    let mut result = 0u32;

    loop {
        let Some(&byte) = slice.get(num_read) else {
            return Err(ProtocolError::UnexpectedEof {
                needed: num_read + 1,
                available: slice.len(),
            });
        };
        if num_read >= 5 {
            return Err(ProtocolError::VarIntTooLarge);
        }
        let value = u32::from(byte & 0x7F);
        result |= value << (7 * num_read);
        num_read += 1;

        if (byte & 0x80) == 0 {
            return Ok((result as i32, num_read));
        }
    }
}

pub fn write_varlong(value: i64, out: &mut Vec<u8>) {
    let mut v = value as u64;
    while (v & !0x7F) != 0 {
        out.push(((v & 0x7F) as u8) | 0x80);
        v >>= 7;
    }
    out.push((v & 0x7F) as u8);
}

pub fn read_varlong(slice: &[u8]) -> ProtocolResult<(i64, usize)> {
    let mut num_read = 0usize;
    let mut result = 0u64;

    loop {
        let Some(&byte) = slice.get(num_read) else {
            return Err(ProtocolError::UnexpectedEof {
                needed: num_read + 1,
                available: slice.len(),
            });
        };
        if num_read >= 10 {
            return Err(ProtocolError::VarLongTooLarge);
        }
        let value = u64::from(byte & 0x7F);
        result |= value << (7 * num_read);
        num_read += 1;

        if (byte & 0x80) == 0 {
            return Ok((result as i64, num_read));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_protocol_777_vector() {
        let mut out = Vec::new();
        write_varint(777, &mut out);
        assert_eq!(out, vec![0x89, 0x06]);
        let (v, used) = read_varint(&out).expect("read");
        assert_eq!((v, used), (777, 2));
    }
}
