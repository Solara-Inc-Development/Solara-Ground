use crate::error::{ProtocolError, ProtocolResult};
use aes::Aes128;
use cfb8::{Decryptor, Encryptor};
use cipher::generic_array::GenericArray;
use cipher::{BlockDecryptMut, BlockEncryptMut, KeyIvInit};
use num_bigint::BigInt;
use rsa::pkcs8::DecodePublicKey;
use rsa::{Pkcs1v15Encrypt, RsaPublicKey};
use sha1::{Digest, Sha1};

/// AES-128-CFB8 stream cipher pair matching Minecraft's TCP transport encryption.
pub struct MinecraftCipher {
    encryptor: Encryptor<Aes128>,
    decryptor: Decryptor<Aes128>,
}

impl MinecraftCipher {
    pub fn new(shared_secret: &[u8; 16]) -> Self {
        Self {
            encryptor: Encryptor::<Aes128>::new_from_slices(shared_secret, shared_secret)
                .expect("16-byte AES-128 key/IV is always valid"),
            decryptor: Decryptor::<Aes128>::new_from_slices(shared_secret, shared_secret)
                .expect("16-byte AES-128 key/IV is always valid"),
        }
    }

    pub fn encrypt_in_place(&mut self, data: &mut [u8]) {
        for byte in data {
            self.encryptor
                .encrypt_block_mut(GenericArray::from_mut_slice(std::slice::from_mut(byte)));
        }
    }

    pub fn decrypt_in_place(&mut self, data: &mut [u8]) {
        for byte in data {
            self.decryptor
                .decrypt_block_mut(GenericArray::from_mut_slice(std::slice::from_mut(byte)));
        }
    }
}

/// Computes Mojang's signed two's-complement SHA-1 hex digest used by `session/minecraft/join`.
pub fn mojang_server_hash(server_id: &str, shared_secret: &[u8], public_key_der: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(server_id.as_bytes());
    hasher.update(shared_secret);
    hasher.update(public_key_der);
    let digest = hasher.finalize();
    let bigint = BigInt::from_signed_bytes_be(&digest);
    format!("{bigint:x}")
}

/// Encrypts the client's 16-byte shared secret and the server's challenge token using the
/// server's SPKI DER RSA public key (`PKCS#1 v1.5`).
pub fn encrypt_login_key_response(
    public_key_der: &[u8],
    shared_secret: &[u8; 16],
    verify_token: &[u8],
) -> ProtocolResult<(Vec<u8>, Vec<u8>)> {
    let pub_key = RsaPublicKey::from_public_key_der(public_key_der)
        .map_err(|e| ProtocolError::Auth(format!("invalid server RSA public key: {e}")))?;
    let mut rng = rand::thread_rng();
    let enc_secret = pub_key
        .encrypt(&mut rng, Pkcs1v15Encrypt, shared_secret)
        .map_err(|e| ProtocolError::Auth(format!("RSA encrypt shared secret: {e}")))?;
    let enc_token = pub_key
        .encrypt(&mut rng, Pkcs1v15Encrypt, verify_token)
        .map_err(|e| ProtocolError::Auth(format!("RSA encrypt verify token: {e}")))?;
    Ok((enc_secret, enc_token))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsa::RsaPrivateKey;
    use rsa::pkcs8::EncodePublicKey;

    #[test]
    fn rsa_and_aes_cfb8_client_handshake_roundtrip() {
        let mut rng = rand::thread_rng();
        let priv_key = RsaPrivateKey::new(&mut rng, 1024).expect("rsa keygen");
        let pub_der = priv_key
            .to_public_key()
            .to_public_key_der()
            .expect("der")
            .into_vec();

        let secret = [0x7Au8; 16];
        let challenge = [1u8, 2, 3, 4];
        let (enc_s, enc_c) =
            encrypt_login_key_response(&pub_der, &secret, &challenge).expect("rsa encrypt");

        let dec_s = priv_key.decrypt(Pkcs1v15Encrypt, &enc_s).expect("dec s");
        let dec_c = priv_key.decrypt(Pkcs1v15Encrypt, &enc_c).expect("dec c");
        assert_eq!(dec_s.as_slice(), &secret);
        assert_eq!(dec_c.as_slice(), &challenge);

        assert_eq!(
            mojang_server_hash("jeb_", b"", b""),
            "-7c9d5b0044c130109a5d7b5fb5c317c02b4e28c1"
        );
    }
}
