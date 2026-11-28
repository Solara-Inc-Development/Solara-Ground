//! Clean-room procedural 16-bit PCM SFX synthesizer, WAV encoder, 3D spatial sound attenuation,
//! and optional non-blocking OS audio playback worker (zero Mojang `.ogg` redistribution).

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, SyncSender};

pub const SAMPLE_RATE_HZ: u32 = 22_050;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfxEvent {
    BlockBreak,
    BlockPlace,
    Footstep,
    Hurt,
    UiClick,
    ItemPickup,
    EatFood,
    BowShoot,
    LevelUp,
}

impl SfxEvent {
    /// Logical key in the user's local Mojang `assets/indexes/*.json` store.
    pub const fn mojang_asset_key(self) -> &'static str {
        match self {
            Self::BlockBreak => "minecraft/sounds/dig/stone1.ogg",
            Self::BlockPlace => "minecraft/sounds/dig/wood1.ogg",
            Self::Footstep => "minecraft/sounds/step/grass1.ogg",
            Self::Hurt => "minecraft/sounds/damage/hit1.ogg",
            Self::UiClick => "minecraft/sounds/random/click.ogg",
            Self::ItemPickup => "minecraft/sounds/random/pop.ogg",
            Self::EatFood => "minecraft/sounds/random/eat1.ogg",
            Self::BowShoot => "minecraft/sounds/random/bow.ogg",
            Self::LevelUp => "minecraft/sounds/random/levelup.ogg",
        }
    }
}

/// Computes linear-quadratic 3D distance volume attenuation in `[0.0, 1.0]`.
#[inline]
pub fn spatial_attenuation(listener: [f32; 3], source: [f32; 3], max_distance: f32) -> f32 {
    let dx = listener[0] - source[0];
    let dy = listener[1] - source[1];
    let dz = listener[2] - source[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    if dist >= max_distance {
        0.0
    } else {
        let norm = 1.0 - (dist / max_distance.max(0.001));
        norm * norm
    }
}

/// Synthesizes a crisp clean-room 16-bit mono PCM sound buffer when local Mojang `.ogg` files are absent.
pub fn synthesize_clean_room_pcm(event: SfxEvent, volume: f32) -> Vec<i16> {
    let (duration_ms, base_freq, sweep_hz, noise_mix): (usize, f32, f32, f32) = match event {
        SfxEvent::BlockBreak => (95, 135.0, -30.0, 0.70),
        SfxEvent::BlockPlace => (70, 190.0, 25.0, 0.45),
        SfxEvent::Footstep => (50, 110.0, -15.0, 0.80),
        SfxEvent::Hurt => (120, 260.0, -90.0, 0.50),
        SfxEvent::UiClick => (28, 680.0, 120.0, 0.12),
        SfxEvent::ItemPickup => (65, 520.0, 420.0, 0.08),
        SfxEvent::EatFood => (85, 165.0, 40.0, 0.62),
        SfxEvent::BowShoot => (75, 380.0, -140.0, 0.28),
        SfxEvent::LevelUp => (140, 620.0, 360.0, 0.05),
    };

    let sample_count = (SAMPLE_RATE_HZ as usize * duration_ms) / 1000;
    let gain = volume.clamp(0.0, 1.0) * 18_000.0;
    let mut out = Vec::with_capacity(sample_count);
    let mut lfsr = 0xACE1u32;

    for i in 0..sample_count {
        let progress = (i as f32) / (sample_count.max(1) as f32);
        let t = (i as f32) / (SAMPLE_RATE_HZ as f32);
        let env = (1.0 - progress).powi(2);
        let freq = base_freq + sweep_hz * progress;
        let tone = (t * freq * std::f32::consts::TAU).sin();
        lfsr ^= lfsr << 13;
        lfsr ^= lfsr >> 17;
        lfsr ^= lfsr << 5;
        let white = ((lfsr & 0xFFFF) as f32 / 32768.0) - 1.0;
        let mixed = (tone * (1.0 - noise_mix) + white * noise_mix) * env * gain;
        out.push(mixed.clamp(-32767.0, 32767.0) as i16);
    }
    out
}

/// Encodes a standard 44-byte RIFF/WAVE header + 16-bit mono PCM payload for OS audio players.
pub fn encode_wav_16bit_mono(pcm: &[i16], sample_rate: u32) -> Vec<u8> {
    let data_bytes = (pcm.len() * 2) as u32;
    let riff_size = 36u32.saturating_add(data_bytes);
    let byte_rate = sample_rate * 2;
    let mut out = Vec::with_capacity(44 + pcm.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // Mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn try_play_wav_os(wav_bytes: &[u8]) {
    #[cfg(target_os = "linux")]
    {
        use std::io::Write;
        use std::process::{Command, Stdio};

        for &(bin, args) in &[
            ("pw-play", &["-"][..]),
            ("paplay", &["--raw=false"][..]),
            ("aplay", &["-q", "-"][..]),
        ] {
            if let Ok(mut child) = Command::new(bin)
                .args(args)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
            {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(wav_bytes);
                }
                let _ = child.wait();
                return;
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = wav_bytes;
    }
}

pub struct SoundTriggerQueue {
    assets_root: PathBuf,
    queued_events: usize,
    audio_tx: Option<SyncSender<Vec<u8>>>,
}

impl Default for SoundTriggerQueue {
    fn default() -> Self {
        Self::new("./assets")
    }
}

impl std::fmt::Debug for SoundTriggerQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SoundTriggerQueue")
            .field("assets_root", &self.assets_root)
            .field("queued_events", &self.queued_events)
            .finish()
    }
}

impl SoundTriggerQueue {
    pub fn new(assets_root: impl Into<PathBuf>) -> Self {
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(16);
        let spawned = std::thread::Builder::new()
            .name("solara-audio-sfx".into())
            .spawn(move || {
                while let Ok(wav) = rx.recv() {
                    try_play_wav_os(&wav);
                }
            })
            .is_ok();

        Self {
            assets_root: assets_root.into(),
            queued_events: 0,
            audio_tx: spawned.then_some(tx),
        }
    }

    pub fn assets_root(&self) -> &Path {
        &self.assets_root
    }

    pub fn trigger_spatial(
        &mut self,
        event: SfxEvent,
        listener: [f32; 3],
        source: [f32; 3],
    ) -> usize {
        let vol = spatial_attenuation(listener, source, 16.0);
        if vol <= 0.01 {
            return 0;
        }
        let pcm = synthesize_clean_room_pcm(event, vol);
        let len = pcm.len();
        self.queued_events += 1;
        if let Some(tx) = &self.audio_tx {
            let _ = tx.try_send(encode_wav_16bit_mono(&pcm, SAMPLE_RATE_HZ));
        }
        len
    }

    pub fn total_triggered(&self) -> usize {
        self.queued_events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spatial_attenuation_and_clean_room_pcm_wav_synth_work() {
        assert!((spatial_attenuation([0.0; 3], [0.0; 3], 16.0) - 1.0).abs() < 1e-5);
        assert_eq!(spatial_attenuation([0.0; 3], [20.0, 0.0, 0.0], 16.0), 0.0);

        let pcm = synthesize_clean_room_pcm(SfxEvent::ItemPickup, 0.8);
        let wav = encode_wav_16bit_mono(&pcm, SAMPLE_RATE_HZ);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(wav.len(), 44 + pcm.len() * 2);

        let mut q = SoundTriggerQueue::new("./assets");
        let samples = q.trigger_spatial(SfxEvent::BlockBreak, [8.0, -56.0, 8.0], [9.0, -57.0, 8.0]);
        assert!(samples > 1500);
        assert_eq!(q.total_triggered(), 1);
    }
}
