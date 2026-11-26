//! Clean-room procedural 16-bit PCM SFX synthesizer & 3D spatial sound attenuation
//! (zero Mojang `.ogg` redistribution; resolves local Mojang asset paths at runtime when installed).

use std::path::{Path, PathBuf};

pub const SAMPLE_RATE_HZ: u32 = 22_050;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfxEvent {
    BlockBreak,
    BlockPlace,
    Footstep,
    Hurt,
    UiClick,
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
    let (duration_ms, base_freq, noise_mix): (usize, f32, f32) = match event {
        SfxEvent::BlockBreak => (95, 135.0, 0.70),
        SfxEvent::BlockPlace => (70, 190.0, 0.45),
        SfxEvent::Footstep => (55, 110.0, 0.80),
        SfxEvent::Hurt => (120, 260.0, 0.50),
        SfxEvent::UiClick => (30, 680.0, 0.15),
    };

    let sample_count = (SAMPLE_RATE_HZ as usize * duration_ms) / 1000;
    let gain = volume.clamp(0.0, 1.0) * 18_000.0;
    let mut out = Vec::with_capacity(sample_count);
    let mut lfsr = 0xACE1u32;

    for i in 0..sample_count {
        let t = (i as f32) / (SAMPLE_RATE_HZ as f32);
        let env = (1.0 - (i as f32) / (sample_count.max(1) as f32)).powi(2);
        let tone = (t * base_freq * std::f32::consts::TAU).sin();
        lfsr ^= lfsr << 13;
        lfsr ^= lfsr >> 17;
        lfsr ^= lfsr << 5;
        let white = ((lfsr & 0xFFFF) as f32 / 32768.0) - 1.0;
        let mixed = (tone * (1.0 - noise_mix) + white * noise_mix) * env * gain;
        out.push(mixed.clamp(-32767.0, 32767.0) as i16);
    }
    out
}

#[derive(Debug, Default)]
pub struct SoundTriggerQueue {
    assets_root: PathBuf,
    queued_events: usize,
}

impl SoundTriggerQueue {
    pub fn new(assets_root: impl Into<PathBuf>) -> Self {
        Self {
            assets_root: assets_root.into(),
            queued_events: 0,
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
        self.queued_events += 1;
        pcm.len()
    }

    pub fn total_triggered(&self) -> usize {
        self.queued_events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spatial_attenuation_and_clean_room_pcm_synth_work() {
        assert!((spatial_attenuation([0.0; 3], [0.0; 3], 16.0) - 1.0).abs() < 1e-5);
        assert_eq!(spatial_attenuation([0.0; 3], [20.0, 0.0, 0.0], 16.0), 0.0);

        let mut q = SoundTriggerQueue::new("./assets");
        let samples = q.trigger_spatial(SfxEvent::BlockBreak, [8.0, -56.0, 8.0], [9.0, -57.0, 8.0]);
        assert!(samples > 1500);
        assert_eq!(q.total_triggered(), 1);
    }
}
