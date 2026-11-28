//! Lightweight 3D voxel block-break particles, combat hit sparks, and Overworld rain streaks
//! for Solara-Ground, designed for zero-stall per-frame CPU stepping (`swap_remove` + hard cap).

use glam::Vec3;

/// Maximum simultaneous 3D particles/raindrops kept active to guarantee zero frame-time spikes.
pub const MAX_ACTIVE_PARTICLES: usize = 768;

/// Single 3D oriented/axis-aligned colored particle rendered in the entity pass.
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub half_extents: Vec3,
    pub color: [u8; 4],
    pub gravity_scale: f32,
    pub life_left: f32,
}

/// Deterministic fast xorshift32 PRNG for zero-syscall visual particle scatter.
#[derive(Debug, Clone)]
pub struct ParticleWorld {
    particles: Vec<Particle>,
    raining: bool,
    rain_accum: f32,
    rng_state: u32,
}

impl Default for ParticleWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl ParticleWorld {
    pub fn new() -> Self {
        Self {
            particles: Vec::with_capacity(MAX_ACTIVE_PARTICLES),
            raining: false,
            rain_accum: 0.0,
            rng_state: 0x9E37_79B9,
        }
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng_state = if x == 0 { 0xA5A5_5A5A } else { x };
        ((self.rng_state >> 8) as f32) * (1.0 / 16_777_216.0)
    }

    #[inline]
    fn push_capped(&mut self, p: Particle) {
        if self.particles.len() >= MAX_ACTIVE_PARTICLES {
            self.particles.swap_remove(0);
        }
        self.particles.push(p);
    }

    pub fn set_raining(&mut self, raining: bool) {
        self.raining = raining;
    }

    pub fn is_raining(&self) -> bool {
        self.raining
    }

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Spawns a 3D burst of 18 voxel fragments when a block breaks (`LEVEL_EVENT 2001`).
    pub fn spawn_block_break(&mut self, block_pos: [i32; 3], base_rgba: [u8; 4]) {
        let center = Vec3::new(
            block_pos[0] as f32 + 0.5,
            block_pos[1] as f32 + 0.5,
            block_pos[2] as f32 + 0.5,
        );
        for _ in 0..18 {
            let ox = (self.next_f32() - 0.5) * 0.72;
            let oy = (self.next_f32() - 0.5) * 0.72;
            let oz = (self.next_f32() - 0.5) * 0.72;
            let vx = ox * 4.6 + (self.next_f32() - 0.5) * 1.6;
            let vy = 2.4 + self.next_f32() * 3.2;
            let vz = oz * 4.6 + (self.next_f32() - 0.5) * 1.6;
            let shade = 0.78 + self.next_f32() * 0.30;
            let color = [
                ((f32::from(base_rgba[0]) * shade).clamp(0.0, 255.0)) as u8,
                ((f32::from(base_rgba[1]) * shade).clamp(0.0, 255.0)) as u8,
                ((f32::from(base_rgba[2]) * shade).clamp(0.0, 255.0)) as u8,
                255,
            ];
            let life = 0.45 + self.next_f32() * 0.45;
            self.push_capped(Particle {
                pos: center + Vec3::new(ox, oy, oz),
                vel: Vec3::new(vx, vy, vz),
                half_extents: Vec3::splat(0.065),
                color,
                gravity_scale: 1.0,
                life_left: life,
            });
        }
    }

    /// Spawns 12 bright melee hit sparks at `pos`.
    pub fn spawn_combat_sparks(&mut self, pos: Vec3) {
        for _ in 0..12 {
            let vx = (self.next_f32() - 0.5) * 5.2;
            let vy = 1.6 + self.next_f32() * 3.4;
            let vz = (self.next_f32() - 0.5) * 5.2;
            let life = 0.32 + self.next_f32() * 0.22;
            self.push_capped(Particle {
                pos,
                vel: Vec3::new(vx, vy, vz),
                half_extents: Vec3::splat(0.05),
                color: [252, 96, 78, 255],
                gravity_scale: 0.75,
                life_left: life,
            });
        }
    }

    /// Advances active particles and spawns falling Overworld 3D rain streaks around `camera_pos`.
    pub fn step(&mut self, dt: f32, camera_pos: Vec3) {
        let dt = dt.clamp(0.0, 0.1);
        let mut i = 0;
        while i < self.particles.len() {
            self.particles[i].life_left -= dt;
            if self.particles[i].life_left <= 0.0 {
                self.particles.swap_remove(i);
                continue;
            }
            let g = self.particles[i].gravity_scale;
            self.particles[i].vel.y -= 18.0 * g * dt;
            let vel = self.particles[i].vel;
            self.particles[i].pos += vel * dt;
            i += 1;
        }

        if self.raining {
            self.rain_accum += dt * 180.0;
            let spawn_cnt = (self.rain_accum as usize).min(24);
            self.rain_accum -= spawn_cnt as f32;
            for _ in 0..spawn_cnt {
                let rx = (self.next_f32() - 0.5) * 26.0;
                let rz = (self.next_f32() - 0.5) * 26.0;
                let ry = 7.0 + self.next_f32() * 8.0;
                self.push_capped(Particle {
                    pos: camera_pos + Vec3::new(rx, ry, rz),
                    vel: Vec3::new(-0.6, -24.0, 0.4),
                    half_extents: Vec3::new(0.018, 0.28, 0.018),
                    color: [112, 172, 250, 205],
                    gravity_scale: 0.0,
                    life_left: 0.75,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_break_and_rain_particles_spawn_move_and_expire() {
        let mut pw = ParticleWorld::new();
        pw.spawn_block_break([4, -59, 4], [128, 128, 132, 255]);
        assert_eq!(pw.particles().len(), 18);

        pw.set_raining(true);
        pw.step(0.05, Vec3::new(0.0, -56.0, 0.0));
        assert!(pw.particles().len() > 18);

        pw.set_raining(false);
        for _ in 0..20 {
            pw.step(0.1, Vec3::ZERO);
        }
        assert!(pw.particles().is_empty());
    }
}
