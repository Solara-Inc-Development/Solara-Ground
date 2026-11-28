//! Client-side world block map, Minecraft player AABB collision physics, and 3D DDA block raycasting.

use crate::{ChunkPos, ChunkSection, OVERWORLD_MIN_Y, OVERWORLD_SECTIONS, generated_blocks};
use std::collections::HashMap;

pub const PLAYER_HALF_WIDTH: f64 = 0.3;
pub const PLAYER_HEIGHT: f64 = 1.8;
pub const GRAVITY_MPS2: f64 = -32.0;
pub const JUMP_VELOCITY_MPS: f64 = 8.4;
pub const TERMINAL_VELOCITY_MPS: f64 = -78.4;

/// Multi-chunk client world store providing fast block queries and live block updates.
#[derive(Debug, Default, Clone)]
pub struct ClientWorldMap {
    chunks: HashMap<ChunkPos, Vec<ChunkSection>>,
}

impl ClientWorldMap {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn insert_chunk(&mut self, pos: ChunkPos, sections: Vec<ChunkSection>) {
        self.chunks.insert(pos, sections);
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn chunks_iter(&self) -> impl Iterator<Item = (&ChunkPos, &Vec<ChunkSection>)> {
        self.chunks.iter()
    }

    pub fn get_block(&self, x: i32, y: i32, z: i32) -> u32 {
        let rel_y = y - OVERWORLD_MIN_Y;
        if !(0..(OVERWORLD_SECTIONS as i32 * 16)).contains(&rel_y) {
            return generated_blocks::default_state::AIR;
        }
        let cpos = ChunkPos {
            x: x.div_euclid(16),
            z: z.div_euclid(16),
        };
        let Some(sections) = self.chunks.get(&cpos) else {
            return generated_blocks::default_state::AIR;
        };
        let sec_idx = (rel_y as usize) / 16;
        let in_y = (rel_y as usize) % 16;
        let lx = x.rem_euclid(16) as usize;
        let lz = z.rem_euclid(16) as usize;
        sections
            .get(sec_idx)
            .map(|s| s.get(lx, in_y, lz))
            .unwrap_or(generated_blocks::default_state::AIR)
    }

    pub fn set_block(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        state_id: u32,
    ) -> Option<(ChunkPos, usize)> {
        let rel_y = y - OVERWORLD_MIN_Y;
        if !(0..(OVERWORLD_SECTIONS as i32 * 16)).contains(&rel_y) {
            return None;
        }
        let cpos = ChunkPos {
            x: x.div_euclid(16),
            z: z.div_euclid(16),
        };
        let sections = self.chunks.get_mut(&cpos)?;
        let sec_idx = (rel_y as usize) / 16;
        let in_y = (rel_y as usize) % 16;
        let lx = x.rem_euclid(16) as usize;
        let lz = z.rem_euclid(16) as usize;
        sections.get_mut(sec_idx)?.set(lx, in_y, lz, state_id);
        Some((cpos, sec_idx))
    }

    #[inline]
    pub fn is_solid_block(&self, x: i32, y: i32, z: i32) -> bool {
        let st = self.get_block(x, y, z);
        !generated_blocks::is_air_state(st) && st != generated_blocks::default_state::WATER
    }

    #[inline]
    pub fn is_water_block(&self, x: i32, y: i32, z: i32) -> bool {
        self.get_block(x, y, z) == generated_blocks::default_state::WATER
    }
}

#[derive(Debug, Clone, Copy)]
struct Aabb {
    min_x: f64,
    min_y: f64,
    min_z: f64,
    max_x: f64,
    max_y: f64,
    max_z: f64,
}

impl Aabb {
    fn from_feet(pos: [f64; 3]) -> Self {
        Self {
            min_x: pos[0] - PLAYER_HALF_WIDTH,
            min_y: pos[1],
            min_z: pos[2] - PLAYER_HALF_WIDTH,
            max_x: pos[0] + PLAYER_HALF_WIDTH,
            max_y: pos[1] + PLAYER_HEIGHT,
            max_z: pos[2] + PLAYER_HALF_WIDTH,
        }
    }

    fn intersects_solid(&self, world: &ClientWorldMap) -> bool {
        const EPS: f64 = 1e-7;
        let x0 = (self.min_x + EPS).floor() as i32;
        let x1 = (self.max_x - EPS).floor() as i32;
        let y0 = (self.min_y + EPS).floor() as i32;
        let y1 = (self.max_y - EPS).floor() as i32;
        let z0 = (self.min_z + EPS).floor() as i32;
        let z1 = (self.max_z - EPS).floor() as i32;

        for y in y0..=y1 {
            for z in z0..=z1 {
                for x in x0..=x1 {
                    if world.is_solid_block(x, y, z) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn intersects_water(&self, world: &ClientWorldMap) -> bool {
        const EPS: f64 = 1e-7;
        let x0 = (self.min_x + EPS).floor() as i32;
        let x1 = (self.max_x - EPS).floor() as i32;
        let y0 = (self.min_y + EPS).floor() as i32;
        let y1 = (self.max_y - EPS).floor() as i32;
        let z0 = (self.min_z + EPS).floor() as i32;
        let z1 = (self.max_z - EPS).floor() as i32;

        for y in y0..=y1 {
            for z in z0..=z1 {
                for x in x0..=x1 {
                    if world.is_water_block(x, y, z) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Kinematic & collision state for the local player (`0.6 x 1.8 x 0.6` hitbox).
#[derive(Debug, Clone, Copy)]
pub struct PlayerBody {
    pub feet_pos: [f64; 3],
    pub velocity: [f64; 3],
    pub on_ground: bool,
    pub in_water: bool,
}

impl PlayerBody {
    pub fn new(feet_pos: [f64; 3]) -> Self {
        Self {
            feet_pos,
            velocity: [0.0, 0.0, 0.0],
            on_ground: false,
            in_water: false,
        }
    }

    pub fn eye_pos(&self) -> [f64; 3] {
        [self.feet_pos[0], self.feet_pos[1] + 1.62, self.feet_pos[2]]
    }

    /// Advances gravity/buoyancy, jump/swimming, and axis-by-axis (`Y`, `X`, `Z`) voxel AABB collision.
    pub fn step(&mut self, world: &ClientWorldMap, wish_vel_xz: [f64; 2], jump: bool, dt: f64) {
        self.in_water = Aabb::from_feet(self.feet_pos).intersects_water(world);
        if self.in_water {
            self.velocity[0] = wish_vel_xz[0] * 0.72;
            self.velocity[2] = wish_vel_xz[1] * 0.72;
            if jump {
                self.velocity[1] = 4.2;
                self.on_ground = false;
            } else {
                self.velocity[1] = (self.velocity[1] * 0.80 - 4.5 * dt).max(-2.4);
            }
        } else {
            self.velocity[0] = wish_vel_xz[0];
            self.velocity[2] = wish_vel_xz[1];
            if jump && self.on_ground {
                self.velocity[1] = JUMP_VELOCITY_MPS;
                self.on_ground = false;
            }
            self.velocity[1] = (self.velocity[1] + GRAVITY_MPS2 * dt).max(TERMINAL_VELOCITY_MPS);
        }

        let dy = self.velocity[1] * dt;
        let dx = self.velocity[0] * dt;
        let dz = self.velocity[2] * dt;

        // Resolve Y axis first (accurate ground & ceiling detection)
        self.on_ground = false;
        let next_y = self.feet_pos[1] + dy;
        let cand_y = Aabb::from_feet([self.feet_pos[0], next_y, self.feet_pos[2]]);
        if !cand_y.intersects_solid(world) {
            self.feet_pos[1] = next_y;
        } else {
            if dy < 0.0 {
                // Snap cleanly onto top surface of supporting block
                self.feet_pos[1] = (self.feet_pos[1] + 1e-4).floor();
                // Ensure non-overlapping if already slightly below integer plane
                while Aabb::from_feet(self.feet_pos).intersects_solid(world) {
                    self.feet_pos[1] += 1.0;
                }
                self.on_ground = true;
            }
            self.velocity[1] = 0.0;
        }

        // Resolve X axis
        let next_x = self.feet_pos[0] + dx;
        let cand_x = Aabb::from_feet([next_x, self.feet_pos[1], self.feet_pos[2]]);
        if !cand_x.intersects_solid(world) {
            self.feet_pos[0] = next_x;
        } else {
            self.velocity[0] = 0.0;
        }

        // Resolve Z axis
        let next_z = self.feet_pos[2] + dz;
        let cand_z = Aabb::from_feet([self.feet_pos[0], self.feet_pos[1], next_z]);
        if !cand_z.intersects_solid(world) {
            self.feet_pos[2] = next_z;
        } else {
            self.velocity[2] = 0.0;
        }
    }
}

/// Result of a 3D Amanatides-Woo voxel DDA raycast.
///
/// `face_direction` uses Minecraft's protocol `Direction` IDs:
/// `0 = Down (-Y), 1 = Up (+Y), 2 = North (-Z), 3 = South (+Z), 4 = West (-X), 5 = East (+X)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxelRayHit {
    pub block_pos: (i32, i32, i32),
    pub adjacent_pos: (i32, i32, i32),
    pub face_direction: i32,
}

/// Performs an exact 3D Amanatides-Woo voxel traversal up to `max_dist` blocks.
pub fn raycast_blocks(
    world: &ClientWorldMap,
    origin: [f64; 3],
    dir: [f64; 3],
    max_dist: f64,
) -> Option<VoxelRayHit> {
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if len < 1e-9 {
        return None;
    }
    let dx = dir[0] / len;
    let dy = dir[1] / len;
    let dz = dir[2] / len;

    let mut x = origin[0].floor() as i32;
    let mut y = origin[1].floor() as i32;
    let mut z = origin[2].floor() as i32;

    let step_x = if dx >= 0.0 { 1 } else { -1 };
    let step_y = if dy >= 0.0 { 1 } else { -1 };
    let step_z = if dz >= 0.0 { 1 } else { -1 };

    let t_delta_x = if dx.abs() < 1e-12 {
        f64::INFINITY
    } else {
        (1.0 / dx).abs()
    };
    let t_delta_y = if dy.abs() < 1e-12 {
        f64::INFINITY
    } else {
        (1.0 / dy).abs()
    };
    let t_delta_z = if dz.abs() < 1e-12 {
        f64::INFINITY
    } else {
        (1.0 / dz).abs()
    };

    let mut t_max_x = if dx.abs() < 1e-12 {
        f64::INFINITY
    } else if dx > 0.0 {
        ((x + 1) as f64 - origin[0]) * t_delta_x
    } else {
        (origin[0] - x as f64) * t_delta_x
    };
    let mut t_max_y = if dy.abs() < 1e-12 {
        f64::INFINITY
    } else if dy > 0.0 {
        ((y + 1) as f64 - origin[1]) * t_delta_y
    } else {
        (origin[1] - y as f64) * t_delta_y
    };
    let mut t_max_z = if dz.abs() < 1e-12 {
        f64::INFINITY
    } else if dz > 0.0 {
        ((z + 1) as f64 - origin[2]) * t_delta_z
    } else {
        (origin[2] - z as f64) * t_delta_z
    };

    let mut last_face = 1i32; // Up
    let mut prev = (x, y + 1, z);
    let mut t = 0.0;

    while t <= max_dist {
        if world.is_solid_block(x, y, z) {
            return Some(VoxelRayHit {
                block_pos: (x, y, z),
                adjacent_pos: prev,
                face_direction: last_face,
            });
        }
        prev = (x, y, z);

        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                x += step_x;
                t = t_max_x;
                t_max_x += t_delta_x;
                last_face = if step_x > 0 { 4 } else { 5 }; // entered via West (-X) or East (+X)
            } else {
                z += step_z;
                t = t_max_z;
                t_max_z += t_delta_z;
                last_face = if step_z > 0 { 2 } else { 3 }; // North (-Z) or South (+Z)
            }
        } else if t_max_y < t_max_z {
            y += step_y;
            t = t_max_y;
            t_max_y += t_delta_y;
            last_face = if step_y > 0 { 0 } else { 1 }; // Down (-Y) or Up (+Y)
        } else {
            z += step_z;
            t = t_max_z;
            t_max_z += t_delta_z;
            last_face = if step_z > 0 { 2 } else { 3 };
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_test_world() -> ClientWorldMap {
        let mut world = ClientWorldMap::new();
        let mut sections: Vec<ChunkSection> = (0..OVERWORLD_SECTIONS)
            .map(|_| ChunkSection::new_air())
            .collect();
        let stone = generated_blocks::default_state::STONE;
        // Solid floor at world y = -60 (section 0, local y = 4)
        for z in 0..16 {
            for x in 0..16 {
                sections[0].set(x, 4, z, stone);
            }
        }
        // Wall block at (5, -59, 5)
        sections[0].set(5, 5, 5, stone);
        world.insert_chunk(ChunkPos { x: 0, z: 0 }, sections);
        world
    }

    #[test]
    fn player_gravity_lands_cleanly_on_floor_and_stops_at_wall() {
        let world = flat_test_world();
        let mut body = PlayerBody::new([2.5, -56.0, 5.5]);

        for _ in 0..40 {
            body.step(&world, [0.0, 0.0], false, 0.05);
        }
        assert!(body.on_ground);
        assert!((body.feet_pos[1] - -59.0).abs() < 1e-5);

        // Walk +X toward wall at x=5..6 -> player half-width 0.3 must stop before x = 4.7
        for _ in 0..40 {
            body.step(&world, [4.0, 0.0], false, 0.05);
        }
        assert!(body.feet_pos[0] <= 4.7001);
    }

    #[test]
    fn dda_raycast_hits_floor_top_face() {
        let world = flat_test_world();
        let hit = raycast_blocks(&world, [2.5, -57.38, 2.5], [0.0, -1.0, 0.0], 5.0)
            .expect("should hit floor");
        assert_eq!(hit.block_pos, (2, -60, 2));
        assert_eq!(hit.adjacent_pos, (2, -59, 2));
        assert_eq!(hit.face_direction, 1); // Up (+Y)
    }

    #[test]
    fn player_swims_upward_in_water_when_holding_jump() {
        let mut world = flat_test_world();
        world.set_block(2, -59, 2, generated_blocks::default_state::WATER);
        world.set_block(2, -58, 2, generated_blocks::default_state::WATER);
        let mut body = PlayerBody::new([2.5, -59.0, 2.5]);
        body.step(&world, [0.0, 0.0], true, 0.05);
        assert!(body.in_water);
        assert!(body.feet_pos[1] > -59.0);
    }
}
