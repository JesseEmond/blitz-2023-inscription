use crate::pathfinding::{Path, Pos};

pub const MAX_PORTS: usize = 20;
pub const MAX_TICK_OFFSETS: usize = 32;

pub fn eval_score(visits: u32, ticks: u16, looped: bool) -> i32 {
    let bonus = if looped { 2 } else { 1 };
    // The last visit back home doesn't count.
    let visits = if looped { visits - 1 } else { visits };
    let base = (visits as i32) * 125 - (ticks as i32) * 3;
    base * bonus
}

pub struct Solution {
    pub score: i32,
    pub spawn: Pos,
    pub paths: Vec<Path>,
}
