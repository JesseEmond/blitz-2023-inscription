use serde::{Deserialize, Serialize};

pub fn eval_score(visits: u32, ticks: u16, looped: bool) -> i32 {
    let bonus = if looped { 2 } else { 1 };
    // The last visit back home doesn't count.
    let visits = if looped { visits - 1 } else { visits };
    let base = (visits as i32) * 125 - (ticks as i32) * 3;
    base * bonus
}


/// Numbers in TypeScript can be anything under the sun but we only accept u16.
/// Change this type if you hit deserialization errors with numbers.
pub type Number = u16;

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Position {
    pub row: Number,
    pub column: Number,
}

#[derive(Deserialize, Debug)]
pub struct TideLevels {
    pub min: Number,
    pub max: Number,
}

#[derive(Deserialize, Debug)]
/// This is a direct translation from TypeScript's `number[][]` field and isn't very rusty.
/// If you feel that this struct is holding you back don't hesitate to change this and implement your own Deserialize impl.
pub struct Topology(pub Vec<Vec<Number>>);

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Map {
    pub rows: Number,
    pub columns: Number,
    pub topology: Topology,
    pub ports: Vec<Position>,
    pub depth: Number,
    pub tide_levels: TideLevels,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GameTick {
    pub current_tick: Number,
    pub total_ticks: Number,
    pub map: Map,
    pub current_location: Option<Position>,
    pub spawn_location: Option<Position>,
    pub visited_port_indices: Vec<Number>,
    pub tide_schedule: Vec<Number>,
    pub is_over: bool,
}

#[derive(Serialize, Debug, PartialEq, Eq, Clone, Copy)]
pub enum Direction {
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
}

#[derive(Serialize, Debug)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Action {
    Spawn { position: Position },
    Sail { direction: Direction },
    Anchor,
    Dock,
}
