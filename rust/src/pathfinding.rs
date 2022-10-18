use std::collections::HashSet;

use crate::game_interface::{Map, Position};

// TODO: consider packing as u32?
#[derive(Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Copy, Clone)]
pub struct Pos {
    pub x: u16,
    pub y: u16,
}

impl Pos {
    pub fn to_position(&self) -> Position {
        Position { row: self.y, column: self.x }
    }

    pub fn from_position(position: &Position) -> Pos {
        Pos { x: position.column, y: position.row }
    }
}

pub struct Graph {
    tide_schedule: Vec<u8>,
    // TODO: consider representing 2 items per u8?
    topology: Vec<Vec<u8>>,
}

impl Graph {
    pub fn new() -> Self {
        Graph { tide_schedule: Vec::new(), topology: Vec::new() }
    }

    pub fn init_map(&mut self, map: &Map) {
        let topology = &map.topology.0;
        self.topology = topology.iter().map(
            |row| row.iter().map(|&e| e as u8).collect()).collect();
    }

    pub fn init_tide_schedule(&mut self, schedule: &Vec<u8>, tick: u32) {
        self.tide_schedule = schedule.clone();
    }
}

#[derive(Debug, Clone)]
pub struct Path {
    pub steps: Vec<Pos>,
    pub cost: i32,
    pub goal: Pos,
}


// TODO: optimizations to consider:
// - https://www.redblobgames.com/pathfinding/a-star/implementation.html#optimize-integer-ids
// - https://harablog.wordpress.com/2011/09/07/jump-point-search/

pub struct Pathfinder {
    pub graph: Graph,
}

impl Pathfinder {
    pub fn new() -> Self {
        Pathfinder { graph: Graph::new() }
    }

    // TODO: use Vec instead of hashset?
    fn a_star_search(
        &mut self, start: Pos, targets: &HashSet<Pos>, tick: u32
        ) -> Option<Pos> {
        None  // TODO
    }

    pub fn shortest_path(
        &mut self, start: Pos, target: Pos, tick: u32
        ) -> Option<Path> {
        let targets = HashSet::from([target]);
        if self.a_star_search(start, &targets, tick).is_none() {
            None
        } else {
            // TODO
            None
        }
    }

    pub fn distance(
        &mut self, start: Pos, target: Pos, tick: u32
        ) -> Option<i32> {
        None  // TODO
    }

    pub fn path_to_closest(
        &mut self, start: Pos, targets: &HashSet<Pos>, tick: u32
        ) -> Option<Path> {
        None  // TODO
    }
}
