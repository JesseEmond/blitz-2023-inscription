// Implementation of graph.rs, without optimizations.
use std::collections::{HashSet};
use std::sync::{Arc};

use crate::game_interface::{GameTick};
use crate::pathfinding::{Path, Pos};
use crate::simple_pathfinding::SimplePathfinder;

pub type Cost = u8;
pub type VertexId = u8;

pub struct SimpleGraph {
    // adjacency[from][to][tick_offset]
    adjacency: Vec<Vec<Vec<u8>>>,
    // paths[from][to][tick_offset]
    pub paths: Vec<Vec<Vec<Path>>>,
    pub ports: Vec<Pos>,
    pub tick_offsets: usize,
    pub start_tick: u16,
    pub max_ticks: u16,
}

impl SimpleGraph {
    pub fn new(game_tick: &Arc<GameTick>) -> Self {
        let tick = game_tick.current_tick as u16;
        let tick_offsets = game_tick.tide_schedule.len();
        let all_ports: Vec<Pos> = game_tick.map.ports.iter()
            .map(Pos::from_position).collect();
        let mut adjacency = vec![
            vec![vec![Cost::MAX; tick_offsets]; all_ports.len()];
            all_ports.len()];
        let placeholder_path = Path {
            steps: Vec::new(), cost: 0, goal: Pos { x: 0, y: 0 }
        };
        let mut paths = vec![
            vec![vec![placeholder_path; tick_offsets]; all_ports.len()];
            all_ports.len()];
        let mut pathfinder = SimplePathfinder::new();
        let schedule: Vec<u8> = game_tick.tide_schedule.iter()
            .map(|&e| e as u8).collect();
        pathfinder.grid.init(&game_tick.map, &schedule);
        let targets = HashSet::from_iter(all_ports.iter().cloned());
        for from in 0..all_ports.len() {
            let mut targets = targets.clone();
            let port = all_ports[from];
            targets.remove(&port);
            let all_offsets_paths = pathfinder.paths_to_all_targets_by_offset(
                &port, &targets, tick);
            for to in 0..all_ports.len() {
                if from == to {
                    continue;  // leave the defaults for from==to
                }
                let offset_paths = all_offsets_paths.get(&all_ports[to]).unwrap();
                for (offset, path) in offset_paths.iter().enumerate() {
                    paths[from][to][offset] = path.clone();
                    adjacency[from][to][offset] = path.cost as Cost;
                }
            }
        }
        SimpleGraph {
            adjacency,
            paths,
            ports: all_ports,
            tick_offsets: tick_offsets,
            start_tick: tick + 1,  // time to spawn
            max_ticks: game_tick.total_ticks as u16,
        }
    }

    pub fn cost(&self, tick_offset: u8, from: VertexId, to: VertexId) -> Cost {
        self.adjacency[from as usize][to as usize][tick_offset as usize]
    }

    pub fn path(&self, tick_offset: u8, from: VertexId, to: VertexId) -> &Path {
        &self.paths[from as usize][to as usize][tick_offset as usize]
    }

    pub fn tick_offset(&self, tick: u16) -> u8 {
        ((tick as usize) % self.tick_offsets) as u8
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value};
    use std::fs;
    use crate::graph::Graph;
    use super::*;

    fn make_game() -> Arc<GameTick> {
        // Note this isn't great, we ideally shouldn't read from disk here.
        let game_file = "./games/35334.json";
        let game_json = fs::read_to_string(game_file)
            .expect("Couldn't read game file");
        let parsed: Value = serde_json::from_str(&game_json)
            .expect("Couldn't parse JSON in game file");
        let game = serde_json::from_value(parsed)
            .expect("Couldn't parse game tick in game file");
        Arc::new(game)
    }

    #[test]
    fn test_match_graph() {
        let game = make_game();
        let slow = SimpleGraph::new(&game);
        let fast = Graph::new(&game);
        for from in 0..game.map.ports.len() {
            let from = from as VertexId;
            for to in 0..game.map.ports.len() {
                let to = to as VertexId;
                if from == to {
                    continue;
                }
                for offset in 0..game.tide_schedule.len() {
                    let offset = offset as u8;
                    assert_eq!(
                        slow.cost(offset, from, to),
                        fast.cost(offset, from, to),
                        "{}->{}, offset {}", from, to, offset);
                }
            }
        }
    }
}
