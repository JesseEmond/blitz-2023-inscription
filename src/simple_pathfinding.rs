// Implementation of pathfinding.rs, without optimizations.
use priority_queue::PriorityQueue;
use std::collections::{HashMap, HashSet};
use std::cmp::Reverse;
use std::iter;

use crate::game_interface::{Map};
use crate::pathfinding::{Path, Pos};

pub struct Grid {
    tide_schedule: Vec<u8>,
    // topology[y][x]
    topology: Vec<Vec<u8>>,
    width: usize,
    height: usize,
}

impl Grid {
    pub fn new() -> Self {
        Grid {
            tide_schedule: Vec::new(),
            topology: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    pub fn init(&mut self, map: &Map, schedule: &[u8]) {
        self.width = map.columns as usize;
        self.height = map.rows as usize;
        self.topology = map.topology.0.iter()
            .map(|row| row.iter().map(|&t| t as u8).collect())
            .collect();
        self.tide_schedule = Vec::from_iter(schedule.to_owned());
    }

    pub fn tide(&self, tick: u16) -> u8 {
        let idx = tick % (self.tide_schedule.len() as u16);
        self.tide_schedule[idx as usize]
    }

    pub fn navigable(&self, pos: &Pos, tick: u16) -> bool {
        self.topology[pos.y as usize][pos.x as usize] < self.tide(tick)
    }

    pub fn neighbors(&self, pos: Pos, tick: u16) -> impl Iterator<Item=Pos> + '_ {
        const DELTAS: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                                         (0, 1), (1, -1), (1, 0), (1, 1)];
        DELTAS.iter().filter_map(move |&(dx, dy)| {
            if pos.x == 0 && dx < 0 || pos.y == 0 && dy < 0 {
                None
            } else {
                let neighbor = Pos {
                    x: (pos.x as i32 + dx) as u16,
                    y: (pos.y as i32 + dy) as u16
                };
                if neighbor.x < self.width as u16 &&
                    neighbor.y < self.height as u16
                    && self.navigable(&neighbor, tick) {
                    Some(neighbor)
                } else {
                    None
                }
            }
        })
    }
}

type Targets = HashSet<Pos>;
type CameFrom = HashMap<State, State>;
type CostSoFar = HashMap<State, u16>;

#[derive(Eq, Hash, PartialEq, Copy, Clone)]
struct State {
    pos: Pos,
    wait: u8,
}


pub struct SimplePathfinder {
    pub grid: Grid,

    came_from: CameFrom,
    cost_so_far: CostSoFar,
}

impl SimplePathfinder {
    pub fn new() -> Self {
        SimplePathfinder {
            grid: Grid::new(),
            came_from: CameFrom::default(),
            cost_so_far: CostSoFar::default(),
        }
    }

    fn reconstruct_path(&self, start: &Pos, goal: &Pos) -> Option<Path> {
        let mut current = State { pos: *goal, wait: 0 };
        if !self.came_from.contains_key(&current) {
            return None;
        }
        let cost = *self.cost_so_far.get(&current).unwrap();
        let mut steps = Vec::new();
        while current.pos != *start || current.wait > 0 {
            steps.push(current.pos);
            current = *self.came_from.get(&current).unwrap();
        }
        steps.push(*start);
        steps.reverse();
        Some(Path { steps, cost, goal: *goal })
    }

    fn a_star_search(&mut self, start: &Pos, targets: &Targets, tick: u16) {
        self.cost_so_far.clear();
        self.came_from.clear();

        let mut targets = targets.clone();

        let mut frontier: PriorityQueue<State, Reverse<u16>> = PriorityQueue::new();
        let start_state = State { pos: *start, wait: 0 };
        frontier.push(start_state, Reverse(0));
        self.came_from.insert(start_state, start_state);
        self.cost_so_far.insert(start_state, 0);
        
        while !frontier.is_empty() {
            let (current, _) = frontier.pop().unwrap();
            let cost = *self.cost_so_far.get(&current).unwrap();
            let current_tick = tick + cost;

            if targets.contains(&current.pos) {
                targets.remove(&current.pos);
                if targets.is_empty() {
                    break;
                }
            }

            let neighbors = self.grid.neighbors(current.pos, current_tick)
                .map(|n| State { pos: n, wait: 0 });
            let wait_here = iter::once(State { pos: current.pos, wait: current.wait + 1 });
            // We must wait if we're stuck on ground
            let forced_wait = !self.grid.navigable(&current.pos, current_tick);
            // No point in waiting longer than a full tide cycle.
            let consider_wait = (current.wait as usize) < self.grid.tide_schedule.len();
            let options = neighbors.filter(|_| !forced_wait)
                .chain(wait_here.filter(|_| consider_wait || forced_wait));

            for next in options {
                let new_cost = cost + 1;
                let old_cost = self.cost_so_far.get(&next).cloned();
                if old_cost.is_none() || new_cost < old_cost.unwrap() {
                    self.cost_so_far.insert(next, new_cost);
                    let h = 0;
                    let f = new_cost + h;
                    self.came_from.insert(next, current);
                    frontier.push(next, Reverse(f));
                }
            }
        }
    }

    pub fn paths_to_all_targets(&mut self, start: &Pos, targets: &Targets,
                                tick: u16) -> HashMap<Pos, Path> {
        let mut out = HashMap::new();
        self.a_star_search(start, targets, tick);
        for target in targets {
            if let Some(path) = self.reconstruct_path(start, target) {
                out.insert(*target, path);
            }
        }
        out
    }

    pub fn paths_to_all_targets_by_offset(
        &mut self, start: &Pos, targets: &Targets, tick: u16
        ) -> HashMap<Pos, Vec<Path>> {
        let mut out = HashMap::new();
        for offset in 0..self.grid.tide_schedule.len() {
            let tick = tick + (offset as u16);
            let all_paths = self.paths_to_all_targets(start, targets, tick);
            for (target,path) in &all_paths {
                out.entry(*target)
                    .and_modify(|paths: &mut Vec<Path>| paths.push(path.clone()))
                    .or_insert_with(|| vec![path.clone()]);
            }
        }
        out
    }

    fn _verify_path(&self, path: &Path, tick: u16) {
        let mut tick = tick;
        for i in 1..path.steps.len() {
            let from = path.steps[i - 1];
            let to = path.steps[i];
            if from != to {  // waiting on land is fine
                assert!(self.grid.navigable(&from, tick),
                        "Sailing from ground {:?}->{:?}", from, to);
                assert!(self.grid.navigable(&to, tick),
                        "Sailing to ground {:?}->{:?}", from, to);
            }
            tick += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value};
    use std::fs;
    use crate::game_interface::{GameTick};
    use crate::pathfinding::Pathfinder;
    use crate::pathfinding::{Targets as FastTargets};
    use super::*;

    fn make_game() -> GameTick {
        // Note this isn't great, we ideally shouldn't read from disk here.
        let game_file = "./games/35334.json";
        let game_json = fs::read_to_string(game_file)
            .expect("Couldn't read game file");
        let parsed: Value = serde_json::from_str(&game_json)
            .expect("Couldn't parse JSON in game file");
        serde_json::from_value(parsed)
            .expect("Couldn't parse game tick in game file")
    }

    #[test]
    fn test_match_pathfinder() {
        let game = make_game();
        let schedule: Vec<u8> = game.tide_schedule.iter()
            .map(|&e| e as u8).collect();
        let mut slow = SimplePathfinder::new();
        slow.grid.init(&game.map, &schedule);
        let mut fast = Pathfinder::new();
        fast.grid.init(&game.map, &schedule);
        let ports: Vec<Pos> = game.map.ports.iter().map(Pos::from_position).collect();
        for start in ports.iter().cloned() {
            let mut targets = Targets::from_iter(ports.iter().cloned());
            targets.remove(&start);
            let fast_targets = FastTargets::from_iter(targets.iter().cloned());
            let slow_all_paths = slow.paths_to_all_targets_by_offset(
                &start, &targets, 0);
            let fast_all_paths = fast.paths_to_all_targets_by_offset(
                &start, &fast_targets, 0);
            for target in &targets {
                let slow_offset_paths = slow_all_paths.get(&target).unwrap();
                let fast_offset_paths = fast_all_paths.get(&target).unwrap();
                for offset in 0..schedule.len() {
                    assert_eq!(
                        slow_offset_paths[offset].cost,
                        fast_offset_paths[offset].cost,
                        "path from {:?} to {:?} offset {}",
                        start, target, offset);
                    slow._verify_path(&slow_offset_paths[offset], offset as u16);
                }
            }
        }
    }
}
