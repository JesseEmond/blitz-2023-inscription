use std::collections::{HashMap, HashSet};
use std::iter;
use priority_queue::PriorityQueue;

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

fn chebyshev_distance(a: &Pos, b: &Pos) -> u32 {
    u32::max((a.x as i32 - b.x as i32).abs() as u32,
             (a.y as i32 - b.y as i32).abs() as u32)
}

pub struct Grid {
    tide_schedule: Vec<u8>,
    start_tick: u32,
    // TODO: consider representing 2 items per u8?
    topology: Vec<Vec<u8>>,
    width: usize,
    height: usize,
}

impl Grid {
    pub fn new() -> Self {
        Grid {
            tide_schedule: Vec::new(),
            topology: Vec::new(),
            start_tick: 0,
            width: 0,
            height: 0,
        }
    }

    pub fn init(&mut self, map: &Map, schedule: &Vec<u8>, tick: u32) {
        let topology = &map.topology.0;
        self.topology = topology.iter().map(
            |row| row.iter().map(|&e| e as u8).collect()).collect();
        self.height = topology.len();
        self.width = topology[0].len();
        self.tide_schedule = schedule.clone();
        self.start_tick = tick;
    }

    pub fn tide(&self, tick: u32) -> u8 {
        let idx = ((tick - self.start_tick) as usize) % self.tide_schedule.len();
        self.tide_schedule[idx]
    }

    pub fn navigable(&self, pos: &Pos, tick: u32) -> bool {
        self.topology[pos.y as usize][pos.x as usize] < self.tide(tick)
    }

    pub fn neighbors(&self, pos: Pos, tick: u32) -> impl Iterator<Item=Pos> + '_ {
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

#[derive(Debug, Clone)]
pub struct Path {
    pub steps: Vec<Pos>,
    pub cost: u32,
    pub goal: Pos,
}


// TODO: optimizations to consider:
// - https://www.redblobgames.com/pathfinding/a-star/implementation.html#optimize-integer-ids
// - https://harablog.wordpress.com/2011/09/07/jump-point-search/

#[derive(Eq, Hash, PartialEq, Copy, Clone)]
pub struct State {
    position: Pos,
    wait: u8,
}

type CameFrom = HashMap<State, State>;
type CostSoFar = HashMap<State, u32>;
type Targets = HashSet<Pos>;

fn heuristic(src: &Pos, targets: &Targets) -> u32 {
    targets.iter().map(|target| chebyshev_distance(src, target)).min().unwrap()
}

pub struct Pathfinder {
    pub grid: Grid,

    came_from: CameFrom,
    cost_so_far: CostSoFar,
}

impl Pathfinder {
    pub fn new() -> Self {
        Pathfinder {
            grid: Grid::new(),
            came_from: CameFrom::new(),
            cost_so_far: CostSoFar::new(),
        }
    }

    fn reconstruct_path(&self, start: &Pos, goal: &Pos) -> Option<Path> {
        let mut current = State { position: *goal, wait: 0 };
        if !self.came_from.contains_key(&current) {
            return None;
        }
        let cost = *self.cost_so_far.get(&current).unwrap();
        let mut steps = Vec::new();
        while current.position != *start || current.wait > 0 {
            steps.push(current.position);
            current = *self.came_from.get(&current).unwrap();
        }
        steps.push(*start);
        steps.reverse();
        Some(Path { steps: steps, cost: cost, goal: *goal })
    }

    // TODO: use Vec instead of hashset?
    fn a_star_search(
        &mut self, start: &Pos, targets: &Targets, tick: u32,
        stop_on_first: bool
        ) -> Option<Pos> {
        self.cost_so_far.clear();
        self.came_from.clear();

        let mut targets = targets.clone();
        let mut first_goal: Option<Pos> = None;

        // TODO: worth using a different data structure?
        // NOTE: Higher priority pops first.
        let mut frontier: PriorityQueue<State, i32> = PriorityQueue::new();
        let start_state = State { position: *start, wait: 0 };
        frontier.push(start_state, 0);
        self.came_from.insert(start_state, start_state);
        self.cost_so_far.insert(start_state, 0);

        while !frontier.is_empty() {
            let (current, _) = frontier.pop().unwrap();
            let cost = self.cost_so_far.get(&current).unwrap().clone();
            let current_tick = tick + cost;

            if targets.contains(&current.position) {
                targets.remove(&current.position);
                first_goal = Some(current.position);
                if stop_on_first {
                    break;
                }
            }
            if targets.is_empty() {
                break;
            }

            let neighbors = self.grid.neighbors(current.position, current_tick)
                .map(|n| State { position: n, wait: 0 });
            let wait_here = iter::once(
                State { position: current.position, wait: current.wait + 1 });
            // We must wait if we're stuck on ground
            let forced_wait = !self.grid.navigable(&current.position,
                                                    current_tick);
            // No point in waiting longer than a full tide cycle.
            let consider_wait = (current.wait as usize) < self.grid.tide_schedule.len();
            let options = neighbors.filter(|_| !forced_wait)
                .chain(wait_here.filter(|_| consider_wait || forced_wait));

            for next in options {
                let new_cost = cost + 1;  // grid.cost(current, next)
                let old_cost = self.cost_so_far.get(&next).cloned();
                if old_cost.is_none() || new_cost < old_cost.unwrap() {
                    self.cost_so_far.insert(next, new_cost);
                    let priority = new_cost + heuristic(&next.position, &targets);
                    frontier.push(next, -(priority as i32));
                    self.came_from.insert(next, current);
                }
            }
        }
        first_goal
    }

    pub fn shortest_path(
        &mut self, start: &Pos, target: &Pos, tick: u32
        ) -> Option<Path> {
        let targets = Targets::from([*target]);
        if let Some(goal) = self.a_star_search(
            start, &targets, tick, /*stop_on_first=*/true) {
            self.reconstruct_path(start, &goal)
        } else {
            None
        }
    }

    pub fn distance(
        &mut self, start: &Pos, target: &Pos, tick: u32
        ) -> Option<u32> {
        let targets = Targets::from([*target]);
        if let Some(goal) = self.a_star_search(
            start, &targets, tick, /*stop_on_first=*/true) {
            let state = State { position: goal, wait: 0 };
            let cost = *self.cost_so_far.get(&state)
                .expect("Found goal, but no cost set?");
            Some(cost)
        } else {
            None
        }
    }

    pub fn path_to_closest(
        &mut self, start: &Pos, targets: &Targets, tick: u32
        ) -> Option<Path> {
        if let Some(goal) = self.a_star_search(
            start, targets, tick, /*stop_on_first=*/true) {
            self.reconstruct_path(start, &goal)
        } else {
            None
        }
    }

    pub fn paths_to_all_targets(
        &mut self, start: &Pos, targets: &Targets, tick: u32
        ) -> HashMap<Pos, Path> {
        let mut out = HashMap::new();
        self.a_star_search(start, targets, tick, /*stop_on_first=*/false);
        for target in targets {
            if let Some(path) = self.reconstruct_path(start, target) {
                out.insert(*target, path);
            }
        }
        out
    }

    // For each goal, gives a list of paths to the other targets: one for each
    // possible tick offset we start at.
    pub fn paths_to_all_targets_by_offset(
        &mut self, start: &Pos, targets: &Targets, tick: u32
        ) -> HashMap<Pos, Vec<Path>> {
        let mut out = HashMap::new();
        for offset in 0..self.grid.tide_schedule.len() {
            let tick = tick + (offset as u32);
            let all_paths = self.paths_to_all_targets(start, targets, tick);
            for (target, path) in &all_paths {
                assert!(out.contains_key(target) || offset == 0);
                out.entry(*target)
                    .and_modify(|paths: &mut Vec<Path>| paths.push(path.clone()))
                    .or_insert(vec![path.clone()]);
            }
        }
        assert!(out.values().all(|v| v.len() == self.grid.tide_schedule.len()));
        out
    }
}
