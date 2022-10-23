use rustc_hash::{FxHashMap, FxHashSet};
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

fn chebyshev_distance(a: &Pos, b: &Pos) -> u16 {
    u16::max((a.x as i32 - b.x as i32).abs() as u16,
             (a.y as i32 - b.y as i32).abs() as u16)
}

pub struct Grid {
    tide_schedule: Vec<u8>,
    start_tick: u16,
    // TODO: consider representing 2 items per u8?
    topology: Vec<Vec<u8>>,
    width: usize,
    height: usize,
}

impl Default for Grid {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn init(&mut self, map: &Map, schedule: &[u8], tick: u16) {
        let topology = &map.topology.0;
        self.topology = topology.iter().map(
            |row| row.iter().map(|&e| e as u8).collect()).collect();
        self.height = topology.len();
        self.width = topology[0].len();
        self.tide_schedule = schedule.to_owned();
        self.start_tick = tick;
    }

    pub fn tide(&self, tick: u16) -> u8 {
        let idx = ((tick - self.start_tick) as usize) % self.tide_schedule.len();
        self.tide_schedule[idx]
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

#[derive(Debug, Clone)]
pub struct Path {
    pub steps: Vec<Pos>,
    pub cost: u16,
    pub goal: Pos,
}

// TODO: optimizations to consider:
// - https://www.redblobgames.com/pathfinding/a-star/implementation.html#optimize-integer-ids
// - https://harablog.wordpress.com/2011/09/07/jump-point-search/

#[derive(Eq, Hash, PartialEq, Copy, Clone)]
pub struct State(u32);

impl State {
    pub fn new(pos: Pos, wait: u8) -> Self {
        State((pos.x as u32) << 16 | (pos.y as u32) << 8 | (wait as u32))
    }
    pub fn position(&self) -> Pos {
        Pos { x: (self.0 >> 16) as u16, y: ((self.0 >> 8) & 0xff) as u16 }
    }
    pub fn wait(&self) -> u8 {
        (self.0 & 0xff) as u8
    }
}

type CameFrom = FxHashMap<State, State>;
type CostSoFar = FxHashMap<State, u16>;
type Targets = FxHashSet<Pos>;

fn heuristic(src: &Pos, targets: &Targets) -> u16 {
    targets.iter().map(|target| chebyshev_distance(src, target)).min().unwrap()
}

pub struct Pathfinder {
    pub grid: Grid,

    came_from: CameFrom,
    cost_so_far: CostSoFar,
}

impl Default for Pathfinder {
    fn default() -> Self {
        Self::new()
    }
}

impl Pathfinder {
    pub fn new() -> Self {
        Pathfinder {
            grid: Grid::new(),
            came_from: CameFrom::default(),
            cost_so_far: CostSoFar::default(),
        }
    }

    fn reconstruct_path(&self, start: &Pos, goal: &Pos) -> Option<Path> {
        let mut current = State::new(*goal, 0);
        if !self.came_from.contains_key(&current) {
            return None;
        }
        let cost = *self.cost_so_far.get(&current).unwrap();
        let mut steps = Vec::new();
        while current.position() != *start || current.wait() > 0 {
            steps.push(current.position());
            current = *self.came_from.get(&current).unwrap();
        }
        steps.push(*start);
        steps.reverse();
        Some(Path { steps, cost, goal: *goal })
    }

    // TODO: use Vec instead of hashset?
    fn a_star_search(
        &mut self, start: &Pos, targets: &Targets, tick: u16,
        stop_on_first: bool
        ) -> Option<Pos> {
        self.cost_so_far.clear();
        self.came_from.clear();

        let mut targets = targets.clone();
        let mut first_goal: Option<Pos> = None;

        // TODO: worth using a different data structure?
        // NOTE: Higher priority pops first.
        let mut frontier: PriorityQueue<State, i32> = PriorityQueue::new();
        let start_state = State::new(*start, 0);
        frontier.push(start_state, 0);
        self.came_from.insert(start_state, start_state);
        self.cost_so_far.insert(start_state, 0);

        while !frontier.is_empty() {
            let (current, _) = frontier.pop().unwrap();
            let cost = *self.cost_so_far.get(&current).unwrap();
            let current_tick = tick + cost;

            if targets.contains(&current.position()) {
                targets.remove(&current.position());
                first_goal = Some(current.position());
                if stop_on_first {
                    break;
                }
            }
            if targets.is_empty() {
                break;
            }

            let neighbors = self.grid.neighbors(current.position(), current_tick)
                .map(|n| State::new(n, 0));
            let wait_here = iter::once(
                State::new(current.position(), current.wait() + 1));
            // We must wait if we're stuck on ground
            let forced_wait = !self.grid.navigable(&current.position(),
                                                    current_tick);
            // No point in waiting longer than a full tide cycle.
            let consider_wait = (current.wait() as usize) < self.grid.tide_schedule.len();
            let options = neighbors.filter(|_| !forced_wait)
                .chain(wait_here.filter(|_| consider_wait || forced_wait));

            for next in options {
                let new_cost = cost + 1;  // grid.cost(current, next)
                let old_cost = self.cost_so_far.get(&next).cloned();
                if old_cost.is_none() || new_cost < old_cost.unwrap() {
                    self.cost_so_far.insert(next, new_cost);
                    let priority = new_cost + heuristic(&next.position(), &targets);
                    frontier.push(next, -(priority as i32));
                    self.came_from.insert(next, current);
                }
            }
        }
        first_goal
    }

    pub fn shortest_path(
        &mut self, start: &Pos, target: &Pos, tick: u16
        ) -> Option<Path> {
        let targets = Targets::from_iter([*target]);
        if let Some(goal) = self.a_star_search(
            start, &targets, tick, /*stop_on_first=*/true) {
            self.reconstruct_path(start, &goal)
        } else {
            None
        }
    }

    pub fn distance(
        &mut self, start: &Pos, target: &Pos, tick: u16
        ) -> Option<u16> {
        let targets = Targets::from_iter([*target]);
        if let Some(goal) = self.a_star_search(
            start, &targets, tick, /*stop_on_first=*/true) {
            let state = State::new(goal, 0);
            let cost = *self.cost_so_far.get(&state)
                .expect("Found goal, but no cost set?");
            Some(cost)
        } else {
            None
        }
    }

    pub fn path_to_closest(
        &mut self, start: &Pos, targets: &Targets, tick: u16
        ) -> Option<Path> {
        if let Some(goal) = self.a_star_search(
            start, targets, tick, /*stop_on_first=*/true) {
            self.reconstruct_path(start, &goal)
        } else {
            None
        }
    }

    pub fn paths_to_all_targets(
        &mut self, start: &Pos, targets: &Targets, tick: u16
        ) -> FxHashMap<Pos, Path> {
        let mut out = FxHashMap::default();
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
        &mut self, start: &Pos, targets: &Targets, tick: u16
        ) -> FxHashMap<Pos, Vec<Path>> {
        let mut out = FxHashMap::default();
        for offset in 0..self.grid.tide_schedule.len() {
            let tick = tick + (offset as u16);
            let all_paths = self.paths_to_all_targets(start, targets, tick);
            for (target, path) in &all_paths {
                assert!(out.contains_key(target) || offset == 0);
                out.entry(*target)
                    .and_modify(|paths: &mut Vec<Path>| paths.push(path.clone()))
                    .or_insert_with(|| vec![path.clone()]);
            }
        }
        assert!(out.values().all(|v| v.len() == self.grid.tide_schedule.len()));
        out
    }
}
