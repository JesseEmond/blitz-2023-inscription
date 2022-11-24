// Implementation of ant_colony_optimization.rs, without speed optimizations.

use arrayvec::ArrayVec;
use rand::{Rng, SeedableRng};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::SmallRng;
use std::cmp::Ordering;
use std::iter;
use std::sync::Arc;

use crate::ant_colony_optimization::{HyperParams};
use crate::challenge::{Solution, eval_score};
use crate::challenge_consts::{MAX_PORTS, TICK_OFFSETS};
use crate::simple_graph::{SimpleGraph, VertexId};
use crate::solvers::{AntColonyOptimizationSolver};

#[derive(Clone)]
pub struct Ant {
    pub start: VertexId,
    pub current: VertexId,
    pub path: ArrayVec<VertexId, MAX_PORTS>,
    pub tick: u16,
    pub score: i32,
    pub seen: u64,
}

// [to]
type EdgeWeights = ArrayVec<f32, MAX_PORTS>;
// [offset][to]
type EtaPows = ArrayVec<ArrayVec<f32, MAX_PORTS>, TICK_OFFSETS>;

// Holds information coming out of a vertex, used for optimization purposes to
// have better cache locality.
pub struct VertexTrails {
    // τ, pheromone strength of each edge.
    pheromones: ArrayVec<f32, MAX_PORTS>,

    // For a given tick offset, a list of pre-computed weights, one for each
    // edge. Used in sampling.
    // offset_trail_weights[tick_offset][to]
    offset_trail_weights: ArrayVec<EdgeWeights, TICK_OFFSETS>,
    // Pre-computed per-edge eta^beta, for each tick offset.
    // eta_pows[tick_offset][to]
    eta_pows: EtaPows,
}

pub struct Colony {
    pub hyperparams: HyperParams,
    pub graph: Arc<SimpleGraph>,

    // Trails for each vertex.
    pub trails: Vec<VertexTrails>,

    global_best: Option<Ant>,
    rng: SmallRng,
}

impl Ant {
    pub fn new() -> Self {
        Ant {
            start: 0,
            current: 0,
            tick: 0,
            path: ArrayVec::new(),
            score: 0,
            seen: 0,
        }
    }

    pub fn reset(&mut self, start: VertexId, tick: u16) {
        self.start = start;
        self.current = start;
        self.tick = tick + 1;  // time to dock our starting port
        self.path.clear();
        self.seen = 1u64 << start;
    }

    pub fn visit(&mut self, port: VertexId, graph: &SimpleGraph) {
        self.path.push(port);
        let cost = graph.cost(graph.tick_offset(self.tick), self.current, port);
        self.current = port;
        let dock_cost = if port == self.start { 0 } else { 1 };
        self.tick += (cost + dock_cost) as u16;
        self.score = self.compute_score(graph, /*simulate_to_end=*/false);
        self.seen |= 1u64 << port;
    }

    pub fn compute_score(&self, graph: &SimpleGraph, simulate_to_end: bool) -> i32 {
        if self.path.is_empty() {
            return 0;
        }
        let looped = self.current == self.start;
        let visits = self.path.len() + 1;  // +1 for spawn port
        let tick = if simulate_to_end && !looped { graph.max_ticks } else { self.tick };
        eval_score(visits as u32, tick, looped)
    }

    fn valid_option(&self, edge_cost: u8, to: VertexId, graph: &SimpleGraph) -> bool {
        let seen = (self.seen & (1u64 << to)) != 0;
        let have_time = self.tick + (edge_cost as u16) + 1 < graph.max_ticks;
        have_time && !seen
    }

    /// Add a path back home to our path, if we should, potential truncating.
    pub fn finalize_path(&mut self, graph: &SimpleGraph) {
        let mut best_score = self.compute_score(graph, /*simulate_to_end=*/true);
        let mut go_home_at_index: Option<usize> = None;
        for num_edges in 1..=self.path.len() {
            let score = self.hypothetical_home_score(graph, num_edges);
            match score {
                Some(score) if score > best_score => {
                    best_score = score;
                    go_home_at_index = Some(num_edges);  // insert after num_edges
                },
                _ => (),
            }
        }
        if let Some(go_home_index) = go_home_at_index {
            let (vertex, tick) = self.simulate_to_num_edges(graph, go_home_index);
            for to in &self.path[go_home_index..] {
                self.seen ^= 1u64 << (*to as u64)
            }
            self.path.truncate(go_home_index);
            self.tick = tick;
            self.current = vertex;
            self.visit(self.start, graph);
            assert!(self.compute_score(graph, /*simulate_to_end=*/true) == best_score);
        }
    }

    /// Score if we kept only first 'num_edges' and went back home if possible.
    fn hypothetical_home_score(&self, graph: &SimpleGraph, num_edges: usize) -> Option<i32> {
        let (vertex, mut tick) = self.simulate_to_num_edges(graph, num_edges);
        let cost_go_home = graph.cost(graph.tick_offset(tick), vertex, self.start);
        tick += cost_go_home as u16;  // no +1, last home dock is "free"
        if tick < graph.max_ticks {
            let visits = num_edges + 2;  // +1 for spawn, +1 for last
            Some(eval_score(visits as u32, tick, /*looped=*/true))
        } else {
            None
        }
    }

    /// Get final (vertex, tick) if we were to use only first 'num_edges' edges.
    fn simulate_to_num_edges(&self, graph: &SimpleGraph, num_edges: usize) -> (VertexId, u16) {
        let mut tick = graph.start_tick + 1;  // +1 to dock first port
        let mut vertex = self.start;
        for to in &self.path[..num_edges] {
            let cost = graph.cost(graph.tick_offset(tick), vertex, *to);
            let dock_cost = if *to == self.start { 0 } else { 1 };
            tick += (cost as u16) + dock_cost;
            vertex = *to;
        }
        (vertex, tick)
    }

    pub fn to_solution(&self, graph: &SimpleGraph) -> Solution {
        let mut repeat_ant = Ant::new();
        repeat_ant.reset(self.start, graph.start_tick);
        let mut paths = Vec::new();
        for to in &self.path {
            let path = graph.path(graph.tick_offset(repeat_ant.tick),
                                  repeat_ant.current, *to);
            paths.push(path.clone());
            repeat_ant.visit(*to, graph);
        }
        Solution {
            score: self.score,
            spawn: graph.ports[self.start as usize],
            paths,
        }
    }
}

impl VertexTrails {
    pub fn new(from: VertexId, graph: &SimpleGraph,
               hyperparams: &HyperParams) -> Self {
        let eta_pows: EtaPows = (0..TICK_OFFSETS).map(|offset| {
            (0..graph.ports.len()).map(|to| {
                let to = to as VertexId;
                if from != to {
                    let distance = graph.cost(offset as u8, from, to);
                    let beta = hyperparams.heuristic_power;
                    let eta = 1.0 / (distance as f32);
                    eta.powf(beta)
                } else {
                    0f32
                }
            }).collect()
        }).collect();
        // TODO: this should have + min, but matching final version to get the
        // same values.
        let base_pheromones = hyperparams.pheromones_init_ratio * (
            hyperparams.max_pheromones - hyperparams.min_pheromones);
        let weights_unset = ArrayVec::from_iter(
                vec![ArrayVec::from_iter(vec![1.0; graph.ports.len()]); TICK_OFFSETS]);
        let mut trails = VertexTrails {
            pheromones: ArrayVec::from_iter(vec![base_pheromones; graph.ports.len()]),
            offset_trail_weights: weights_unset,
            eta_pows,
        };
        for to in 0..graph.ports.len() {
            trails.update_weights(to as VertexId);
        }
        trails
    }

    pub fn evaporate_add(&mut self, to: VertexId, evaporation: f32, add: f32,
                         hyperparams: &HyperParams) {
        let pheromones = self.pheromones[to as usize];
        self.set_pheromones(
            to, (1.0 - evaporation) * pheromones + evaporation * add,
            hyperparams);
    }

    pub fn evaporate(&mut self, to: VertexId, evaporation: f32,
                     hyperparams: &HyperParams) {
        let pheromones = self.pheromones[to as usize];
        self.set_pheromones(to, pheromones * (1.0 - evaporation), hyperparams);
    }

    pub fn add(&mut self, to: VertexId, add: f32, hyperparams: &HyperParams) {
        let pheromones = self.pheromones[to as usize];
        self.set_pheromones(to, pheromones + add, hyperparams);
    }

    pub fn set_pheromones(&mut self, to: VertexId, pheromones: f32,
                          hyperparams: &HyperParams) {
        let pheromones = pheromones.clamp(hyperparams.min_pheromones,
                                          hyperparams.max_pheromones);
        self.pheromones[to as usize] = pheromones;
        self.update_weights(to);
    }

    pub fn weights(&self, tick_offset: u8) -> &EdgeWeights {
        &self.offset_trail_weights[tick_offset as usize]
    }

    fn update_weights(&mut self, to: VertexId) {
        for offset in 0..TICK_OFFSETS {
            let pheromones = self.pheromones[to as usize];
            let eta_pow = self.eta_pows[offset][to as usize];
            self.offset_trail_weights[offset][to as usize] = pheromones * eta_pow;
        }
    }
}

impl Colony {
    pub fn new(graph: &Arc<SimpleGraph>, hyperparams: HyperParams) -> Self {
        let trails = (0..graph.ports.len())
            .map(|from| VertexTrails::new(from as VertexId, &graph, &hyperparams))
            .collect();
        Colony {
            graph: graph.clone(),
            rng: SmallRng::seed_from_u64(hyperparams.seed),
            global_best: None,
            trails,
            hyperparams,
        }
    }

    pub fn run(&mut self) -> Solution {
        for _ in 0..self.hyperparams.iterations {
            self.run_iteration();
        }
        let best_ant = self.global_best.as_ref().unwrap();
        best_ant.to_solution(&self.graph)
    }

    fn run_iteration(&mut self) {
        self.construct_solutions();
        self.update_trails();
    }

    fn construct_solutions(&mut self) {
        let ants: Vec<Ant> = iter::repeat(()).take(self.hyperparams.ants)
            .map(|_| self.construct_solution()).collect();
        let local_best = ants.iter().max_by_key(|ant| ant.score).cloned().unwrap();
        if self.global_best.is_none() || local_best.score > self.global_best.as_ref().unwrap().score {
            self.global_best = Some(local_best.clone());
        }
    }
    
    fn construct_solution(&mut self) -> Ant {
        let mut ant = Ant::new();
        let start = self.rng.gen_range(0..self.graph.ports.len());
        ant.reset(start as VertexId, self.graph.start_tick);
        while let Some(to) = self.sample_option(&ant) {
            // Local trail update
            let pheromone_add = self.hyperparams.local_evaporation_rate *
                self.hyperparams.min_pheromones;
            self.trails[ant.current as usize].evaporate_add(
                to, self.hyperparams.local_evaporation_rate, pheromone_add,
                &self.hyperparams);
            ant.visit(to, &self.graph);
        }
        ant.finalize_path(&self.graph);
        ant
    }

    fn sample_option(&mut self, ant: &Ant) -> Option<VertexId> {
        let tick_offset = self.graph.tick_offset(ant.tick);
        let weights = self.trails[ant.current as usize].weights(tick_offset)
            .iter().enumerate()
            .map(|(to, &w)| {
                let to = to as VertexId;
                let cost = self.graph.cost(tick_offset, ant.current, to);
                if ant.valid_option(cost, to, &self.graph) {
                    w
                } else {
                    0f32
                }
            });
        let rand_valid_option = |rng: &mut SmallRng| -> Option<VertexId> {
            let all_options: Vec<VertexId> = (0..self.graph.ports.len())
                .map(|v| v as VertexId)
                .filter(|&i| {
                    let cost = self.graph.cost(tick_offset, ant.current, i);
                    ant.valid_option(cost, i, &self.graph)
                }).collect();
            if all_options.is_empty() {
                None
            } else {
                Some(all_options[rng.gen_range(0..all_options.len())])
            }
        };
        if self.rng.gen::<f32>() < self.hyperparams.exploitation_probability {
            // Greedy exploitation
            let to = weights.enumerate()
                .max_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx).unwrap() as VertexId;
            let cost = self.graph.cost(tick_offset, ant.current, to);
            if ant.valid_option(cost, to, &self.graph) {
                Some(to)
            } else {
                rand_valid_option(&mut self.rng)
            }
        } else if let Ok(distribution) = WeightedIndex::new(weights) {
            let option = distribution.sample(&mut self.rng);
            Some(option as VertexId)
        } else {
            rand_valid_option(&mut self.rng)
        }
    }

    fn update_trails(&mut self) {
        // Decay existing trails
        for from in 0..self.graph.ports.len() {
            let from = from as VertexId;
            for to in 0..self.graph.ports.len() {
                let to = to as VertexId;
                if from == to {
                    continue;
                }
                self.trails[from as usize].evaporate(
                    to, self.hyperparams.evaporation_rate, &self.hyperparams);
            }
        }

        // Global trail update
        let best = self.global_best.as_ref().unwrap();
        let add = 1.0 / (best.tick as f32);
        let pheromone_add = self.hyperparams.evaporation_rate * add;
        let mut from = best.start;
        let best_path = best.path.clone();
        for to in best_path {
            self.trails[from as usize].add(to, pheromone_add, &self.hyperparams);
            from = to;
        }
    }
}

pub struct SimpleAntColonyOptimizationSolver {
    pub hyperparams: HyperParams,
}

impl SimpleAntColonyOptimizationSolver {
    pub fn do_solve(&mut self, graph: &Arc<SimpleGraph>) -> Option<Solution> {
        let mut colony = Colony::new(graph, self.hyperparams.clone());
        Some(colony.run())
    }
}

impl Default for SimpleAntColonyOptimizationSolver {
    fn default() -> Self {
        let hyperparams = AntColonyOptimizationSolver::default().hyperparams;
        SimpleAntColonyOptimizationSolver { hyperparams }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value};
    use std::fs;
    use crate::game_interface::{GameTick};
    use crate::graph::Graph;
    use crate::solvers::{Solver};
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
    fn test_match_ant_colony_optimization() {
        let game = make_game();
        let slow_graph = Arc::new(SimpleGraph::new(&game));
        let fast_graph = Arc::new(Graph::new(&game));
        let mut slow = SimpleAntColonyOptimizationSolver::default();
        let mut fast = AntColonyOptimizationSolver::default();
        let slow_sln = slow.do_solve(&slow_graph).unwrap();
        let fast_sln = fast.do_solve(&fast_graph).unwrap();
        println!("slow: spawn {:?} path {:?} score {}",
                 slow_sln.spawn, slow_sln.paths, slow_sln.score);
        println!("fast: spawn {:?} path {:?} score {}",
                 fast_sln.spawn, fast_sln.paths, fast_sln.score);
        assert_eq!(slow_sln.score, fast_sln.score);
    }
}
