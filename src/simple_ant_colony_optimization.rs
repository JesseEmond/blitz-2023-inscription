// Implementation of ant_colony_optimization.rs, without speed optimizations.

use rand::{Rng, SeedableRng};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::SmallRng;
use std::cmp::Ordering;
use std::collections::{HashSet};
use std::iter;
use std::sync::Arc;

use crate::ant_colony_optimization::{HyperParams};
use crate::challenge::{Solution, eval_score};
use crate::simple_graph::{SimpleGraph, VertexId};
use crate::solvers::{AntColonyOptimizationSolver};

/// Same as HyperParams, but with 'alpha' added (not included in final version).
#[derive(Clone)]
pub struct ExtendedHyperParams {
    pub base: HyperParams,
    /// Alpha parameter, power applied to the pheromones when computing weights.
    pub pheromones_power: f32,
}

#[derive(Clone)]
pub struct Ant {
    pub start: VertexId,
    pub current: VertexId,
    pub path: Vec<VertexId>,
    pub tick: u16,
    pub score: i32,
    pub seen: HashSet<VertexId>,
}

pub struct Colony {
    pub hyperparams: ExtendedHyperParams,
    pub graph: Arc<SimpleGraph>,

    // pheromones[from][to]
    pheromones: Vec<Vec<f32>>,
    global_best: Option<Ant>,
    rng: SmallRng,
}

impl Ant {
    pub fn new() -> Self {
        Ant {
            start: 0,
            current: 0,
            tick: 0,
            path: Vec::new(),
            score: 0,
            seen: HashSet::new(),
        }
    }

    pub fn reset(&mut self, start: VertexId, tick: u16) {
        self.start = start;
        self.current = start;
        self.tick = tick + 1;  // time to dock our starting port
        self.path.clear();
        self.seen.clear();
        self.seen.insert(start);
    }

    pub fn visit(&mut self, port: VertexId, graph: &SimpleGraph) {
        self.path.push(port);
        let cost = graph.cost(graph.tick_offset(self.tick), self.current, port);
        self.current = port;
        let dock_cost = if port == self.start { 0 } else { 1 };
        self.tick += (cost + dock_cost) as u16;
        self.score = self.compute_score(graph, /*simulate_to_end=*/false);
        assert!(port == self.start || !self.seen.contains(&port));
        self.seen.insert(port);
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
        let have_time = self.tick + (edge_cost as u16) + 1 < graph.max_ticks;
        have_time && !self.seen.contains(&to)
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
                self.seen.remove(to);
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

impl Colony {
    pub fn new(graph: &Arc<SimpleGraph>,
               hyperparams: ExtendedHyperParams) -> Self {
        let base_pheromones = hyperparams.base.pheromones_init_ratio
            * (hyperparams.base.max_pheromones - hyperparams.base.min_pheromones)
            + hyperparams.base.min_pheromones;
        let pheromones = vec![
            vec![base_pheromones; graph.ports.len()];
            graph.ports.len()];
        Colony {
            graph: graph.clone(),
            rng: SmallRng::seed_from_u64(hyperparams.base.seed),
            global_best: None,
            hyperparams,
            pheromones,
        }
    }

    pub fn run(&mut self) -> Solution {
        for _ in 0..self.hyperparams.base.iterations {
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
        let ants: Vec<Ant> = iter::repeat(()).take(self.hyperparams.base.ants)
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
            let pheromone_add = self.hyperparams.base.local_evaporation_rate *
                self.hyperparams.base.min_pheromones;
            self.evaporate_add_pheromones(
                ant.current, to, self.hyperparams.base.local_evaporation_rate,
                pheromone_add);
            ant.visit(to, &self.graph);
        }
        ant.finalize_path(&self.graph);
        ant
    }

    fn sample_option(&mut self, ant: &Ant) -> Option<VertexId> {
        let options: Vec<VertexId> = (0..self.graph.ports.len())
            .map(|v| v as VertexId)
            .filter(|&to| {
                let cost = self.graph.cost(self.graph.tick_offset(ant.tick),
                                           ant.current, to);
                to != ant.current && ant.valid_option(cost, to, &self.graph)
            }).collect();
        let weights: Vec<f32> = options.iter().map(|&to| {
            let alpha = self.hyperparams.pheromones_power;
            let beta = self.hyperparams.base.heuristic_power;
            let distance = self.graph.cost(
                self.graph.tick_offset(ant.tick), ant.current, to);
            let eta = 1.0 / (distance as f32);
            let pheromones = self.pheromones[ant.current as usize][to as usize];
            pheromones.powf(alpha) * eta.powf(beta)
        }).collect();
        if options.is_empty() {
            None
        } else if self.rng.gen::<f32>() < self.hyperparams.base.exploitation_probability {
            let idx = weights.iter().enumerate()
                .max_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx).unwrap();
            Some(options[idx])
        } else if let Ok(distribution) = WeightedIndex::new(weights) {
            Some(options[distribution.sample(&mut self.rng)])
        } else {
            Some(options[self.rng.gen_range(0..options.len())])
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
                self.evaporate_pheromones(
                    from, to, self.hyperparams.base.evaporation_rate);
            }
        }

        // Global trail update
        let best = self.global_best.as_ref().unwrap();
        let add = 1.0 / (best.tick as f32);
        let pheromone_add = self.hyperparams.base.evaporation_rate * add;
        let mut from = best.start;
        let best_path = best.path.clone();
        for to in best_path {
            self.add_pheromones(from, to, pheromone_add);
            from = to;
        }
    }

    fn evaporate_add_pheromones(
        &mut self, from: VertexId, to: VertexId, evaporation: f32, add: f32) {
        let pheromones = self.pheromones[from as usize][to as usize];
        self.set_pheromones(
            from, to, (1.0 - evaporation) * pheromones + evaporation * add);
    }

    fn evaporate_pheromones(&mut self, from: VertexId, to: VertexId,
                            evaporation: f32) {
        let pheromones = self.pheromones[from as usize][to as usize];
        self.set_pheromones(from, to, pheromones * (1.0 - evaporation));
    }

    fn add_pheromones(&mut self, from: VertexId, to: VertexId, add: f32) {
        let pheromones = self.pheromones[from as usize][to as usize];
        self.set_pheromones(from, to, pheromones + add);
    }

    fn set_pheromones(&mut self, from: VertexId, to: VertexId, pheromones: f32) {
        let pheromones = pheromones.clamp(self.hyperparams.base.min_pheromones,
                                          self.hyperparams.base.max_pheromones);
        self.pheromones[from as usize][to as usize] = pheromones;
    }
}

pub struct SimpleAntColonyOptimizationSolver {
    pub hyperparams: ExtendedHyperParams,
}

impl SimpleAntColonyOptimizationSolver {
    pub fn do_solve(&mut self, graph: &Arc<SimpleGraph>) -> Option<Solution> {
        let mut colony = Colony::new(graph, self.hyperparams.clone());
        Some(colony.run())
    }
}

impl Default for SimpleAntColonyOptimizationSolver {
    fn default() -> Self {
        let base = AntColonyOptimizationSolver::default().hyperparams;
        SimpleAntColonyOptimizationSolver {
            hyperparams: ExtendedHyperParams { base, pheromones_power: 1.0 }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value};
    use crate::game_interface::{GameTick};
    use crate::graph::Graph;
    use crate::solvers::{Solver};
    use super::*;

    fn make_game() -> Arc<GameTick> {
        let game_json = include_str!("../games/35334.json");
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
