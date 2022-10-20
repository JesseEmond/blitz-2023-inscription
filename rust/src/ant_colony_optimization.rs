use log::{info};
use std::iter;
use rand::{Rng, SeedableRng};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::SmallRng;

use crate::game_interface::{eval_score};
use crate::graph::{EdgeId, Graph, VertexId};
use crate::pathfinding::{Path, Pos};

// Based on the documentation at:
// https://staff.washington.edu/paymana/swarm/stutzle99-eaecs.pdf
// http://www.scholarpedia.org/article/Ant_colony_optimization

// TODO: to consider trying:
// - Compare to AS
// - MMAS (min-max ant system): bounds on pheromone, init to max
// - Update pheromones using global best
// - Update pheromones using global best, on a schedule
// - Rank-based version of any-system?
// - Include local search before updates?
// - Update both directions of edge?
// - Update only trails for specific tick offset?

pub struct HyperParams {
    // Number of rounds of ant simulations to do.
    pub iterations: usize,

    // 'm' ants that construct solutions at each iteration
    pub ants: usize,

    // 'ρ' (rho) used in pheromone (τ) updates, i.e.
    // τ = (1 - ρ) τ + ρ total_fitness
    pub evaporation_rate: f32,

    // q0 used when choosing actions. With probability q0, the best move is
    // exploited instead of sampling.
    pub exploitation_threshold: f32,

    // 'α' (alpha) used when sampling actions, applied to the pheromone τ: τ^α
    pub pheromone_trail_power: f32,

    // 'β' (beta) used when sampling actions, applied to the heuristic η: η^β
    pub heuristic_power: f32,

    // 'τ0' base pheromones for local updates, in the following formula:
    // τ = (1 − ξ) · τ + ξ τ0
    pub base_pheromones: f32,
    // 'ξ' evaporation rate for local updates, in the following formula:
    // τ = (1 − ξ) · τ + ξ τ0
    pub local_evaporation_rate: f32,
}

#[derive(Clone)]
pub struct Ant {
    pub start: VertexId,
    pub edges: Vec<EdgeId>,
    pub tick: u32,
    pub score: i32,
    pub unseen: Vec<VertexId>,
}

#[derive(Clone)]
pub struct Trail {
    // τ, pheromone strength of this tail.
    pub pheromone: f32,
}

pub struct Colony {
    pub hyperparams: HyperParams,
    pub graph: Graph,
    // One trail for each edge.
    pub edge_trails: Vec<Trail>,
    pub global_best: Option<Ant>,
    pub rng: SmallRng
}

pub struct Solution {
    pub score: i32,
    pub spawn: Pos,
    pub paths: Vec<Path>,
}

impl HyperParams {
    pub fn default_params(iterations: usize) -> Self {
        // Some reasonable default params, based on section 1.5.2 of
        // https://staff.washington.edu/paymana/swarm/stutzle99-eaecs.pdf
        HyperParams {
            iterations: iterations,
            ants: 25,
            evaporation_rate: 0.2,
            exploitation_threshold: 0.0,  // Note: not in the MMAS example used
            pheromone_trail_power: 1.0,
            heuristic_power: 2.0,
            base_pheromones: 0.0,  // Unused for MMAS setup
            local_evaporation_rate: 0.0,  // No local evaporation
        }
    }
}

impl Ant {
    pub fn new() -> Self {
        Ant {
            start: 0,
            tick: 0,
            edges: Vec::new(),
            score: 0,
            unseen: Vec::new(),
        }
    }

    pub fn reset(&mut self, start: VertexId, tick: u32, graph: &Graph) {
        self.start = start;
        self.tick = tick + 1;  // Time to dock our start
        self.edges.clear();
        self.unseen = (0..graph.vertices.len()).filter(|&v| v != start).collect();
    }

    fn visit(&mut self, edge_id: EdgeId, graph: &Graph) {
        self.edges.push(edge_id);
        let edge = graph.edge(edge_id);
        let cost = edge.path(self.tick).cost;
        self.tick += cost + 1;  // +1 to dock
        self.score = self.compute_score(graph, /*simulate_to_end=*/false);
        let unseen_idx = self.unseen.iter().position(|v| *v == edge.to);
        if let Some(unseen_idx) = unseen_idx {
            self.unseen.swap_remove(unseen_idx);
        } else {
            assert!(edge.to == self.start,
                    "visiting vertex not tagged as unseen (and not going home)");
        }
    }

    fn compute_score(&self, graph: &Graph, simulate_to_end: bool) -> i32 {
        if self.edges.is_empty() {
            return 0;
        }
        let current = self.current_vertex(graph);
        let looped = current == self.start;
        let visits = self.edges.len() + 1;  // +1 for spawn
        let tick = if simulate_to_end && !looped { graph.max_ticks } else { self.tick };
        eval_score(visits as u32, tick, looped)
    }

    // Consider our score if we kept only our first 'num_edges' edges, plus a
    // trip back home, if there's enough ticks to do so.
    fn hypothetical_home_score(
        &self, graph: &Graph, num_edges: usize
        ) -> Option<i32> {
        let (vertex, mut tick) = self.simulate_to_num_edges(graph, num_edges);
        let edge_go_home = graph.vertex_edge_to(vertex, self.start)
            .expect("No path home..?");
        let cost_go_home = graph.edge(edge_go_home).path(tick).cost;
        tick += cost_go_home + 1; // +1 to dock it
        if tick < graph.max_ticks {
            let visits = num_edges + 2;  // +1 for spawn, +1 for last
            Some(eval_score(visits as u32, tick, /*looped=*/true))
        } else {
            None
        }
    }

    // Helper to get the final (vertex, tick) if we were to only use 'num_edges'
    // of the ant's stored edges.
    fn simulate_to_num_edges(
        &self, graph: &Graph, num_edges: usize
        ) -> (VertexId, u32) {
        let mut tick = graph.start_tick + 1;  // +1 to dock initially
        let mut vertex = self.start;
        for edge_id in &self.edges[..num_edges] {
            let cost = graph.edge(*edge_id).path(tick).cost;
            tick += cost + 1;  // +1 to dock it
            vertex = graph.edge(*edge_id).to;
        }
        (vertex, tick)
    }

    fn current_vertex(&self, graph: &Graph) -> VertexId {
        if self.edges.is_empty() {
            self.start
        } else {
            graph.edge(*self.edges.last().unwrap()).to
        }
    }

    fn options(&self, graph: &Graph) -> Vec<EdgeId> {
        graph.vertex(self.current_vertex(graph)).edges.iter()
            .filter(|&edge_id| {
                let edge = graph.edge(*edge_id);
                let cost = edge.path(self.tick).cost;
                self.unseen.contains(&edge.to) && self.tick + cost + 1 < graph.max_ticks
            }).cloned().collect()
    }

    // Add a path back home to our path, if we should, potentially truncating.
    fn finalize_path(&mut self, graph: &Graph) {
        let mut best_score = self.compute_score(graph, /*simulate_to_end=*/true);
        let mut go_home_at_index: Option<usize> = None;
        for num_edges in 1..=self.edges.len() {
            let score = self.hypothetical_home_score(graph, num_edges);
            if !score.is_none() && score.unwrap() > best_score {
                best_score = score.unwrap();
                go_home_at_index = Some(num_edges);  // insert after num_edges
            }
        }
        if let Some(go_home_index) = go_home_at_index {
            let (vertex, tick) = self.simulate_to_num_edges(graph, go_home_index);
            self.unseen.extend_from_slice(&self.edges[go_home_index..]);
            self.edges.truncate(go_home_index);
            self.tick = tick;
            let edge_go_home = graph.vertex_edge_to(vertex, self.start)
                .expect("No path home..?");
            self.visit(edge_go_home, graph);
            assert!(self.compute_score(graph, /*simulate_to_end=*/true) == best_score);
        }
    }
}

impl Trail {
    pub fn new(pheromone: f32) -> Self {
        Trail { pheromone: pheromone }
    }

    pub fn add(&mut self, pheromone: f32) {
        self.pheromone += pheromone;
    }

    pub fn evaporate(&mut self, evaporation_rate: f32) {
        self.pheromone *= 1.0 - evaporation_rate;
    }
}

impl Colony {
    pub fn new(graph: Graph, hyperparams: HyperParams, seed: u64) -> Self {
        // TODO: change min if doing MMAS
        let edge_trails = vec![Trail::new(0.0); graph.edges.len()];
        Colony {
            graph: graph,
            edge_trails: edge_trails,
            global_best: None,
            hyperparams: hyperparams,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    pub fn run(&mut self) -> Solution {
        for iter in 0..self.hyperparams.iterations {
            info!("ACO iteration #{iter}/{total}", iter = iter + 1,
                   total = self.hyperparams.iterations);
            self.run_iteration();
            let best_ant = self.global_best.as_ref().expect("No solution found...?");
            info!("  best global score: {best}", best = best_ant.score);
        }
        let best_ant = self.global_best.as_ref().expect("No solution found...?");
        Solution::from_ant(&best_ant, &self.graph)
    }

    fn run_iteration(&mut self) {
        self.construct_solutions();
        // TODO: local search?
        self.update_trails();
    }

    fn construct_solutions(&mut self) -> Vec<Ant> {
        let ants: Vec<Ant> = iter::repeat(()).take(self.hyperparams.ants)
            .map(|_| self.construct_solution()).collect();
        let local_best = ants.iter().max_by_key(|ant| ant.score).cloned().unwrap();
        info!("  local best score: {score}", score = local_best.score);
        // Note: include '=' too so that the best can vary for equal scores.
        if self.global_best.is_none()
            || local_best.score >= self.global_best.as_ref().unwrap().score {
            self.global_best = Some(local_best);
        }
        ants
    }

    fn pick_option(&mut self, ant: &Ant) -> Option<EdgeId> {
        let options = ant.options(&self.graph);
        if !options.is_empty() {
            Some(self.sample_option(options, ant.tick))
        } else {
            None
        }
    }

    fn sample_option(&mut self, options: Vec<EdgeId>, tick: u32) -> EdgeId {
        let weights: Vec<f32> = options.iter().map(|&edge_id| {
            let distance = self.graph.edge(edge_id).path(tick).cost;
            let alpha = self.hyperparams.pheromone_trail_power;
            let beta = self.hyperparams.heuristic_power;
            let tau = self.edge_trails[edge_id].pheromone;
            let eta = 1.0 / (distance as f32);
            tau.powf(alpha) * eta.powf(beta)
        }).collect();
        if weights.iter().sum::<f32>() == 0f32 {
            let closest = options.iter().min_by_key(
                |&e| self.graph.edge(*e).path(tick).cost).cloned().unwrap();
            return closest;
        }
        // TODO: sometimes take greedy one (based on exploitation_threshold)
        let distribution = WeightedIndex::new(&weights).unwrap();
        options[distribution.sample(&mut self.rng)]
    }

    fn construct_solution(&mut self) -> Ant {
        let mut ant = Ant::new();
        let start = self.rng.gen_range(0..self.graph.vertices.len());
        ant.reset(start, self.graph.start_tick, &self.graph);
        while let Some(edge_id) = self.pick_option(&ant) {
            // Local trail update
            self.edge_trails[edge_id].evaporate(self.hyperparams.local_evaporation_rate);
            let pheromone_add = self.hyperparams.local_evaporation_rate
                * self.hyperparams.base_pheromones;
            self.edge_trails[edge_id].add(pheromone_add);
            // TODO: also update other direction?
            ant.visit(edge_id, &self.graph);
        }
        ant.finalize_path(&self.graph);
        ant
    }
    
    fn update_trails(&mut self) {
        // Decay all existing trails
        for trail in &mut self.edge_trails {
            trail.evaporate(self.hyperparams.evaporation_rate);
        }

        // Global trail update
        // TODO: sometimes pick local best?
        let best = self.global_best.as_ref().expect("No global best at update time!");
        let pheromone_add = self.hyperparams.evaporation_rate * (best.score as f32);
        for edge_id in &best.edges {
            self.edge_trails[*edge_id].add(pheromone_add);
            // TODO: also update other direction?
        }
    }
}

impl Solution {
    pub fn from_ant(ant: &Ant, graph: &Graph) -> Self {
        let mut repeat_ant = Ant::new();
        repeat_ant.reset(ant.start, graph.start_tick, graph);
        let mut paths = Vec::new();
        for edge_id in &ant.edges {
            paths.push(graph.edge(*edge_id).path(repeat_ant.tick).clone());
            repeat_ant.visit(*edge_id, graph);
        }
        Solution {
            score: ant.score,
            spawn: graph.vertex(ant.start).position,
            paths: paths,
        }
    }
}
