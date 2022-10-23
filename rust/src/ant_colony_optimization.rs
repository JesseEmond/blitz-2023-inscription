use arrayvec::ArrayVec;
use log::{debug, info};
use serde::{Deserialize};
use std::cmp::Ordering;
use std::iter;
use rand::{Rng, SeedableRng};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::SmallRng;

use crate::game_interface::{eval_score};
use crate::graph::{EdgeId, Graph, VertexId, MAX_VERTEX_EDGES, MAX_TICK_OFFSETS, MAX_VERTICES};
use crate::pathfinding::{Path, Pos};

// Based on the documentation at:
// https://staff.washington.edu/paymana/swarm/stutzle99-eaecs.pdf
// http://www.scholarpedia.org/article/Ant_colony_optimization

// TODO: to consider trying:
// - Compare to AS
// - MMAS (min-max ant system): bounds on pheromone, init to max
// - Update pheromones using local best
// - Update pheromones using global best, on a schedule
// - Pheromone at the edge-tick_offset level
// - Update best when score is equal
// - Rank-based version of any-system?
// - Include local search before updates?
// - Update both directions of edge?
// - Update only trails for specific tick offset?

#[derive(Deserialize, Debug)]
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
    pub exploitation_probability: f32,

    // NOTE: we don't support 'α' (alpha) as pheromone trail exponent to speed
    // up computation. Instead, beta can be swept separately.

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
    pub edges: ArrayVec<EdgeId, MAX_VERTICES>,
    pub tick: u16,
    pub tick_offset: u8,  // offset in the tide schedule, for optimization.
    pub score: i32,
    pub seen: u64,  // mask of seen vertices
}

type EtaPows = ArrayVec<ArrayVec<f32, MAX_VERTEX_EDGES>, MAX_TICK_OFFSETS>;
type Costs = ArrayVec::<ArrayVec::<u16, MAX_VERTEX_EDGES>, MAX_TICK_OFFSETS>;
type EdgeWeights = ArrayVec<f32, MAX_VERTEX_EDGES>;

// Holds the trails coming out of a vertex, used for optimization purposes to
// precompute Vecs of weights.
pub struct VertexTrails {
    // TODO move data together..?

    // τ, pheromone strength of each edge.
    pub pheromones: ArrayVec<f32, MAX_VERTEX_EDGES>,

    // For a given tick offset, a list of pre-computed weights, one for each
    // edge.Used in sampling.
    pub offset_trail_weights: ArrayVec<EdgeWeights, MAX_TICK_OFFSETS>,
    // Pre-computed per-edge eta^beta, for each tick offset.
    pub eta_pows: EtaPows,

    // For a given offset, cost of the edge path. Used for faster lookups.
    costs: Costs,
    // Edges go where, for faster lookup.
    goes_to: ArrayVec<VertexId, MAX_VERTEX_EDGES>,
}

pub struct Colony {
    pub hyperparams: HyperParams,
    pub graph: Graph,
    // Trails for each vertex.
    pub trails: Vec<VertexTrails>,
    pub global_best: Option<Ant>,
    pub rng: SmallRng,
}

pub struct Solution {
    pub score: i32,
    pub spawn: Pos,
    pub paths: Vec<Path>,
}

impl HyperParams {
    pub fn default_params(iterations: usize) -> Self {
        HyperParams {
            iterations,
            ants: 25,
            evaporation_rate: 0.2,
            exploitation_probability: 0.1,
            heuristic_power: 3.0,
            base_pheromones: 0.01,
            local_evaporation_rate: 0.01,
        }
    }
}

impl Ant {
    pub fn new() -> Self {
        Ant {
            start: 0,
            tick: 0,
            tick_offset: 0,
            edges: ArrayVec::new(),
            score: 0,
            seen: 0,
        }
    }

    pub fn reset(&mut self, start: VertexId, tick: u16, tick_offset: u8) {
        self.start = start;
        self.tick = tick + 1;  // Time to dock our start
        self.tick_offset = tick_offset;
        self.edges.clear();
        assert!(start < 64);
        self.seen = 1u64 << start;
    }

    fn visit(&mut self, edge_id: EdgeId, graph: &Graph) {
        self.edges.push(edge_id);
        let edge = graph.edge(edge_id);
        let cost = edge.path(self.tick).cost;
        self.tick += cost + 1;  // +1 to dock
        self.tick_offset = (self.tick % graph.tick_offsets as u16) as u8;
        self.score = self.compute_score(graph, /*simulate_to_end=*/false);
        if self.seen & (1u64 << edge.to) != 0 {
            assert!(edge.to == self.start,
                    "visiting vertex not tagged as unseen (and not going home)");
        }
        self.seen |= 1u64 << edge.to;
    }

    fn compute_score(&self, graph: &Graph, simulate_to_end: bool) -> i32 {
        if self.edges.is_empty() {
            return 0;
        }
        let current = self.current_vertex(graph);
        let looped = current == self.start;
        let visits = self.edges.len() + 1;  // +1 for spawn
        let tick = if simulate_to_end && !looped {
            graph.max_ticks
        } else if looped {
            self.tick - 1  // last docking home tick doesn't count
        } else {
            self.tick
        };
        eval_score(visits as u32, tick, looped)
    }

    // Consider our score if we kept only our first 'num_edges' edges, plus a
    // trip back home, if there's enough ticks to do so.
    fn hypothetical_home_score(
        &self, graph: &Graph, num_edges: usize
        ) -> Option<i32> {
        let (vertex, mut tick) = self.simulate_to_num_edges(graph, num_edges);
        let edge_go_home = graph.vertex_edge_to(vertex, self.start).expect("No path home..?");
        let cost_go_home = graph.edge(edge_go_home).path(tick).cost;
        tick += cost_go_home; // no +1, last home docking doesn't count.
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
        ) -> (VertexId, u16) {
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

    fn valid_option(&self, edge_cost: u16, to_vertex_id: VertexId, graph: &Graph) -> bool {
        let seen = (self.seen & (1u64 << to_vertex_id)) != 0;
        self.tick + edge_cost + 1 < graph.max_ticks && !seen
    }

    // Add a path back home to our path, if we should, potentially truncating.
    fn finalize_path(&mut self, graph: &Graph) {
        let mut best_score = self.compute_score(graph, /*simulate_to_end=*/true);
        let mut go_home_at_index: Option<usize> = None;
        for num_edges in 1..=self.edges.len() {
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
            for edge_id in &self.edges[go_home_index..] {
                let edge = graph.edge(*edge_id);
                self.seen ^= 1u64 << edge.to;
            }
            self.edges.truncate(go_home_index);
            self.tick = tick;
            let edge_go_home = graph.vertex_edge_to(vertex, self.start)
                .expect("No path home..?");
            self.visit(edge_go_home, graph);
            assert!(self.compute_score(graph, /*simulate_to_end=*/true) == best_score);
        }
    }
}

impl VertexTrails {
    pub fn new(vertex_id: VertexId, graph: &Graph, hyperparams: &HyperParams) -> Self {
        let vertex = graph.vertex(vertex_id);
        let eta_pows: EtaPows = (0..graph.tick_offsets).map(|offset| {
            vertex.edges.iter().map(|&edge_id| {
                let distance = graph.edge(edge_id).path(offset as u16).cost;
                let beta = hyperparams.heuristic_power;
                let eta = 1.0 / (distance as f32);
                eta.powf(beta)
            }).collect()
        }).collect();
        let costs: Costs = (0..graph.tick_offsets).map(|offset| {
            vertex.edges.iter().map(|&edge_id| {
                graph.edge(edge_id).path(offset as u16).cost
            }).collect()
        }).collect();

        VertexTrails {
            pheromones: ArrayVec::from_iter(vec![hyperparams.base_pheromones; vertex.edges.len()]),
            offset_trail_weights: ArrayVec::from_iter(vec![ArrayVec::from_iter(vec![1.0; vertex.edges.len()]); graph.tick_offsets]),
            eta_pows,
            costs,
            goes_to: vertex.edges.iter().map(|&edge_id| graph.edge(edge_id).to).collect(),
        }
    }

    pub fn evaporate_add(&mut self, option_idx: usize,
                         evaporation: f32, pheromone: f32) {
        self.pheromones[option_idx] = (1.0 - evaporation * self.pheromones[option_idx])
            + evaporation * pheromone;
        self.update_weights(option_idx);
    }

    pub fn add(&mut self, option_idx: usize, pheromone: f32) {
        self.pheromones[option_idx] += pheromone;
        self.update_weights(option_idx);
    }

    pub fn evaporate(&mut self, option_idx: usize, evaporation_rate: f32) {
        self.pheromones[option_idx] *= 1.0 - evaporation_rate;
        self.update_weights(option_idx);
    }

    pub fn update_weights(&mut self, option_idx: usize) {
        for offset in 0..self.offset_trail_weights.len() {
            let tau = self.pheromones[option_idx];
            self.offset_trail_weights[offset][option_idx] = tau * self.eta_pow(option_idx, offset as u16);
        }
    }

    pub fn weights(&self, tick: u16) -> &EdgeWeights {
        let offset = (tick as usize) % self.offset_trail_weights.len();
        unsafe {
            self.offset_trail_weights.get_unchecked(offset)
        }
    }

    pub fn cost(&self, tick_offset: u8, edge_idx: u8) -> u16 {
        unsafe {
            *self.costs.get_unchecked(tick_offset as usize).get_unchecked(edge_idx as usize)
        }
    }

    fn eta_pow(&self, option_idx: usize, tick: u16) -> f32 {
        let offset = (tick as usize) % self.eta_pows.len();
        self.eta_pows[offset][option_idx]
    }
}

impl Colony {
    pub fn new(graph: Graph, hyperparams: HyperParams, seed: u64) -> Self {
        let vertex_trails = (0..graph.vertices.len()).map(|vertex_id| {
            VertexTrails::new(vertex_id as VertexId, &graph, &hyperparams)
        }).collect();
        Colony {
            graph,
            trails: vertex_trails,
            global_best: None,
            hyperparams,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    pub fn run(&mut self) -> Solution {
        debug_log_graph(&self.graph);
        debug_log_heuristics(&self.graph);
        for iter in 0..self.hyperparams.iterations {
            info!("ACO iteration #{iter}/{total}", iter = iter + 1,
                   total = self.hyperparams.iterations);
            debug_log_pheromones(self, iter);
            debug_log_scores(self, iter);
            self.run_iteration();
            let best_ant = self.global_best.as_ref().expect("No solution found...?");
            info!("  best global score: {best}", best = best_ant.score);
        }
        let best_ant = self.global_best.as_ref().expect("No solution found...?");
        Solution::from_ant(best_ant, &self.graph)
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
        debug_log_ant(&local_best, "[LOGGING_LOCAL_BEST_ANT]");
        for (i, ant) in ants.iter().enumerate() {
            debug_log_ant(ant, format!("[LOGGING_ANT]{i} ").as_str());
        }
        if self.global_best.is_none()
            || local_best.score > self.global_best.as_ref().unwrap().score {
            self.global_best = Some(local_best);
        }
        debug_log_ant(self.global_best.as_ref().unwrap(), "[LOGGING_GLOBAL_BEST_ANT]");
        ants
    }

    fn sample_option(&mut self, ant: &Ant) -> Option<EdgeId> {
        let tick = ant.tick;
        let vertex_id = ant.current_vertex(&self.graph);
        let trail = &self.trails[vertex_id as usize];
        let vertex = &self.graph.vertex(vertex_id);
        let weights = trail.weights(tick).iter().enumerate().map(|(i, w)| {
            let valid = ant.valid_option(trail.cost(ant.tick_offset, i as u8),
                                         trail.goes_to[i], &self.graph);
            w * ((valid as i32) as f32)
        });
        let rand_valid_option = |rng: &mut SmallRng| -> Option<EdgeId> {
            let all_options: Vec<EdgeId> = vertex.edges.iter().filter(|&edge_id| {
                    let edge = &self.graph.edge(*edge_id);
                    ant.valid_option(edge.path(tick).cost, edge.to, &self.graph)
                }).cloned().collect();
            if all_options.is_empty() {
                None
            } else {
                Some(all_options[rng.gen_range(0..all_options.len())])
            }
        };
        if self.rng.gen::<f32>() < self.hyperparams.exploitation_probability {
            // Greedy exploitation
            let idx = weights.enumerate()
                .max_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx).unwrap();
            if ant.valid_option(trail.cost(ant.tick_offset, idx as u8),
                                trail.goes_to[idx], &self.graph) {
                Some(vertex.edges[idx])
            } else {
                rand_valid_option(&mut self.rng)
            }
        } else if let Ok(distribution) = WeightedIndex::new(weights) {
            Some(vertex.edges[distribution.sample(&mut self.rng)])
        } else {
            rand_valid_option(&mut self.rng)
        }
    }

    fn construct_solution(&mut self) -> Ant {
        let mut ant = Ant::new();
        let start = self.rng.gen_range(0..self.graph.vertices.len());
        ant.reset(start as VertexId, self.graph.start_tick, 0);
        while let Some(edge_id) = self.sample_option(&ant) {
            let from = self.graph.edge(edge_id).from;
            let edge_idx = self.graph.vertex(from).edges.iter().position(|&e| e == edge_id).unwrap();
            // Local trail update
            let pheromone_add = self.hyperparams.local_evaporation_rate
                * self.hyperparams.base_pheromones;
            self.trails[from as usize].evaporate_add(edge_idx,
                self.hyperparams.local_evaporation_rate,pheromone_add);
            // TODO: also update other direction?
            ant.visit(edge_id, &self.graph);
        }
        ant.finalize_path(&self.graph);
        ant
    }
    
    fn update_trails(&mut self) {
        // Decay all existing trails
        for vertex_trails in self.trails.iter_mut() {
            for option_idx in 0..vertex_trails.pheromones.len() {
                vertex_trails.evaporate(option_idx, self.hyperparams.evaporation_rate);
            }
        }

        // Global trail update
        // TODO: sometimes pick local best?
        let best = self.global_best.as_ref().expect("No global best at update time!");
        let upper_bound_score = eval_score((self.graph.vertices.len() + 1) as u32,
                                           /*ticks=*/0, /*looped=*/true);
        // TODO: compute pheromone add differently?
        let add = 1.0 / (upper_bound_score + 1 - best.score) as f32;

        let pheromone_add = self.hyperparams.evaporation_rate * add;
        for edge_id in &best.edges {
            let edge = self.graph.edge(*edge_id);
            let vertex = self.graph.vertex(edge.from);
            let edge_idx = vertex.edges.iter().position(|&e| e == *edge_id).unwrap();
            self.trails[edge.from as usize].add(edge_idx, pheromone_add)
            // TODO: also update other direction?
        }
    }
}

impl Solution {
    pub fn from_ant(ant: &Ant, graph: &Graph) -> Self {
        let mut repeat_ant = Ant::new();
        repeat_ant.reset(ant.start, graph.start_tick, 0);
        let mut paths = Vec::new();
        for edge_id in &ant.edges {
            let path_id = graph.edge(*edge_id).path(repeat_ant.tick).path;
            paths.push(graph.paths[path_id as usize].clone());
            repeat_ant.visit(*edge_id, graph);
        }
        Solution {
            score: ant.score,
            spawn: graph.vertex(ant.start).position,
            paths,
        }
    }
}


// All the following are pretty hacky log outputs, optionally parsed to produce
// visualizations.
fn debug_log_graph(graph: &Graph) {
    debug!("[LOGGING_GRAPH_START_TICK]{tick}", tick = graph.start_tick);
    debug!("[LOGGING_GRAPH_MAX_TICK]{tick}", tick = graph.max_ticks);
    for vertex in &graph.vertices {
        debug!("[LOGGING_GRAPH_VERTICES]{x} {y} {edges:?}",
              x = vertex.position.x, y = vertex.position.y,
              edges = vertex.edges);
    }
    for edge in &graph.edges {
        let path_costs: Vec<u16> = edge.paths.iter().map(|p| p.cost).collect();
        debug!("[LOGGING_GRAPH_EDGES]{from} {to} {path_costs:?}",
              from = edge.from, to = edge.to);
    }
}
fn debug_log_pheromones(colony: &Colony, iter: usize) {
    let mut pheromones: Vec<f32> = vec![0.0; colony.graph.edges.len()];
    for (vertex_id, vertex) in colony.graph.vertices.iter().enumerate() {
        for (edge_idx, edge_id) in vertex.edges.iter().enumerate() {
            pheromones[*edge_id as usize] = colony.trails[vertex_id].pheromones[edge_idx];
        }
    }
    debug!("[LOGGING_PHEROMONES]{iter} {pheromones:?}");
}
fn debug_log_ant(ant: &Ant, tag: &str) {
    debug!("{tag}{start} {edges:?} {score}", start = ant.start,
          edges = ant.edges, score = ant.score);
}
fn debug_log_heuristics(graph: &Graph) {
    for edge in &graph.edges {
        let dists: Vec<u16> = edge.paths.iter().map(|p| p.cost).collect();
        let min = 1.0 / (*dists.iter().max().unwrap() as f32);
        let max = 1.0 / (*dists.iter().min().unwrap() as f32);
        debug!("[LOGGING_HEURISTIC]{min} {max}");
    }
}
fn debug_log_scores(colony: &Colony, iter: usize) {
    for (idx, vertex) in colony.graph.vertices.iter().enumerate() {
        for (edge_idx, edge) in vertex.edges.iter().enumerate() {
            let distances: Vec<u16> = colony.graph.edge(*edge).paths.iter().map(|p| p.cost).collect();
            let min_dist = *distances.iter().min().unwrap();
            let max_dist = *distances.iter().max().unwrap();
            let tau = colony.trails[idx].pheromones[edge_idx];
            let beta = colony.hyperparams.heuristic_power;
            let min_eta = 1.0 / (max_dist as f32);
            let max_eta = 1.0 / (min_dist as f32);
            let min_weight = tau * min_eta.powf(beta);
            let max_weight = tau * max_eta.powf(beta);
            debug!("[LOGGING_WEIGHTS]{iter} {idx} {edge} {min_weight} {max_weight}");
        }
    }
}
