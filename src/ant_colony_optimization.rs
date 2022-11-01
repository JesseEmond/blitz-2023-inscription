// TODO: fix for adjacency matrix graph
use arrayvec::ArrayVec;
use log::{debug};
use serde::{Deserialize};
use std::cmp::Ordering;
use std::iter;
use std::sync::Arc;
use rand::{Rng, SeedableRng};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::SmallRng;

use crate::challenge::{Solution, eval_score};
use crate::challenge_consts::{MAX_PORTS, TICK_OFFSETS};
use crate::graph::{Graph, VertexId};

// Based on the documentation at:
// https://staff.washington.edu/paymana/swarm/stutzle99-eaecs.pdf
// http://www.scholarpedia.org/article/Ant_colony_optimization

// TODO: to consider trying:
// - Compare to AS
// - MMAS (min-max ant system): bounds on pheromone, init to max
// - Update pheromones using local best
// - Update pheromones using global best on a schedule
// - Pheromone at the edge-tick_offset level
// - Update best when score is equal? Only sometimes?
// - Rank-based version of any-system?
// - Include other local search before updates?
// - Update both directions of edge?
// - Update only trails for specific tick offset?

#[derive(Deserialize, Debug, Clone)]
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
    // up computation. Instead, we rely on beta being swept (note: not
    // strictly equivalent).

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
    pub current: VertexId,
    pub path: ArrayVec<VertexId, MAX_PORTS>,
    pub tick: u16,
    pub tick_offset: u8,  // offset in the tide schedule, for optimization.
    pub score: i32,
    pub seen: u64,  // mask of seen vertices
}

// [offset][to]
type EtaPows = ArrayVec<ArrayVec<f32, MAX_PORTS>, TICK_OFFSETS>;
// [to]
type EdgeWeights = ArrayVec<f32, MAX_PORTS>;

// Holds the trails coming out of a vertex, used for optimization purposes to
// precompute Vecs of weights.
pub struct VertexTrails {
    pub vertex: VertexId,

    // τ, pheromone strength of each edge.
    pub pheromones: ArrayVec<f32, MAX_PORTS>,

    // For a given tick offset, a list of pre-computed weights, one for each
    // edge. Used in sampling.
    // offset_trail_weights[tick_offset][to]
    pub offset_trail_weights: ArrayVec<EdgeWeights, TICK_OFFSETS>,
    // Pre-computed per-edge eta^beta, for each tick offset.
    // eta_pows[tick_offset][to]
    pub eta_pows: EtaPows,
}

pub struct Colony {
    pub hyperparams: HyperParams,
    pub graph: Arc<Graph>,
    // Trails for each vertex.
    pub trails: Vec<VertexTrails>,
    pub global_best: Option<Ant>,
    pub rng: SmallRng,
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
            current: 0,
            tick: 0,
            tick_offset: 0,
            path: ArrayVec::new(),
            score: 0,
            seen: 0,
        }
    }

    pub fn reset(&mut self, start: VertexId, tick: u16, tick_offset: u8) {
        self.start = start;
        self.current = start;
        self.tick = tick + 1;  // Time to dock our start
        self.tick_offset = tick_offset;
        self.path.clear();
        assert!(start < 64);
        self.seen = 1u64 << start;
    }

    fn visit(&mut self, port: VertexId, graph: &Graph) {
        self.path.push(port);
        self.tick_offset = graph.tick_offset(self.tick);
        let cost = graph.cost(self.tick_offset, self.current, port);
        self.current = port;
        self.tick += (cost + 1) as u16;  // +1 to dock
        self.score = self.compute_score(graph, /*simulate_to_end=*/false);
        if self.seen & (1u64 << port) != 0 {
            assert!(port == self.start,
                    "visiting vertex not tagged as unseen (and not going home)");
        }
        self.seen |= 1u64 << port;
    }

    fn compute_score(&self, graph: &Graph, simulate_to_end: bool) -> i32 {
        if self.path.is_empty() {
            return 0;
        }
        let looped = self.current == self.start;
        let visits = self.path.len() + 1;  // +1 for spawn
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
        let cost_go_home = graph.cost(graph.tick_offset(tick), vertex, self.start);
        tick += cost_go_home as u16; // no +1, last home docking doesn't count.
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
        for to in &self.path[..num_edges] {
            let cost = graph.cost(graph.tick_offset(tick), vertex, *to);
            tick += (cost as u16) + 1;  // +1 to dock it
            vertex = *to;
        }
        (vertex, tick)
    }

    fn valid_option(&self, edge_cost: u8, to_vertex_id: VertexId, graph: &Graph) -> bool {
        let seen = (self.seen & (1u64 << to_vertex_id)) != 0;
        self.tick + (edge_cost as u16) + 1 < graph.max_ticks && !seen
    }

    // // Add a path back home to our path, if we should, potentially truncating.
    fn finalize_path(&mut self, graph: &Graph) {
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
                self.seen ^= 1u64 << (*to as u64);
            }
            self.path.truncate(go_home_index);
            self.tick = tick;
            self.current = vertex;
            self.visit(self.start, graph);
            assert!(self.compute_score(graph, /*simulate_to_end=*/true) == best_score);
        }
    }
}

impl VertexTrails {
    pub fn new(vertex_id: VertexId, graph: &Graph, hyperparams: &HyperParams) -> Self {
        let eta_pows: EtaPows = (0..TICK_OFFSETS).map(|offset| {
            (0..graph.ports.len()).map(|other| {
                let other = other as VertexId;
                if vertex_id != other {
                    let distance = graph.cost(offset as u8, vertex_id, other);
                    let beta = hyperparams.heuristic_power;
                    let eta = 1.0 / (distance as f32);
                    eta.powf(beta)
                } else {
                    0f32
                }
            }).collect()
        }).collect();

        VertexTrails {
            vertex: vertex_id,
            pheromones: ArrayVec::from_iter(vec![hyperparams.base_pheromones; graph.ports.len()]),
            offset_trail_weights: ArrayVec::from_iter(
                vec![ArrayVec::from_iter(vec![1.0; graph.ports.len()]); TICK_OFFSETS]),
            eta_pows,
        }
    }

    pub fn evaporate_add(&mut self, to: VertexId, evaporation: f32,
                         pheromone: f32) {
        self.pheromones[to as usize] = (1.0 - evaporation * self.pheromones[to as usize])
            + evaporation * pheromone;
        self.update_weights(to);
    }

    pub fn add(&mut self, to: VertexId, pheromone: f32) {
        self.pheromones[to as usize] += pheromone;
        self.update_weights(to);
    }

    pub fn evaporate(&mut self, to: VertexId, evaporation_rate: f32) {
        self.pheromones[to as usize] *= 1.0 - evaporation_rate;
        self.update_weights(to);
    }

    pub fn update_weights(&mut self, to: VertexId) {
        for offset in 0..self.offset_trail_weights.len() {
            let tau = self.pheromones[to as usize];
            self.offset_trail_weights[offset][to as usize] = tau * self.eta_pow(offset as u8, to);
        }
    }

    pub fn weights(&self, tick: u16) -> &EdgeWeights {
        let offset = (tick as usize) % TICK_OFFSETS;
        unsafe {
            self.offset_trail_weights.get_unchecked(offset)
        }
    }

    fn eta_pow(&self, tick_offset: u8, to: VertexId) -> f32 {
        unsafe {
            *self.eta_pows.get_unchecked(tick_offset as usize).get_unchecked(to as usize)
        }
    }
}

impl Colony {
    pub fn new(graph: &Arc<Graph>, hyperparams: HyperParams, seed: u64) -> Self {
        let vertex_trails = (0..graph.ports.len()).map(|vertex_id| {
            VertexTrails::new(vertex_id as VertexId, &graph, &hyperparams)
        }).collect();
        Colony {
            graph: graph.clone(),
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
            debug!("ACO iteration #{iter}/{total}", iter = iter + 1,
                   total = self.hyperparams.iterations);
            debug_log_pheromones(self, iter);
            debug_log_scores(self, iter);
            self.run_iteration();
            let best_ant = self.global_best.as_ref().expect("No solution found...?");
            debug!("  best global score: {best}", best = best_ant.score);
        }
        let best_ant = self.global_best.as_ref().expect("No solution found...?");
        Solution::from_ant(best_ant, &self.graph)
    }

    fn run_iteration(&mut self) {
        self.construct_solutions();
        // Note: our path finalization logic is essentially a local search.
        self.update_trails();
    }

    fn construct_solutions(&mut self) -> Vec<Ant> {
        let ants: Vec<Ant> = iter::repeat(()).take(self.hyperparams.ants)
            .map(|_| self.construct_solution()).collect();
        let local_best = ants.iter().max_by_key(|ant| ant.score).cloned().unwrap();
        debug!("  local best score: {score}", score = local_best.score);
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

    fn sample_option(&mut self, ant: &Ant) -> Option<VertexId> {
        let tick = ant.tick;
        let trail = &self.trails[ant.current as usize];
        let weights = trail.weights(tick).iter().enumerate().map(|(i, w)| {
            let i = i as VertexId;
            let cost = self.graph.cost(ant.tick_offset, ant.current, i);
            let valid = i != ant.current && ant.valid_option(cost, i, &self.graph);
            w * ((valid as i32) as f32)
        });
        let rand_valid_option = |rng: &mut SmallRng| -> Option<VertexId> {
            let all_options: Vec<VertexId> = (0..self.graph.ports.len())
                .map(|v| v as VertexId)
                .filter(|&i| {
                    let cost = self.graph.cost(ant.tick_offset, ant.current, i);
                    i != ant.current && ant.valid_option(cost, i, &self.graph)
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
                .map(|(idx, _)| idx).unwrap();
            let to = to as VertexId;
            if ant.valid_option(self.graph.cost(ant.tick_offset, ant.current, to),
                                to, &self.graph) {
                Some(to)
            } else {
                rand_valid_option(&mut self.rng)
            }
        } else if let Ok(distribution) = WeightedIndex::new(weights) {
            let options: Vec<VertexId> = (0..self.graph.ports.len())
                .map(|v| v as VertexId).collect();
            Some(options[distribution.sample(&mut self.rng)])
        } else {
            rand_valid_option(&mut self.rng)
        }
    }

    fn construct_solution(&mut self) -> Ant {
        let mut ant = Ant::new();
        let start = self.rng.gen_range(0..self.graph.ports.len());
        ant.reset(start as VertexId, self.graph.start_tick, 0);
        while let Some(to) = self.sample_option(&ant) {
            // Local trail update
            let pheromone_add = self.hyperparams.local_evaporation_rate
                * self.hyperparams.base_pheromones;
            self.trails[ant.current as usize].evaporate_add(to,
                self.hyperparams.local_evaporation_rate,pheromone_add);
            // TODO: also update other direction?
            ant.visit(to, &self.graph);
        }
        ant.finalize_path(&self.graph);
        // TODO: should only do local trail update here, after finalizing?
        ant
    }
    
    fn update_trails(&mut self) {
        // Decay all existing trails
        for vertex_trails in self.trails.iter_mut() {
            for to in 0..vertex_trails.pheromones.len() {
                let to = to as VertexId;
                if vertex_trails.vertex == to {
                    continue;
                }
                vertex_trails.evaporate(to, self.hyperparams.evaporation_rate);
            }
        }

        // Global trail update
        // TODO: sometimes pick local best?
        let best = self.global_best.as_ref().expect("No global best at update time!");
        let upper_bound_score = eval_score((self.graph.ports.len() + 1) as u32,
                                           /*ticks=*/self.graph.ports.len() as u16,
                                           /*looped=*/true);
        // TODO: compute pheromone add differently?
        let add = 1.0 / (upper_bound_score + 1 - best.score) as f32;

        let pheromone_add = self.hyperparams.evaporation_rate * add;
        let mut from = best.start;
        for to in &best.path {
            self.trails[from as usize].add(*to, pheromone_add);
            // TODO: also update other direction?
            from = *to;
        }
    }
}

impl Solution {
    pub fn from_ant(ant: &Ant, graph: &Graph) -> Self {
        let mut repeat_ant = Ant::new();
        repeat_ant.reset(ant.start, graph.start_tick, 0);
        let mut paths = Vec::new();
        for to in &ant.path {
            let path = graph.path(graph.tick_offset(repeat_ant.tick),
                                  repeat_ant.current, *to);
            paths.push(path.clone());
            repeat_ant.visit(*to, graph);
        }
        Solution {
            score: ant.score,
            spawn: graph.ports[ant.start as usize],
            paths,
        }
    }
}


// All the following are pretty hacky log outputs, optionally parsed to produce
// visualizations.
fn debug_log_graph(graph: &Graph) {
    debug!("[LOGGING_GRAPH_START_TICK]{tick}", tick = graph.start_tick);
    debug!("[LOGGING_GRAPH_MAX_TICK]{tick}", tick = graph.max_ticks);
    for from in 0..graph.ports.len() {
        let from = from as VertexId;
        let port = graph.ports[from as usize];
        debug!("[LOGGING_GRAPH_VERTICES]{x} {y}", x = port.x, y = port.y);
        for to in 0..graph.ports.len() {
            let to = to as VertexId;
            if from == to {
                continue;
            }
            let costs: Vec<u8> = (0..TICK_OFFSETS).map(
                |t| graph.cost(t as u8, from, to)).collect();
            debug!("[LOGGING_GRAPH_EDGES]{from} {to} {costs:?}");
        }
    }
}
fn debug_log_pheromones(colony: &Colony, iter: usize) {
    for from in 0..colony.graph.ports.len() {
        let from = from as VertexId;
        let pheromones = colony.trails[from as usize].pheromones.clone();
        debug!("[LOGGING_PHEROMONES]{iter} {from} {pheromones:?}");
    }
}
fn debug_log_ant(ant: &Ant, tag: &str) {
    debug!("{tag}{start} {path:?} {score}", start = ant.start,
          path = ant.path, score = ant.score);
}
fn debug_log_heuristics(graph: &Graph) {
    for from in 0..graph.ports.len() {
        let from = from as VertexId;
        for to in 0..graph.ports.len() {
            let to = to as VertexId;
            if from == to {
                continue;
            }
            let dists: Vec<u8> = (0..TICK_OFFSETS).map(
                |t| graph.cost(t as u8, from, to)).collect();
            let min = 1.0 / (*dists.iter().max().unwrap() as f32);
            let max = 1.0 / (*dists.iter().min().unwrap() as f32);
            debug!("[LOGGING_HEURISTIC]{min} {max}");
        }
    }
}
fn debug_log_scores(colony: &Colony, iter: usize) {
    for from in 0..colony.graph.ports.len() {
        let from = from as VertexId;
        for to in 0..colony.graph.ports.len() {
            let to = to as VertexId;
            if from == to {
                continue;
            }
            let distances: Vec<u8> = (0..TICK_OFFSETS).map(
                |t| colony.graph.cost(t as u8, from, to)).collect();
            let min_dist = *distances.iter().min().unwrap();
            let max_dist = *distances.iter().max().unwrap();
            let tau = colony.trails[from as usize].pheromones[to as usize];
            let beta = colony.hyperparams.heuristic_power;
            let min_eta = 1.0 / (max_dist as f32);
            let max_eta = 1.0 / (min_dist as f32);
            let min_weight = tau * min_eta.powf(beta);
            let max_weight = tau * max_eta.powf(beta);
            debug!("[LOGGING_WEIGHTS]{iter} {from} {to} {min_weight} {max_weight}");
        }
    }
}
