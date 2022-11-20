use arrayvec::ArrayVec;
use log::{debug};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::iter;
use std::sync::Arc;
use rand::{Rng, SeedableRng};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::SmallRng;
use serde_json::{json};

use crate::challenge::{Solution, eval_score};
use crate::challenge_consts::{MAX_PORTS, TICK_OFFSETS};
use crate::graph::{Graph, VertexId};
use crate::pathfinding::{Pos};

// Based on the documentation at:
// https://staff.washington.edu/paymana/swarm/stutzle99-eaecs.pdf
// http://www.scholarpedia.org/article/Ant_colony_optimization
// https://www.researchgate.net/publication/277284831_MAX-MIN_ant_system

// TODO: Add sweepable params to:
// - Add pheromones from all ants
// - Update pheromones using local best with probability
// - Update pheromones considering global best only after X% iterations
// - global score update on equality

// TODO: try
// - What is missing from ~vanilla AS?
// - What is missing from ~vanilla ACO?
// - Pheromone at the tick_offset level
// - Rank-based version of any-system?
// - Include other local search before updates?
// - Update both directions of edge?

#[derive(Deserialize, Debug, Clone)]
pub struct HyperParams {
    /// Number of rounds of ant simulations to do.
    pub iterations: usize,

    /// 'm' ants that construct solutions at each iteration
    pub ants: usize,

    /// 'ρ' (rho) used in pheromone (τ) updates, i.e.
    /// τ = (1 - ρ) τ + ρ total_fitness
    pub evaporation_rate: f32,

    /// q0 used when choosing actions. With probability q0, the best move is
    /// exploited instead of sampling.
    pub exploitation_probability: f32,

    // NOTE: we don't support 'α' (alpha) as pheromone trail exponent to speed
    // up computation. Instead, we rely on beta being swept (note: not
    // strictly equivalent).

    /// 'β' (beta) used when sampling actions, applied to the heuristic η: η^β
    pub heuristic_power: f32,

    /// 'ξ' evaporation rate for local updates, in the following formula:
    /// τ = (1 − ξ) · τ + ξ τ0
    pub local_evaporation_rate: f32,

    /// Min τ value (from MAX-MIN Ant System).
    /// Also used as a base 'τ0' value for local updates, in the ACO formula:
    /// τ = (1 − ξ) · τ + ξ τ0
    pub min_pheromones: f32,
    /// Max τ value (from MAX-MIN Ant System).
    pub max_pheromones: f32,
    /// Ratio from 0 to 1 for the initial pheromone value.
    /// A value of 0 means min_pheromones, a value of 1 means max_pheromones.
    pub pheromones_init_ratio: f32,

    /// Seed to use for randomness.
    pub seed: u64,
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
    // Min and max pheromone values
    pub min: f32,
    pub max: f32,
}

pub struct Colony {
    pub hyperparams: HyperParams,
    pub graph: Arc<Graph>,
    // Trails for each vertex.
    pub trails: Vec<VertexTrails>,
    pub global_best: Option<Ant>,
    pub rng: SmallRng,

    // Used for debugging purposes. If set, is populated & debug-logged.
    visualization: Option<ColonyVisualization>,
}

/// Debug struct to log as JSON for visualization, contains all iterations info.
#[derive(Serialize, Debug, Clone)]
pub struct ColonyVisualization {
    pub graph: GraphVisualization,
    // Local/global best, and all ants for every steps
    pub local_ants: Vec<AntVisualization>,
    pub global_ants: Vec<AntVisualization>,
    pub all_ants: Vec<Vec<AntVisualization>>,
    // Pheromone/heuristic/weight info for every steps
    pub weights: Vec<WeightsVisualization>,
}

/// Debug struct used for visualization of an individual ant.
#[derive(Serialize, Debug, Clone)]
pub struct AntVisualization {
    pub start: VertexId,
    pub score: i32,
    pub path: Vec<VertexId>,
}

/// Debug struct used for visualization of the graph.
#[derive(Serialize, Debug, Clone)]
pub struct GraphVisualization {
    pub vertices: Vec<Pos>,
    // adjacency[from][to][tick_offset]
    pub adjacency: Vec<Vec<Vec<u8>>>,
}

/// Debug struct used for visualization of pheromones, heuristics, and weights.
#[derive(Serialize, Debug, Clone)]
pub struct WeightsVisualization {
    // pheromones[from][to]
    pub pheromones: Vec<Vec<f32>>,
    // heuristics[from][to]
    pub min_heuristics: Vec<Vec<f32>>,
    pub max_heuristics: Vec<Vec<f32>>,
    // weights[from][to]
    pub min_weights: Vec<Vec<f32>>,
    pub max_weights: Vec<Vec<f32>>,
}

impl HyperParams {
    pub fn default_params(iterations: usize) -> Self {
        HyperParams {
            iterations,
            ants: 25,
            evaporation_rate: 0.2,
            exploitation_probability: 0.1,
            heuristic_power: 3.0,
            local_evaporation_rate: 0.01,
            min_pheromones: 0.01,
            max_pheromones: 250.0,
            pheromones_init_ratio: 1.0,
            seed: 42,
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

    pub fn reset(&mut self, start: VertexId, tick: u16, graph: &Graph) {
        self.start = start;
        self.current = start;
        self.tick = tick + 1;  // Time to dock our start
        self.tick_offset = graph.tick_offset(self.tick);
        self.path.clear();
        assert!(start < 64);
        self.seen = 1u64 << start;
    }

    fn visit(&mut self, port: VertexId, graph: &Graph) {
        self.path.push(port);
        let cost = graph.cost(self.tick_offset, self.current, port);
        self.current = port;
        let dock_cost = if port == self.start { 0 } else { 1 };
        self.tick += (cost + dock_cost) as u16;
        self.tick_offset = graph.tick_offset(self.tick);
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
            let dock_cost = if *to == self.start { 0 } else { 1 };
            tick += (cost as u16) + dock_cost;
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
            self.tick_offset = graph.tick_offset(tick);
            self.current = vertex;
            self.visit(self.start, graph);
            assert!(self.compute_score(graph, /*simulate_to_end=*/true) == best_score);
        }
    }

    fn to_solution(&self, graph: &Graph) -> Solution {
        let mut repeat_ant = Ant::new();
        repeat_ant.reset(self.start, graph.start_tick, graph);
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

        let base_pheromones = hyperparams.pheromones_init_ratio * (
            hyperparams.max_pheromones - hyperparams.min_pheromones);

        VertexTrails {
            vertex: vertex_id,
            pheromones: ArrayVec::from_iter(vec![base_pheromones; graph.ports.len()]),
            offset_trail_weights: ArrayVec::from_iter(
                vec![ArrayVec::from_iter(vec![1.0; graph.ports.len()]); TICK_OFFSETS]),
            min: hyperparams.min_pheromones,
            max: hyperparams.max_pheromones,
            eta_pows,
        }
    }

    pub fn evaporate_add(&mut self, to: VertexId, evaporation: f32,
                         pheromone: f32) {
        self.set_pheromones(
            to, (1.0 - evaporation) * self.pheromones[to as usize]
            + evaporation * pheromone);
        self.update_weights(to);
    }

    pub fn add(&mut self, to: VertexId, pheromone: f32) {
        self.set_pheromones(to, self.pheromones[to as usize] + pheromone);
        self.update_weights(to);
    }

    pub fn evaporate(&mut self, to: VertexId, evaporation_rate: f32) {
        self.set_pheromones(
            to, self.pheromones[to as usize] * (1.0 - evaporation_rate));
        self.update_weights(to);
    }

    fn update_weights(&mut self, to: VertexId) {
        for offset in 0..self.offset_trail_weights.len() {
            let tau = self.pheromones[to as usize];
            self.offset_trail_weights[offset][to as usize] = tau * self.eta_pow(offset as u8, to);
        }
    }

    fn set_pheromones(&mut self, to: VertexId, pheromones: f32) {
        self.pheromones[to as usize] = pheromones.clamp(self.min, self.max);
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
    pub fn new(graph: &Arc<Graph>, hyperparams: HyperParams) -> Self {
        let vertex_trails = (0..graph.ports.len()).map(|vertex_id| {
            VertexTrails::new(vertex_id as VertexId, &graph, &hyperparams)
        }).collect();
        let seed = hyperparams.seed;
        let visualization: Option<ColonyVisualization> = if cfg!(feature = "visualization_dump") {
            Some(ColonyVisualization::new(graph))
        } else {
            None
        };
        Colony {
            graph: graph.clone(),
            trails: vertex_trails,
            global_best: None,
            hyperparams,
            rng: SmallRng::seed_from_u64(seed),
            visualization,
        }
    }

    pub fn run(&mut self) -> Solution {
        for iter in 0..self.hyperparams.iterations {
            debug!("ACO iteration #{iter}/{total}", iter = iter + 1,
                   total = self.hyperparams.iterations);
            self.run_iteration();
            let best_ant = self.global_best.as_ref().expect("No solution found...?");
            debug!("  best global score: {best}", best = best_ant.score);
        }
        let best_ant = self.global_best.as_ref().expect("No solution found...?");
        if let Some(viz) = &self.visualization {
            debug!("[VIZ_DATA] {}", json!(viz));
        }
        best_ant.to_solution(&self.graph)
    }

    fn run_iteration(&mut self) {
        if let Some(viz) = &mut self.visualization {
            // Note weights/pheromones before tick starts
            viz.weights.push(WeightsVisualization::new(&self.trails,
                                                       &self.graph,
                                                       &self.hyperparams));
        }

        self.construct_solutions();
        // Note: our path finalization logic is essentially a local search.
        self.update_trails();
    }

    fn construct_solutions(&mut self) -> Vec<Ant> {
        let ants: Vec<Ant> = iter::repeat(()).take(self.hyperparams.ants)
            .map(|_| self.construct_solution()).collect();
        let local_best = ants.iter().max_by_key(|ant| ant.score).cloned().unwrap();
        debug!("  local best score: {score}", score = local_best.score);
        if self.global_best.is_none()
            || local_best.score > self.global_best.as_ref().unwrap().score {
            self.global_best = Some(local_best.clone());
        }
        if let Some(viz) = &mut self.visualization {
            viz.local_ants.push(AntVisualization::new(&local_best));
            viz.global_ants.push(AntVisualization::new(
                    &self.global_best.as_ref().unwrap()));
            viz.all_ants.push(ants.iter().map(AntVisualization::new).collect());
        }
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
        ant.reset(start as VertexId, self.graph.start_tick, &self.graph);
        while let Some(to) = self.sample_option(&ant) {
            // Local trail update
            let pheromone_add = self.hyperparams.local_evaporation_rate
                * self.hyperparams.min_pheromones;
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
        let add = 1.0 / (best.tick as f32);

        let pheromone_add = self.hyperparams.evaporation_rate * add;
        let mut from = best.start;
        for to in &best.path {
            self.trails[from as usize].add(*to, pheromone_add);
            // TODO: also update other direction?
            from = *to;
        }
    }
}

impl ColonyVisualization {
    fn new(graph: &Graph) -> Self {
        ColonyVisualization {
            graph: GraphVisualization::new(graph),
            local_ants: Vec::new(),
            global_ants: Vec::new(),
            all_ants: Vec::new(),
            weights: Vec::new(),
        }
    }
}

impl GraphVisualization {
    fn new(graph: &Graph) -> Self {
        let mut adjacency = Vec::new();
        for from in 0..graph.ports.len() {
            let mut from_adj = Vec::new();
            for to in 0..graph.ports.len() {
                let mut offset_adj = Vec::new();
                for offset in 0..TICK_OFFSETS {
                    offset_adj.push(graph.cost(offset as u8,
                                               from as VertexId,
                                               to as VertexId));
                }
                from_adj.push(offset_adj);
            }
            adjacency.push(from_adj);
        }
        GraphVisualization {
            vertices: graph.ports.to_vec(),
            adjacency,
        }
    }
}

impl AntVisualization {
    fn new(ant: &Ant) -> Self {
        AntVisualization {
            start: ant.start,
            score: ant.score,
            path: ant.path.to_vec(),
        }
    }
}

impl WeightsVisualization {
    fn new(trails: &Vec<VertexTrails>, graph: &Graph,
           hyperparams: &HyperParams) -> Self {
        // // pheromones[from][to]
        // pub pheromones: Vec<Vec<f32>>,
        // // heuristics[from][to]
        // pub min_heuristics: Vec<Vec<f32>>,
        // pub max_heuristics: Vec<Vec<f32>>,
        // // weights[from][to]
        // pub min_weights: Vec<Vec<f32>>,
        // pub max_weights: Vec<Vec<f32>>,
        let mut pheromones = Vec::new();
        let mut min_heuristics = Vec::new();
        let mut max_heuristics = Vec::new();
        let mut min_weights = Vec::new();
        let mut max_weights = Vec::new();
        for from in 0..graph.ports.len() {
            let mut from_pheromones = Vec::new();
            let mut from_min_heuristics = Vec::new();
            let mut from_max_heuristics = Vec::new();
            let mut from_min_weights: Vec<f32> = Vec::new();
            let mut from_max_weights: Vec<f32> = Vec::new();
            for to in 0..graph.ports.len() {
                from_pheromones.push(trails[from].pheromones[to]);
                let beta = hyperparams.heuristic_power;
                // Undo precomputed eta pow computation to recover eta.
                let heuristics = (0..TICK_OFFSETS).map(
                    |t| trails[from].eta_pow(t as u8, to as u8).powf(1f32 / beta));
                let weights = (0..TICK_OFFSETS).map(
                    |t| trails[from].weights(t as u16)[to]);
                from_min_heuristics.push(
                    heuristics.clone()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap());
                from_max_heuristics.push(
                    heuristics
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap());
                from_min_weights.push(
                    weights.clone()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap());
                from_max_weights.push(
                    weights
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .unwrap());
            }
            pheromones.push(from_pheromones);
            min_heuristics.push(from_min_heuristics);
            max_heuristics.push(from_max_heuristics);
            min_weights.push(from_min_weights);
            max_weights.push(from_max_weights);
        }
        WeightsVisualization {
            pheromones, min_heuristics, max_heuristics,
            min_weights, max_weights
        }
    }
}
