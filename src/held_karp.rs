// Exact solver for the Traveling Salesman Problem with the Held-Karp algorithm.
// See https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm
//
// Essentially, we compute g(S, e), "what's the shortest path from vertex 1
// going through a subset S of vertices, then going to e", starting with |S|=1
// (direct cost from 1 to v, then v to e), we can get the next size by trying
// every possible new second-to-last vertex and picking the smallest.
//
// The final TSP solution starting at '1' is the g(S, i) for the 'i' that gives
// the smallest distance.

use log::info;
use std::sync::{mpsc, Arc};
use std::thread;

use crate::challenge::{Solution, eval_score};
use crate::challenge_consts::{MAX_PORTS, NUM_THREADS, TICK_OFFSETS};
use crate::graph::{Graph, VertexId};

const MAX_MASK_ITEMS: usize = MAX_PORTS - 1;
// Size needed for an array indexed by masks.
const NUM_MASKS: usize = 1 << MAX_MASK_ITEMS;

// We flatten arrays conceptually indexed by a mask 'S' and vertex 'e' to a
// flat array where masks of similar sizes (number of '1' bits) are close in
// memory. This is the size of that flattened array.
const FLAT_SIZE: usize = (MAX_PORTS - 1) * NUM_MASKS;

type Set = [VertexId; MAX_PORTS];
type Cost = u16;

#[derive(Clone)]
struct Tour {
    cost: Cost,  // in terms of total distance
    vertices: Vec<VertexId>,
}

pub fn held_karp(
    graph: &Arc<Graph>, max_starts: usize, check_shorter_tours: bool
    ) -> Option<Solution> {
    let mut best_tour = Tour { cost: Cost::MAX, vertices: Vec::new() };
    let mut handles = vec![];
    let (tx, rx) = mpsc::channel();
    for i in 0..usize::min(NUM_THREADS, max_starts) {
        let tx = tx.clone();
        let graph = graph.clone();
        handles.push(thread::spawn(move || {
            let mut held_karp = HeldKarp::new();
            for start in (i..usize::min(max_starts, graph.ports.len()))
                .step_by(NUM_THREADS) {
                let start = start as VertexId;
                let full_tour = held_karp.traveling_salesman(&graph, start);
                full_tour.verify_tour(&graph);
                if check_shorter_tours {
                    let tour = held_karp.consider_shorter_tours(&graph, start,
                                                                &full_tour);
                    tx.send(tour).unwrap();
                } else {
                    tx.send(full_tour).unwrap();
                }
            }
        }));
    }
    drop(tx);  // Drop the last sender, wait until all threads are done.
    while let Ok(tour) = rx.recv() {
        if tour.score(graph) > best_tour.score(graph) {
            best_tour = tour;
        }
    }
    for handle in handles {
        handle.join().unwrap();
    }
    best_tour.to_solution(&graph)
}

struct HeldKarp {
    // This is a flattened array of the g(S, e) that gives us, for a given 'S',
    // the minimal cost of going through all cities in 'S', then going to
    // possible 'e' cities.
    // It is built in a way that keeps masks of similar sizes close in memory.
    // See https://www.math.uwaterloo.ca/~bico/papers/comp_chapterDP.pdf
    g: Vec<Cost>,
    // This is p(S, e) that gives us, for a given 'S', the predecessor city that
    // we visited to end up to each possible 'e' cities. Also flattened.
    p: Vec<VertexId>,

    // Precomputed binomial[m][k] gives m-choose-k.
    // TODO: refactor to Binomial helper
    binomial: [[usize; MAX_PORTS]; MAX_PORTS],
    // For a given subset size 's', the start index in the array 'v' for that
    // subset's data.
    b: [usize; MAX_PORTS],
}

// TODO: refactor set handling to custom struct?
fn first_set(set: &mut Set, s: usize) {
    for i in 0..s {
        set[i] = i as VertexId;
    }
}

fn next_set(set: &mut Set, s: usize) {
    let mut i = 0;
    while i < s - 1 && set[i] + 1 == set[i + 1] {
        set[i] = i as VertexId;
        i += 1;
    }
    set[i] += 1;
}

impl HeldKarp {
    pub fn new() -> Self {
        let mut binomial =  [[0usize; MAX_PORTS]; MAX_PORTS];
        for m in 0..MAX_PORTS {
            binomial[m][0] = 1;
            binomial[m][m] = 1;
        }
        // Use recurrence relationship
        // m-choose-k = (m-1)-choose-(k-1) + (m-1)-choose-k
        for m in 1..MAX_PORTS {
            for k in 1..m {
                binomial[m][k] = binomial[m-1][k-1] + binomial[m-1][k];
            }
        }
        let mut b = [0usize; MAX_PORTS];
        for i in 1..MAX_PORTS {
            b[i] = b[i-1] + (i-1) * (binomial[MAX_MASK_ITEMS][i-1]);
        }
        HeldKarp {
            g: vec![Cost::MAX; FLAT_SIZE],
            p: vec![VertexId::MAX; FLAT_SIZE],
            binomial,
            b,
        }
    }

    pub fn traveling_salesman(&mut self, graph: &Graph, start: VertexId) -> Tour {
        let start_tick = graph.start_tick + 1;  // time to dock spawn

        for i in 0..FLAT_SIZE {
            self.g[i] = Cost::MAX;
            self.p[i] = VertexId::MAX;
        }

        // Precompute conversion from IDs ignoring 'start', back to their proper
        // VertexId.
        let untranslated: [VertexId; MAX_PORTS] = array_init::array_init(
            |v: usize| self.untranslate(start, v as VertexId));
        let mut set: Set = [0; MAX_PORTS];
        first_set(&mut set, 1);

        // Initialize for |S|=1 (with S = {k})
        while (set[0] as usize) < graph.ports.len() - 1 {
            let k = set[0];
            // Start at +1 to dock spawn.
            // Initial cost is the start tick, plus cost from spawn to 'k'.
            let cost = graph.cost(graph.tick_offset(start_tick), start,
                                  untranslated[k as usize]);
            let idx = self.flat_index_region(&set, 1);
            // +1 to dock 'k'
            self.g[idx] = start_tick + (cost as Cost) + 1;
            self.p[idx] = k;
            next_set(&mut set, 1);
        }

        // From |S|=s, deduce S' (|S'| = s+1) g(S', k) by picking the
        // before-last city that minimizes cost, for each k.
        let max_set_items = graph.ports.len() - 1;
        // TODO: helper struct to support this
        let mut set_minus_k = [0; MAX_PORTS];
        for s in 2..=max_set_items {
            first_set(&mut set, s);
            while (set[s - 1] as usize) < graph.ports.len() - 1 {
                let k_index_base = self.flat_index_region(&set, s);
                let mut m_index_base = self.b[s - 1];
                for t in 1..s {
                    set_minus_k[t - 1] = set[t];
                    m_index_base += (s-1) * self.binomial[set_minus_k[t - 1] as usize][t];
                }
                for k in 0..s {
                    let (min_cost, min_vertex) = (0..(s-1)).map(|m| {
                        let current_cost = self.g[m_index_base + m];
                        let m_k_cost = graph.cost(
                            graph.tick_offset(current_cost),
                            untranslated[set_minus_k[m] as usize],
                            untranslated[set[k] as usize]);
                        // + 1 to dock
                        let cost = current_cost + (m_k_cost as Cost) + 1;
                        (cost, set_minus_k[m])
                    }).min_by_key(|&(cost, _)| cost).unwrap();
                    self.g[k_index_base + k] = min_cost;
                    self.p[k_index_base + k] = min_vertex;
                    m_index_base -= (s-1) * self.binomial[set_minus_k[k] as usize][k+1];
                    set_minus_k[k] = set[k];
                    m_index_base += (s-1) * self.binomial[set_minus_k[k] as usize][k+1];
                }
                next_set(&mut set, s);
            }
        }

        // Find the best tour by checking paths back to each start.
        first_set(&mut set, max_set_items);
        let mut total_cost = Cost::MAX;
        let mut last_city: VertexId = VertexId::MAX;
        for k in 0..max_set_items {
            let idx = self.flat_index_region(&set, max_set_items) + k;
            let current_cost = self.g[idx];
            let k_start_cost = graph.cost(graph.tick_offset(current_cost),
                                          untranslated[set[k] as usize], start);
            // Note: no "+ 1" for docking, the last dock tick doesn't count.
            let cost = current_cost + (k_start_cost as Cost);
            if cost < total_cost {
                total_cost = cost;
                last_city = set[k];
            }
        }
        let vertices = self.backtrack(start, last_city, &set, max_set_items);
        info!("Starting at port ID {}, cost would be {}, vertices: {vertices:?}",
              start, total_cost);
        Tour { cost: total_cost, vertices }
    }

    // Backtrack to get vertices for a tour.
    fn backtrack(
        &self, start: VertexId, last: VertexId, set: &Set, set_size: usize
        ) -> Vec<VertexId> {
        let mut set = set.clone();
        let mut vertices = Vec::with_capacity(set_size + 2);
        vertices.push(start);
        let mut vertex = last;
        for s in (1..=set_size).rev() {
            vertices.push(self.untranslate(start, vertex));
            let vertex_idx = set.iter().position(|&v| v == vertex).unwrap();
            vertex = self.p[self.flat_index_region(&set, s) + vertex_idx];
            for i in vertex_idx..s {
                set[i] = set[i+1];
            }
        }
        vertices.push(start);
        vertices.reverse();
        vertices
    }

    // Sets are in the space where the 'start' vertex is removed, this utility
    // is used to convert back to the vertex IDs.
    fn untranslate(&self, start: VertexId, v: VertexId) -> VertexId {
        if v < start { v } else { v + 1 }
    }

    // Checks every subset of ports to see if visiting these would give a better
    // score than the best full tour found. This is slow.
    pub fn consider_shorter_tours(&self, graph: &Graph, start: VertexId,
                                  full_tour: &Tour) -> Tour {
        let mut best_tour = full_tour.clone();
        let mut best_score = full_tour.score(graph);
        // Note: assume that best_score is valid (<= max ticks), for simplicity.
        assert!(full_tour.cost <= graph.max_ticks,
                "Maps with invalid full-tours not supported");
        // Note: only considering tours that loop (i.e. each city is worth
        // 150 * 2 pts). 
        let min_ports = ((best_score + 299) / 300) as usize;

        let mut set: Set = [0; MAX_PORTS];
        let min_set_items = min_ports - 1;  // -1 since start is implicit
        let max_set_items = graph.ports.len() - 1; // no need t
        // Note: max_set_items excluded since we already have full_tour.
        for s in min_set_items..max_set_items {
            first_set(&mut set, s);
            while (set[s - 1] as usize) < graph.ports.len() - 1 {
                let k_index_base = self.flat_index_region(&set, s);
                for k in 0..s {
                    let current_cost = self.g[k_index_base + k];
                    let k_start_cost = graph.cost(
                        graph.tick_offset(current_cost),
                        self.untranslate(start, set[k]), start);
                    // Note: no '+1' for final dock -- it's free.
                    let cost = current_cost + (k_start_cost as Cost);
                    let visits = s + 2;  // +2 for the start port (looped)
                    let score = eval_score(visits as u32, cost,
                                           /*looped=*/true);
                    if score > best_score {
                        best_tour = Tour {
                            cost,
                            vertices: self.backtrack(start, k as VertexId,
                                                     &set, s),
                        };
                        best_score = score;
                    }
                }
                next_set(&mut set, s);
            }
        }

        best_tour
    }

    // Index of the start where the given 'set' would be in a flattened array.
    // The values of 'set' will be there in order.
    fn flat_index_region(&self, set: &Set, s: usize) -> usize {
        let mut loc = 0usize;
        for i in 0..s {
            loc += self.binomial[set[i] as usize][i+1];
        }
        self.b[s] + s * loc
    }

    // Debugging function, used to investigate a (better) given tour
    // step-by-step to iron out why a step wasn't taken.
    fn _debug_tour(&mut self, tour: &Tour, graph: &Graph) {
        let start = tour.vertices[0];
        let our_tour = self.traveling_salesman(graph, start);
        assert!(our_tour.vertices.len() == tour.vertices.len(),
                "There's an optimal solution with < 20 cities");

        // Used for creating sets, in a space where 'start' isn't a vertex.
        let translate = |v: VertexId| if v <= start { v } else { v - 1 };

        // +1 to dock start
        let mut our_tick = graph.start_tick + 1;
        let mut our_set: Set = [VertexId::MAX; MAX_PORTS];
        let mut tour_tick = graph.start_tick + 1;
        let mut tour_set: Set = [VertexId::MAX; MAX_PORTS];
        for i in 1..tour.vertices.len() {
            info!("Target tour tick: {}", tour_tick);
            let prev_vertex = tour.vertices[i-1];
            let vertex = tour.vertices[i];
            let s = i;
            let cost = graph.cost(graph.tick_offset(tour_tick), prev_vertex,
                                  vertex);
            info!("Target tour picked: {}->{} (cost {}+1)", prev_vertex+1,
                  vertex+1, cost);
            tour_tick += (cost as u16) + 1;
            tour_set[i-1] = translate(vertex);
            if vertex != start {
                info!("{}", self._show(&tour_set, s, i-1, start));
            }

            info!("Our tick: {}", our_tick);
            let our_prev_vertex = our_tour.vertices[i-1];
            let our_vertex = our_tour.vertices[i];
            let our_cost = graph.cost(graph.tick_offset(our_tick),
                                      our_prev_vertex, our_vertex);
            info!("Our tour picked: {}->{} (cost {}+1)", our_prev_vertex+1,
                  our_vertex+1, our_cost);
            our_tick += (our_cost as u16) + 1;
            our_set[i-1] = translate(our_vertex);
            if our_vertex != start {
                info!("{}", self._show(&our_set, s, i-1, start));
            }
            info!("-----------------------");
        }
        panic!("Debug logs before here.");
    }

    // Use for debugging
    fn _show(&self, set: &Set, s: usize, e_idx: usize, start: VertexId) -> String {
        info!("looking up idx={} with s={}", self.flat_index_region(set, s) + e_idx, s);
        let g = self.g[self.flat_index_region(set, s) + e_idx];
        let p = self.p[self.flat_index_region(set, s) + e_idx];
        format!("g({mask}, {e}) = {g}   p({mask}, {e}) = {p}",
                mask = self._show_set(set, s, start),
                e = self.untranslate(start, set[e_idx]) + 1,
                p = self.untranslate(start, p) + 1)
    }

    // Use for debugging
    fn _show_set(&self, set: &Set, s: usize, start: VertexId) -> String {
        let set_vertices: Vec<_> = (0..s)
            .map(|i| (self.untranslate(start, set[i])+1).to_string()).collect();
        format!("{{{}}}", set_vertices.join(","))
    }
}

impl Tour {
    fn to_solution(&self, graph: &Graph) -> Option<Solution> {
        if self.cost < graph.max_ticks {
            let spawn = graph.ports[self.vertices[0] as usize];
            let mut paths = Vec::with_capacity(self.vertices.len() - 1);
            let mut tick = graph.start_tick + 1;  // +1 to dock spawn
            for i in 1..self.vertices.len() {
                let prev = self.vertices[i-1];
                let next = self.vertices[i];
                let path = graph.path(graph.tick_offset(tick), prev, next);
                paths.push(path.clone());
                tick += path.cost + 1;  // + 1 to dock it
            }
            Some(Solution { score: self.score(graph), spawn, paths })
        } else {
            None
        }
    }

    fn score(&self, graph: &Graph) -> i32 {
        if self.vertices.is_empty() {
            return 0;
        }
        let visits = self.vertices.len();
        let did_loop = self.vertices.last().unwrap() == self.vertices.first().unwrap();
        assert!(did_loop, "Only support looping tours");
        let ticks = if did_loop { self.cost } else { graph.max_ticks };
        // Note: to support the following, must remove visits with tick > max
        assert!(ticks <= graph.max_ticks, "Tour going over limit not supported");
        eval_score(visits as u32, ticks, did_loop)
    }

    fn verify_tour(&self, graph: &Graph) {
        let mut tick = graph.start_tick + 1;  // +1 to dock spawn
        for i in 1..self.vertices.len() {
            let from = self.vertices[i - 1];
            let to = self.vertices[i];
            let offset = tick % (TICK_OFFSETS as u16);
            tick += graph.cost(offset as u8, from, to) as u16 + 1;
        }
        tick -= 1;
        assert!(tick == self.cost, "Tour would give cost {}, but we got {}",
                tick, self.cost);
    }

    // This is used only for debugging when comparing to a solution from another
    // approach.
    fn _from_solution(graph: &Graph, solution: &Solution) -> Self {
        let get_idx = |p| graph.ports.iter().position(|&v| v == p).unwrap();
        let mut cost = graph.start_tick + 1;  // +1 to dock spawn
        let mut vertices = vec![get_idx(solution.spawn) as VertexId];
        for path in &solution.paths {
            vertices.push(get_idx(path.goal) as VertexId);
            cost += path.cost + 1;  // +1 to dock target
        }
        cost -= 1;  // The last dock back to spawn doesn't count
        Tour { vertices, cost }
    }
}
