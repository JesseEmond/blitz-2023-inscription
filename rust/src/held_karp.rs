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

use crate::challenge::{Solution, eval_score};
use crate::challenge_consts::{MAX_PORTS, TICK_OFFSETS};
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

struct Tour {
    cost: Cost,  // in terms of total distance
    vertices: Vec<VertexId>,
}

pub fn held_karp(graph: &Graph) -> Option<Solution> {
    let mut held_karp = HeldKarp::new();
    let tour = (0..graph.ports.len())
        .map(|v| v as VertexId)
        .map(|start| held_karp.traveling_salesman(graph, start))
        .min_by_key(|t| t.cost).unwrap();
    tour.verify_tour(graph);
    tour.to_solution(graph)
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
    binomial: [[usize; MAX_PORTS]; MAX_PORTS],
    // For a given subset size 's', the start index in the array 'v' for that
    // subset's data.
    b: [usize; MAX_PORTS],
}

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

        // TODO: try having g[mask][b][e], computing all starts at once?
        // TODO: precompute lower-bound for all g[mask]? skip computations 
        // with lower-bound > seen? (branch & bound idea)

        // Sets are in the space where the 'start' vertex is removed, this
        // utility is used to convert back to the vertex IDs.
        let untranslated: [VertexId; MAX_PORTS] = array_init::array_init(|v: usize| {
            let v = v as VertexId;
            if v < start { v } else { v + 1 }
        });
        let mut set: Set = [0; MAX_PORTS];
        first_set(&mut set, 1);

        // Initialize for |S|=1 (with S = {k})
        while (set[0] as usize) < graph.ports.len() - 1 {
            let k = set[0];
            // Start at +1 to dock spawn.
            let tick = start_tick + 1;
            // Initial cost is the start tick, plus cost from spawn to 'k'.
            let cost = graph.cost(graph.tick_offset(tick), start,
                                  untranslated[k as usize]);
            // +1 to dock 'k'
            let idx = self.flat_index_region(&set, 1);
            self.g[idx] = tick + (cost as Cost) + 1;
            self.p[idx] = k;
            next_set(&mut set, 1);
        }

        // From |S|=s, deduce S' (|S'| = s+1) g(S', k) by picking the
        // before-last city that minimizes cost, for each k.
        let max_set_items = graph.ports.len() - 1;
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

        // Backtrack to get vertices.
        let mut vertices = Vec::with_capacity(graph.ports.len() + 1);
        vertices.push(start);
        let mut vertex = last_city;
        for s in (1..=max_set_items).rev() {
            vertices.push(untranslated[vertex as usize]);
            let vertex_idx = set.iter().position(|&v| v == vertex).unwrap();
            vertex = self.p[self.flat_index_region(&set, s) + vertex_idx];
            for i in vertex_idx..s {
                set[i] = set[i+1];
            }
        }
        vertices.reverse();
        info!("Starting at port ID {}, cost would be {}, vertices: {vertices:?}",
              start, total_cost);
        Tour { cost: total_cost, vertices: vertices, }
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

    // Use for debugging
    fn _show(&self, set: &Set, s: usize, e_idx: usize) -> String {
        let g = self.g[self.flat_index_region(set, s) + e_idx];
        let p = self.p[self.flat_index_region(set, s) + e_idx];
        format!("g({mask}, {e}) = {g}   p({mask}, {e}) = {p}",
                mask = self._show_set(set, s),
                e = set[e_idx] + 1, p = p + 1)
    }

    // Use for debugging
    fn _show_set(&self, set: &Set, s: usize) -> String {
        let set_vertices: Vec<_> = (0..s).map(|i| (set[i]+1).to_string()).collect();
        format!("{{{}}}", set_vertices.join(","))
    }
}

impl Tour {
    fn to_solution(&self, graph: &Graph) -> Option<Solution> {
        if self.cost < graph.max_ticks {
            let visits = self.vertices.len();
            let score = eval_score(visits as u32, self.cost, /*looped=*/true);
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
            Some(Solution { score: score, spawn: spawn, paths: paths })
        } else {
            None
        }
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
}
