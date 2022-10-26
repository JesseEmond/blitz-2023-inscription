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

use arrayvec::ArrayVec;
use log::info;

use crate::challenge::{Solution, eval_score, MAX_PORTS};
use crate::graph::{Graph, VertexId};

// Size needed for an array indexed by masks.
const NUM_MASKS: usize = 1 << MAX_PORTS;

// Mask of whether a city 'i' was seen (bit (1 << i)).
// Must set the lower bits, since this is used directly as an index in an array.
type Mask = u32;
type Cost = u16;

struct Tour {
    cost: Cost,  // in terms of total distance
    vertices: Vec<VertexId>,
}

pub fn held_karp(graph: &Graph) -> Option<Solution> {
    let tour = HeldKarp::new().traveling_salesman(graph);
    tour.verify_tour(graph);
    tour.to_solution(graph)
}

// Costs g(S, e) for a fixed 'S', for possible values of 'e'.
type SubsetCosts = [Cost; MAX_PORTS];
// Pointer to the before-last city visited for a given mask.
// I.e., the last city that we used to reach the given last city (index) 'e'.
type Predecessors = [VertexId; MAX_PORTS];
// Pointer to the city we started with, for a given mask/last city.
type Starts = [VertexId; MAX_PORTS];

struct HeldKarp {
    // This is the g(S, e) that gives us, for a given 'S', the minimal cost of
    // going through all cities in 'S', then going to possible 'e' cities.
    g: Vec<SubsetCosts>,
    // This is p(S, e) that gives us, for a given 'S', the predecessor city that
    // we visited to end up to each possible 'e' cities.
    p: Vec<Predecessors>,
    // Because our starting vertex influences edge costs, on top of Held-Karp we
    // consider all possible starts in our search and keep track for a given
    // g(S, e) what starting node was used.
    s: Vec<Starts>,
}

// Generate the lexicographically next bit permutation for a pattern of N bits
// set to 1. E.g. for N=3, if the bit pattern is 00010011, the next patterns are
// 00010101, 00010110, 00011001, etc.
fn next_mask(mask: Mask) -> Mask {
    // https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    let v = mask as i32;
    let t = (v | (v - 1)) as i32;
    let w = (t + 1) | (((!t & -!t) - 1) >> (v.trailing_zeros() + 1));
    w as Mask
}

impl HeldKarp {
    pub fn new() -> Self {
        HeldKarp {
            g: vec![[0; MAX_PORTS]; NUM_MASKS],
            p: vec![[0; MAX_PORTS]; NUM_MASKS],
            s: vec![[0; MAX_PORTS]; NUM_MASKS],
        }
    }

    pub fn traveling_salesman(&mut self, graph: &Graph) -> Tour {
        let start_tick = graph.start_tick + 1;  // time to dock spawn

        // Initialize for |S|=1 (with S = {k})
        for k in 0..graph.ports.len() {
            let k = k as VertexId;
            let mask = (1 << k) as Mask;
            // Initial cost is just the start tick (+1 to dock spawn)
            self.g[mask as usize][k as usize] = start_tick + 1;
            self.p[mask as usize][k as usize] = k;
            self.s[mask as usize][k as usize] = k;
        }

        // From |S|=s, deduce S' (|S'| = s+1) g(S', k) by picking the
        // before-last city that minimizes cost, for each k.
        let max_submask_items = graph.ports.len();
        for s in 2..=max_submask_items {
            // Gives a mask with the 's' last bits set to 1.
            let mut subset_mask = ((1 << s) - 1) as Mask;
            // Done when all bits are to the left
            let end_mask = (subset_mask << (max_submask_items - s)) as Mask;
            while subset_mask <= end_mask {
                // TODO: just do combinations on the array...?
                // TODO: get the ones set with trailing_zeros() + (x&(x-1)) to
                //       clear, repeat s times?
                let set: ArrayVec<_, MAX_PORTS> = ArrayVec::from_iter(
                    (0..graph.ports.len()).filter(|&k| {
                        let mask = (1 << k) as Mask;
                        subset_mask & mask != 0
                    }));
                for k in &set {
                    let k = *k as VertexId;
                    let k_mask = (1 << k) as Mask;
                    let no_k_submask = subset_mask & !k_mask;
                    let (m, m_cost, start) = set.iter()
                        .filter(|&m| (*m as VertexId) != k).map(|&m| {
                        let m = m as VertexId;
                        let current_cost = self.g[no_k_submask as usize][m as usize];
                        let tick = current_cost;
                        let tick_offset = graph.tick_offset(tick);
                        let m_k_cost = graph.cost(tick_offset, m, k);
                        let cost = current_cost + (m_k_cost as Cost);
                        let start = self.s[no_k_submask as usize][m as usize];
                        (m, cost, start)
                    }).min_by_key(|&(_, cost, _)| cost).unwrap();
                    self.g[subset_mask as usize][k as usize] = m_cost + 1;
                    self.p[subset_mask as usize][k as usize] = m;
                    self.s[subset_mask as usize][k as usize] = start;
                }

                subset_mask = next_mask(subset_mask);
            }
        }

        // Find the best tour by checking paths back to each start.
        let all_mask = ((1 << max_submask_items) - 1) as Mask;
        let (last_city, total_cost, start) = (0..graph.ports.len()).map(|k| {
            let k = k as VertexId;
            let current_cost = self.g[all_mask as usize][k as usize];
            let start = self.s[all_mask as usize][k as usize];
            let tick = current_cost;
            let tick_offset = graph.tick_offset(tick);
            let k_start_cost = graph.cost(tick_offset, k, start);
            // Note: no "+ 1" for docking, the last dock tick doesn't count.
            let cost = current_cost + (k_start_cost as Cost);
            (k, cost, start)
        }).min_by_key(|&(_, cost, _)| cost).unwrap();

        // Backtrack to get vertices.
        let mut vertices = Vec::with_capacity(graph.ports.len() + 1);
        vertices.push(start);
        let mut vertex = last_city;
        let mut mask = all_mask;
        while mask != 0 {
            vertices.push(vertex);
            let prev_vertex = vertex;
            vertex = self.p[mask as usize][vertex as usize];
            mask &= !(1 << prev_vertex);
        }
        vertices.reverse();
        info!("Starting at port ID {}, cost would be {}, vertices: {vertices:?}",
              start, total_cost);
        Tour { cost: total_cost, vertices: vertices, }
    }

    // Use for debugging
    fn _show(&self, graph: &Graph, mask: Mask, e: VertexId) -> String {
        let g = self.g[mask as usize][e as usize];
        let p = self.p[mask as usize][e as usize];
        let s = self.s[mask as usize][e as usize];
        format!("g({mask}, {e}) = {g}   p({mask}, {e}) = {p}   s({mask}, {e}) = {s}",
                mask = self._show_mask(graph, mask),
                e = e + 1, p = p + 1, s = s + 1)
    }

    // Use for debugging
    fn _show_mask(&self, graph: &Graph, mask: Mask) -> String {
        let mask_vertices: Vec<_> = (0..graph.ports.len())
            .filter(|&v| (1 << v) as Mask & mask != 0)
            .map(|v| (v+1).to_string()).collect();
        format!("{{{}}}", mask_vertices.join(","))
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
            let offset = tick % (graph.tick_offsets as u16);
            tick += graph.cost(offset as u8, from, to) as u16 + 1;
        }
        tick -= 1;
        assert!(tick == self.cost, "Tour would give cost {}, but we got {}",
                tick, self.cost);
    }
}
