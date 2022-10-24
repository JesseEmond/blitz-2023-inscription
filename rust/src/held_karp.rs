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
use log::{info};

use crate::challenge::{Solution, eval_score, MAX_PORTS};
use crate::graph::{Graph, VertexId};
use crate::pathfinding::{Pos};

// Size needed for an array indexed by masks.
// We exclude the currently search city, so num_cities - 1.
const NUM_MASKS: usize = 1 << (MAX_PORTS - 1);

// Mask of whether a city 'i' was seen (bit (1 << i)).
// Must set the lower bits, since this is used directly as an index in an array.
type Mask = u32;
type Cost = u16;

struct Tour {
    cost: Cost,  // in terms of total distance
    vertices: Vec<VertexId>,
}

pub fn held_karp(graph: &Graph) -> Option<Solution> {
    let mut held_karp = HeldKarp::new();
    // TODO: log score for each
    (0..graph.ports.len())
        .map(|start| held_karp.traveling_salesman(graph, start as VertexId))
        .min_by_key(|tour| tour.cost).unwrap().to_solution(graph)
}

// Costs g(S, e) for a fixed 'S', for possible values of 'e'.
type SubsetCosts = [Cost; MAX_PORTS - 1];
// Pointer to the before-last city visited for a given mask.
// I.e., the last city that we used to reach the given last city (index) 'e'.
type Predecessors = [VertexId; MAX_PORTS - 1];

struct HeldKarp {
    // This is the g(S, e) that gives us, for a given 'S', the minimal cost of
    // going through all cities in 'S', then going to possible 'e' cities.
    g: Vec<SubsetCosts>,
    // This is p(S, e) that gives us, for a given 'S', the predecessor city that
    // we visited to end up to each possible 'e' cities.
    p: Vec<Predecessors>,
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
            g: vec![[0; { MAX_PORTS - 1 }]; NUM_MASKS],
            p: vec![[0; { MAX_PORTS - 1 }]; NUM_MASKS],
        }
    }

    pub fn traveling_salesman(
        &mut self, graph: &Graph, start: VertexId
        ) -> Tour {
        // All submasks are in a space where 'start' is excluded to take up 
        // less space, so we "translate" the ids > start to 'id - 1'.
        // TODO: store as precomputed arrays?
        let to_relative = |v: VertexId| if v <= start { v } else { v - 1 };
        let from_relative = |v: VertexId| if v < start { v } else { v + 1 };
        // TODO: better to loop on it?
        let others: ArrayVec<VertexId, { MAX_PORTS - 1 }> = ArrayVec::from_iter(
            graph.others(start).map(|k| to_relative(k)));

        let start_tick = graph.start_tick + 1;  // time to dock spawn
            
        // Initialize for |S|=1 (with S = {k})
        for k in &others {
            let mask = (1 << *k) as Mask;
            let cost = graph.cost(
                graph.tick_offset(start_tick), start, from_relative(*k));
            // Initial cost must take into account the start tick
            self.g[mask as usize][*k as usize] = start_tick + (cost + 1) as Cost;
            self.p[mask as usize][*k as usize] = *k;
        }

        // From |S|=s, deduce S' (|S'| = s+1) g(S', k) by picking the
        // before-last city that minimizes cost, for each k.
        let max_submask_items = graph.ports.len() - 1;
        for s in 2..=max_submask_items {
            // Gives a mask with the 's' last bits set to 1.
            let mut subset_mask = ((1 << s) - 1) as Mask;
            // Done when all bits are to the left
            let end_mask = (subset_mask << (max_submask_items - s)) as Mask;
            while subset_mask <= end_mask {
                // TODO: just do combinations on the array...?
                // TODO: get the ones set with trailing_zeros() + (x&(x-1)) to
                //       clear, repeat s times?
                let set: ArrayVec<_, { MAX_PORTS - 1 }> = ArrayVec::from_iter(
                    others.iter().filter(|&k| {
                        let mask = (1 << k) as Mask;
                        subset_mask & mask != 0
                    }));
                for k in &set {
                    let k = *k;
                    let k_mask = (1 << *k) as Mask;
                    let no_k_submask = subset_mask & !k_mask;
                    let (m, m_cost) = set.iter().filter(|&m| **m != *k).map(|&m| {
                        let current_cost = self.g[no_k_submask as usize][*m as usize];
                        let tick = current_cost;
                        let tick_offset = graph.tick_offset(tick);
                        let m_k_cost = graph.cost(tick_offset, from_relative(*m),
                                                  from_relative(*k));
                        let cost = current_cost + (m_k_cost as Cost);
                        (*m, cost)
                    }).min_by_key(|&(_, cost)| cost).unwrap();
                    self.g[subset_mask as usize][*k as usize] = m_cost + 1;
                    self.p[subset_mask as usize][*k as usize] = m;
                }

                subset_mask = next_mask(subset_mask);
            }
        }

        // Find the best tour by checking paths back to 'start'.
        let all_mask = ((1 << max_submask_items) - 1) as Mask;
        let (last_city, total_cost) = others.iter().map(|&k| {
            let current_cost = self.g[all_mask as usize][k as usize];
            let tick = current_cost;
            let tick_offset = graph.tick_offset(tick);
            let k_start_cost = graph.cost(tick_offset, from_relative(k), start);
            // Note: no "+ 1" for docking, the last dock tick doesn't count.
            let cost = current_cost + (k_start_cost as Cost);
            (k, cost)
        }).min_by_key(|&(_, cost)| cost).unwrap();

        // Backtrack to get vertices.
        let mut vertices = Vec::with_capacity(graph.ports.len() + 1);
        vertices.push(start);
        let mut vertex = last_city;
        let mut mask = all_mask;
        while mask != 0 {
            vertices.push(from_relative(vertex));
            let prev_vertex = vertex;
            vertex = self.p[mask as usize][vertex as usize];
            mask &= !(1 << prev_vertex);
        }
        vertices.push(start);
        vertices.reverse();
        Tour { cost: total_cost, vertices: vertices, }
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
}
