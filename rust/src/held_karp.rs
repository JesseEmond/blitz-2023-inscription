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

use crate::challenge::{Solution, MAX_PORTS};
use crate::graph::{Graph, VertexId};
use crate::pathfinding::{Pos};

// Size needed for an array indexed by masks.
// We exclude the currently search city, so num_cities - 1.
const NUM_MASKS: usize = 1 << (MAX_PORTS - 1);

// Mask of whether a city 'i' was seen (bit (1 << i)).
// Must set the lower bits, since this is used directly as an index in an array.
type SeenMask = u32;

struct Tour {
    cost: u32,  // in terms of total distance
    vertices: Vec<VertexId>,
}

pub fn held_karp(graph: &Graph) -> Solution {
    let mut held_karp = HeldKarp::new();
    (0..graph.ports.len())
        .map(|start| held_karp.traveling_salesman(graph, start as VertexId))
        .min_by_key(|tour| tour.cost).unwrap().to_solution()
}

type Cost = u16;
// Costs g(S, e) for a fixed 'S', for possible values of 'e'.
type SubsetCosts = [Cost; MAX_PORTS];

struct HeldKarp {
    // This is the g(S, e) that gives us, for a given 'S', the minimal cost of
    // going through all cities in 'S', then going to possible 'e' cities.
    g: Vec<SubsetCosts>,
}

// Generate the lexicographically next bit permutation for a pattern of N bits
// set to 1. E.g. for N=3, if the bit pattern is 00010011, the next patterns are
// 00010101, 00010110, 00011001, etc.
fn next_mask(mask: SeenMask) -> SeenMask {
    // https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    let v = mask as i32;
    let t = (v | (v - 1)) as i32;
    let w = (t + 1) | (((!t & -!t) - 1) >> (v.trailing_zeros() + 1));
    w as SeenMask
}

impl HeldKarp {
    pub fn new() -> Self {
        HeldKarp { g: vec![[0; MAX_PORTS]; NUM_MASKS] }
    }

    pub fn traveling_salesman(
        &mut self, graph: &Graph, start: VertexId
        ) -> Tour {
        // All submasks are in a space where 'start' is excluded, so we
        // "translate" the ids > start to 'id - 1'.
        let translate: [u8; MAX_PORTS] =
            (0..graph.ports.len()).map(|v| {
                let v = v as VertexId;
                if v < start { v } else if v > start { v - 1 } else { v }
            }).collect::<Vec<_>>().try_into().unwrap();
        let others: [VertexId; MAX_PORTS - 1] = graph.others(start)
            .collect::<Vec<_>>().try_into().unwrap();
            
        let tick = graph.start_tick + 1;  // time to dock spawn
        for k in others {
            let mask = 1 << (translate[k as usize] as SeenMask);
            // TODO: adjacency matrix
            // self.g[k as usize][mask as usize] = graph.
            self.g[mask as usize][k as usize] = 0;
        }
        //TODO continue
        Tour { cost: 0, vertices: Vec::new() }
    }
}

impl Tour {
    fn to_solution(&self) -> Solution {
        // TODO
        Solution { score: 0, spawn: Pos { x: 0, y: 0 }, paths: Vec::new() }
    }
}
