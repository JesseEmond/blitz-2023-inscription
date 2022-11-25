// Implementation of held_karp.rs, without optimizations.

use std::sync::{mpsc, Arc};
use std::thread;

use crate::challenge::{Solution, eval_score};
use crate::challenge_consts::{MAX_PORTS, NUM_THREADS};
use crate::simple_graph::{SimpleGraph, VertexId};

const MAX_MASK_ITEMS: usize = MAX_PORTS - 1;
// Size needed for an array indexed by masks.
const NUM_MASKS: usize = 1 << MAX_MASK_ITEMS;

// We flatten arrays conceptually indexed by a mask 'S' and vertex 'e' to a
// flat array where masks of similar sizes (number of '1' bits) are close in
// memory. This is the size of that flattened array.
const FLAT_SIZE: usize = (MAX_PORTS - 1) * NUM_MASKS;

type Cost = u16;
// Note: use array instead of ArrayVec since it can be okay to access past the
// size on last iterations of set combinations.
type SetElements = [VertexId; MAX_PORTS];

struct Tour {
    cost: Cost,  // in terms of total distance
    vertices: Vec<VertexId>,
}

fn held_karp(graph: &Arc<SimpleGraph>) -> Option<Solution> {
    let mut best_tour = Tour { cost: Cost::MAX, vertices: Vec::new() };
    let mut handles = vec![];
    let (tx, rx) = mpsc::channel();
    for i in 0..NUM_THREADS {
        let tx = tx.clone();
        let graph = graph.clone();
        handles.push(thread::spawn(move || {
            let mut held_karp = HeldKarp::new();
            for start in (i..graph.ports.len()).step_by(NUM_THREADS) {
                let start = start as VertexId;
                let full_tour = held_karp.traveling_salesman(&graph, start);
                tx.send(full_tour).unwrap();
            }
        }));
    }
    drop(tx);  // Drop the last sender, wait until all threads are done.
    while let Ok(tour) = rx.recv() {
        if tour.score() > best_tour.score() {
            best_tour = tour;
        }
    }
    for handle in handles {
        handle.join().unwrap();
    }
    best_tour.to_solution(&graph)
}

struct HeldKarp {
    /// g(S, e) min cost of going through all nodes in 'S', ending in 'e'.
    /// Flattened array, see FlatSubsetsHelper for indexing in it.
    g: Vec<Cost>,
    /// p(S, e) predecessor to 'e' of going through nodes in 'S', ending in 'e'.
    /// Used when backtracking.
    /// Flattened array, see FlatSubsetsHelper for indexing in it.
    p: Vec<VertexId>,

    /// Helper to manipulate an array of combinations, ordered by subset size.
    flat_subsets_helper: FlatSubsetsHelper,
}

/// Helper for precomputed values for dealing with a precomputed flat array of
/// combinations, ordered by subset sizes.
struct FlatSubsetsHelper {
    /// Precomputed binomial values n-choose-k.
    pub binomial: Binomial,
    /// For a subset size, the start index in a flat array for the subset data.
    subset_starts: [usize; MAX_PORTS],
}

/// Precomputed binomial values (n-choose-k).
struct Binomial {
    /// binomials[n][k] gives the value for n-choose-k.
    binomials: [[usize; MAX_PORTS]; MAX_PORTS],
}

/// Helper to hold a current set of vertices ('S' or 'S \ {k}' in Held-Karp).
struct Combination {
    pub elements: SetElements,
    pub size: usize,
    /// In a sequential array of all possible subsets, base index for this set.
    /// Element 0 is at 'index_base', element 1 at 'index_base+1', ...
    pub index_base: usize,
}

impl HeldKarp {
    pub fn new() -> Self {
        HeldKarp {
            g: Vec::new(),
            p: Vec::new(),
            flat_subsets_helper: FlatSubsetsHelper::new(),
        }
    }

    pub fn traveling_salesman(
        &mut self, graph: &SimpleGraph, start: VertexId) -> Tour {
        let start_tick = graph.start_tick + 1;  // time to dock spawning port
        self.g = vec![Cost::MAX; FLAT_SIZE];
        self.p = vec![VertexId::MAX; FLAT_SIZE];

        let max_set_items = graph.ports.len() - 1;

        // For |S|=1 (S={k}), smallest cost is the cost of start->k.
        let mut set = Combination::first(/*size=*/1, &self.flat_subsets_helper);
        while !set.is_done(max_set_items) {
            let k = *set.elements.first().unwrap();
            let cost = graph.cost(
                graph.tick_offset(start_tick),
                start, self.untranslate(start, k as VertexId)) as Cost;
            // +1 to dock
            self.g[set.element_index(0)] = start_tick + cost + 1;
            self.p[set.element_index(0)] = k;
            set.next(&self.flat_subsets_helper);
        }

        // For |S|=s, smallest cost depends on |S'|=s-1 values of g(S', k).
        for s in 2..=max_set_items {
            let mut set = Combination::first(s, &self.flat_subsets_helper);
            while !set.is_done(max_set_items) {
                let mut set_minus_k = Combination::set_minus_k(
                    &set, &self.flat_subsets_helper);
                for k in 0..s {
                    let (min_cost, min_vertex) = (0..(s-1)).map(|m| {
                        let current_cost = self.g[set_minus_k.element_index(m)];
                        let m_k_cost = graph.cost(
                            graph.tick_offset(current_cost),
                            self.untranslate(start, set_minus_k.elements[m]),
                            self.untranslate(start, set.elements[k])) as Cost;
                        let cost = current_cost + m_k_cost + 1;  // +1 to dock
                        (cost, set_minus_k.elements[m])
                    }).min_by_key(|&(cost, _)| cost).unwrap();
                    self.g[set.element_index(k)] = min_cost;
                    self.p[set.element_index(k)] = min_vertex;
                    set_minus_k.next_minus_k(k, set.elements[k],
                                             &self.flat_subsets_helper);
                }
                set.next(&self.flat_subsets_helper);
            }
        }

        // Find the best tour by checking paths back to the start.
        let set = Combination::first(max_set_items, &self.flat_subsets_helper);
        let (total_cost, last_city) = (0..max_set_items).map(|k| {
            let current_cost = self.g[set.element_index(k)];
            let k_start_cost = graph.cost(
                graph.tick_offset(current_cost),
                self.untranslate(start, set.elements[k]), start) as Cost;
            // Note: no +1 for docking, the last dock tick doesn't count.
            let cost = current_cost + k_start_cost;
            (cost, set.elements[k])
        }).min_by_key(|&(cost, _)| cost).unwrap();

        let vertices = self.backtrack(start, last_city, max_set_items);
        Tour { cost: total_cost, vertices }
    }

    fn backtrack(&self, start: VertexId, last: VertexId,
                 set_size: usize) -> Vec<VertexId> {
        let mut set = Combination::first(set_size, &self.flat_subsets_helper);
        let mut vertices = Vec::with_capacity(set_size + 2);
        vertices.push(start);
        let mut vertex = last;
        for _s in (1..=set_size).rev() {
            vertices.push(self.untranslate(start, vertex));
            let vertex_idx = set.find_index(vertex).unwrap();
            vertex = self.p[set.element_index(vertex_idx)];
            set.remove(vertex_idx, &self.flat_subsets_helper);
        }
        vertices.push(start);
        vertices.reverse();
        vertices
    }

    /// Sets are in a space where 'start' is removed. This utility converts back
    /// to the proper vertex IDs.
    fn untranslate(&self, start: VertexId, v: VertexId) -> VertexId {
        if v < start { v } else { v + 1 }
    }
}

impl FlatSubsetsHelper {
    fn new() -> Self {
        let binomial = Binomial::new();
        let mut subset_starts = [0usize; MAX_PORTS];
        for i in 1..MAX_PORTS {
            let prev_size = (i-1) * binomial.n_choose_k(MAX_MASK_ITEMS, i-1);
            subset_starts[i] = subset_starts[i-1] + prev_size;
        }
        FlatSubsetsHelper { binomial, subset_starts }
    }

    /// Index of the start where a given 'set' would be in a flattened array.
    /// The elements of 'set' will be there, sequentially.
    fn flat_index(&self, set: &SetElements, size: usize) -> usize {
        let mut loc = 0usize;
        for i in 0..size {
            loc += self.binomial.n_choose_k(set[i] as usize, i+1);
        }
        self.flat_index_region(size) + size * loc
    }

    /// Start index where combinations of a given size would be in a flat array.
    /// The elements of combinations of that size will be there, sequentially.
    fn flat_index_region(&self, combination_size: usize) -> usize {
        self.subset_starts[combination_size]
    }
}

impl Binomial {
    fn new() -> Self {
        let mut binomial = [[0usize; MAX_PORTS]; MAX_PORTS];
        for n in 0..MAX_PORTS {
            binomial[n][0] = 1;
            binomial[n][n] = 1;
        }
        // Use recurrence relationship
        // n-choose-k = (n-1)-choose-(k-1) + (n-1)-choose-k
        for n in 1..MAX_PORTS {
            for k in 1..n {
                binomial[n][k] = binomial[n-1][k-1] + binomial[n-1][k];
            }
        }
        Binomial { binomials: binomial }
    }

    fn n_choose_k(&self, n: usize, k: usize) -> usize {
        self.binomials[n][k]
    }
}

impl Combination {
    /// Returns the first set combination for a given subset size.
    fn first(size: usize, flat_subsets_helper: &FlatSubsetsHelper) -> Self {
        let elements = array_init::array_init(|i| i as VertexId);
        let index_base = flat_subsets_helper.flat_index_region(size);
        Combination { elements, size, index_base }
    }

    /// Creates a set 'S' with one less element (initially without 0th elem).
    /// Use this in conjunction with 'next_minus_k'.
    fn set_minus_k(set: &Combination,
                   flat_subsets_helper: &FlatSubsetsHelper) -> Self {
        let mut elements = [0; MAX_PORTS];
        for i in 1..set.size {
            elements[i - 1] = set.elements[i];
        }
        let size = set.size - 1;
        let index_base = flat_subsets_helper.flat_index(&elements, size);
        Combination { elements, size, index_base }
    }

    /// Returns the next combination of the same size.
    fn next(&mut self, flat_subsets_helper: &FlatSubsetsHelper) {
        let mut i = 0;
        while i < self.size - 1 && self.elements[i] + 1 == self.elements[i + 1] {
            self.elements[i] = i as VertexId;
            i += 1;
        }
        self.elements[i] += 1;
        self.index_base = flat_subsets_helper.flat_index(
            &self.elements, self.size);
    }

    /// Go to the next version of set 'S' with its 'k'th element missing.
    fn next_minus_k(&mut self, k: usize, kth_elem: VertexId,
                    flat_subsets_helper: &FlatSubsetsHelper) {
        // Note that the 'minus-k' version of S starts with {1, ..., s-1}, so we
        // go to the next one by setting the kth element of 'S\{k}' to be S[k],
        // which gives us the next k being missing (e.g. for k=0, becomes
        // {0, 2, ..., s-1}).
        let s = self.size;
        // Do flat index computations in-place on only the edited element as an
        // optimization.
        self.index_base -= s * flat_subsets_helper.binomial.n_choose_k(
            self.elements[k] as usize, k+1);
        self.elements[k] = kth_elem;
        self.index_base += s * flat_subsets_helper.binomial.n_choose_k(
            self.elements[k] as usize, k+1);
    }

    /// If we are past the final combination of our current size.
    fn is_done(&self, max_elements: usize) -> bool {
        let last = self.elements[self.size - 1] as usize;
        last >= max_elements
    }

    /// Return index in a flattened array for the requested set element.
    fn element_index(&self, set_idx: usize) -> usize {
        self.index_base + set_idx
    }

    fn remove(&mut self, idx: usize, flat_subsets_helper: &FlatSubsetsHelper) {
        for i in (idx+1)..self.size {
            self.elements[i-1] = self.elements[i];
        }
        self.size -= 1;
        self.index_base = flat_subsets_helper.flat_index(
            &self.elements, self.size);
    }

    fn find_index(&self, vertex: VertexId) -> Option<usize> {
        self.elements.iter().position(|&v| v == vertex)
    }
}

impl Tour {
    pub fn to_solution(&self, graph: &SimpleGraph) -> Option<Solution> {
        if self.cost < graph.max_ticks {
            let spawn = graph.ports[self.vertices[0] as usize];
            let mut paths = Vec::with_capacity(self.vertices.len() - 1);
            let mut tick = graph.start_tick + 1;  // +1 to dock spawn
            for i in 1..self.vertices.len() {
                let prev = self.vertices[i-1];
                let next = self.vertices[i];
                let path = graph.path(graph.tick_offset(tick), prev, next);
                paths.push(path.clone());
                tick += path.cost + 1;  // +1 to dock it
            }
            Some(Solution { score: self.score(), spawn, paths })
        } else {
            None
        }
    }

    pub fn score(&self) -> i32 {
        if self.vertices.is_empty() {
            return 0;
        }
        let visits = self.vertices.len();
        let ticks = self.cost;
        eval_score(visits as u32, ticks, /*looped=*/true)
    }
}

pub struct SimpleExactTspSolver;

impl SimpleExactTspSolver {
    pub fn do_solve(&mut self, graph: &Arc<SimpleGraph>) -> Option<Solution> {
        held_karp(&graph)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value};
    use std::fs;
    use crate::game_interface::{GameTick};
    use crate::graph::Graph;
    use crate::solvers::{ExactTspSolver, Solver};
    use super::*;

    fn make_game() -> Arc<GameTick> {
        // Note this isn't great, we ideally shouldn't read from disk here.
        let game_file = "./games/35334.json";
        let game_json = fs::read_to_string(game_file)
            .expect("Couldn't read game file");
        let parsed: Value = serde_json::from_str(&game_json)
            .expect("Couldn't parse JSON in game file");
        let game = serde_json::from_value(parsed)
            .expect("Couldn't parse game tick in game file");
        Arc::new(game)
    }

    #[test]
    fn test_match_held_karp() {
        let game = make_game();
        let slow_graph = Arc::new(SimpleGraph::new(&game));
        let fast_graph = Arc::new(Graph::new(&game));
        let mut slow = SimpleExactTspSolver{};
        let mut fast = ExactTspSolver{};
        let slow_sln = slow.do_solve(&slow_graph).unwrap();
        let fast_sln = fast.do_solve(&fast_graph).unwrap();
        println!("slow: spawn {:?} path {:?} score {}",
                 slow_sln.spawn, slow_sln.paths, slow_sln.score);
        println!("fast: spawn {:?} path {:?} score {}",
                 fast_sln.spawn, fast_sln.paths, fast_sln.score);
        assert_eq!(slow_sln.score, fast_sln.score);
    }
}
