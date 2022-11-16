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

// Note: use array instead of ArrayVec since it can be okay to access past the
// size on last iterations of set combinations.
type SetElements = [VertexId; MAX_PORTS];
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
        self.subset_starts[size] + size * loc
    }

    /// Start index where combinations of a given size would be in a flat array.
    /// The elements of combinations of that size will be there, sequentially.
    fn flat_index_region(&self, combination_size: usize) -> usize {
        self.subset_starts[combination_size]
    }
}

impl Binomial {
    fn new() -> Self {
        let mut binomial =  [[0usize; MAX_PORTS]; MAX_PORTS];
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
        Combination { elements, index_base, size }
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
        Combination { elements, index_base, size }
    }

    /// Direct construction. used for debugging purposes.
    fn _new(elements: &SetElements, size: usize,
            flat_subsets_helper: &FlatSubsetsHelper) -> Self {
        let index_base = flat_subsets_helper.flat_index(elements, size);
        Combination { elements: elements.clone(), index_base, size }
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
    /// Doing incremental to move to the next set
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

impl HeldKarp {
    pub fn new() -> Self {
        HeldKarp {
            g: vec![Cost::MAX; FLAT_SIZE],
            p: vec![VertexId::MAX; FLAT_SIZE],
            flat_subsets_helper: FlatSubsetsHelper::new(),
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

        let max_set_items = graph.ports.len() - 1;

        // Initialize for |S|=1 (with S = {k})
        let mut set = Combination::first(/*size=*/1, &self.flat_subsets_helper);
        while !set.is_done(max_set_items) {
            let k = *set.elements.first().unwrap();
            // Start at +1 to dock spawn.
            // Initial cost is the start tick, plus cost from spawn to 'k'.
            let cost = graph.cost(graph.tick_offset(start_tick), start,
                                  untranslated[k as usize]);
            // +1 to dock 'k'
            self.g[set.element_index(0)] = start_tick + (cost as Cost) + 1;
            self.p[set.element_index(0)] = k;
            set.next(&self.flat_subsets_helper);
        }

        // From |S|=s, deduce S' (|S'| = s+1) g(S', k) by picking the
        // before-last city that minimizes cost, for each k.
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
                            untranslated[set_minus_k.elements[m] as usize],
                            untranslated[set.elements[k] as usize]);
                        // + 1 to dock
                        let cost = current_cost + (m_k_cost as Cost) + 1;
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
        let mut total_cost = Cost::MAX;
        let mut last_city: VertexId = VertexId::MAX;
        for k in 0..max_set_items {
            let current_cost = self.g[set.element_index(k)];
            let k_start_cost = graph.cost(
                graph.tick_offset(current_cost),
                untranslated[set.elements[k] as usize],
                start);
            // Note: no "+ 1" for docking, the last dock tick doesn't count.
            let cost = current_cost + (k_start_cost as Cost);
            if cost < total_cost {
                total_cost = cost;
                last_city = set.elements[k];
            }
        }
        let vertices = self.backtrack(start, last_city, max_set_items);
        info!("Starting at port ID {}, cost would be {}, vertices: {vertices:?}",
              start, total_cost);
        Tour { cost: total_cost, vertices }
    }

    // Backtrack to get vertices for a tour.
    fn backtrack(
        &self, start: VertexId, last: VertexId, set_size: usize
        ) -> Vec<VertexId> {
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

    // Sets are in the space where the 'start' vertex is removed, this utility
    // is used to convert back to the vertex IDs.
    fn untranslate(&self, start: VertexId, v: VertexId) -> VertexId {
        if v < start { v } else { v + 1 }
    }

    // Checks every subset of ports to see if visiting these would give a better
    // score than the best full tour found. This is slow.
    pub fn consider_shorter_tours(&self, graph: &Graph, start: VertexId,
                                  full_tour: &Tour) -> Tour {
        let full_tour_score = full_tour.score(graph);
        let mut best_tour = full_tour.clone();
        let mut best_score = full_tour_score;
        // Note: only considering tours that loop (i.e. each city is worth
        // 150 * 2 pts). 
        let min_ports = ((best_score + 299) / 300) as usize;

        let min_set_items = min_ports - 1;  // -1 since start is implicit
        let max_set_items = graph.ports.len() - 1; // no need t
        // Note: max_set_items excluded since we already have full_tour.
        for s in min_set_items..max_set_items {
            let mut set = Combination::first(s, &self.flat_subsets_helper);
            while !set.is_done(max_set_items) {
                for k in 0..s {
                    let current_cost = self.g[set.element_index(k)];
                    let k_start_cost = graph.cost(
                        graph.tick_offset(current_cost),
                        self.untranslate(start, set.elements[k]), start);
                    // Note: no '+1' for final dock -- it's free.
                    let cost = current_cost + (k_start_cost as Cost);
                    let visits = s + 2;  // +2 for the start port (looped)
                    let score = eval_score(visits as u32, cost,
                                           /*looped=*/true);
                    if score > best_score {
                        let vertices = self.backtrack(start, set.elements[k], s);
                        best_tour = Tour { cost, vertices };
                        best_score = score;
                    }
                }
                set.next(&self.flat_subsets_helper);
            }
        }

        if best_tour.cost < full_tour.cost {
            info!("New best! Start@{}, cost={}, score={}(>{}), ports={}, vertices: {:?}",
                  start, best_tour.cost, best_score, full_tour_score,
                  best_tour.vertices.len()-1, best_tour.vertices);
        }
        best_tour
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
        let mut our_set = array_init::array_init(|_| VertexId::MAX);
        let mut tour_tick = graph.start_tick + 1;
        let mut tour_set = array_init::array_init(|_| VertexId::MAX);
        for i in 1..tour.vertices.len() {
            info!("Target tour tick: {}", tour_tick);
            let prev_vertex = tour.vertices[i-1];
            let vertex = tour.vertices[i];
            let cost = graph.cost(graph.tick_offset(tour_tick), prev_vertex,
                                  vertex);
            info!("Target tour picked: {}->{} (cost {}+1)", prev_vertex+1,
                  vertex+1, cost);
            tour_tick += (cost as u16) + 1;
            tour_set[i-1] = translate(vertex);
            if vertex != start {
                let set = Combination::_new(&tour_set, i,
                                            &self.flat_subsets_helper);
                info!("{}", self._show(&set, i-1, start));
            }

            info!("Our tick: {}", our_tick);
            let our_prev_vertex = our_tour.vertices[i-1];
            let our_vertex = our_tour.vertices[i];
            let our_cost = graph.cost(graph.tick_offset(our_tick),
                                      our_prev_vertex, our_vertex);
            info!("Our tour picked: {}->{} (cost {}+1)", our_prev_vertex+1,
                  our_vertex+1, our_cost);
            our_tick += (our_cost as u16) + 1;
            our_set[i - 1] = translate(our_vertex);
            if our_vertex != start {
                let set = Combination::_new(&our_set, i,
                                            &self.flat_subsets_helper);
                info!("{}", self._show(&set, i-1, start));
            }
            info!("-----------------------");
        }
        panic!("Debug logs before here.");
    }

    // Use for debugging
    fn _show(&self, set: &Combination, e_idx: usize, start: VertexId) -> String {
        let idx = set.element_index(e_idx);
        info!("looking up idx={} with s={}", idx, set.size);
        let g = self.g[idx];
        let p = self.p[idx];
        format!("g({mask}, {e}) = {g}   p({mask}, {e}) = {p}",
                mask = self._show_set(set, start),
                e = self.untranslate(start, set.elements[e_idx]) + 1,
                p = self.untranslate(start, p) + 1)
    }

    // Use for debugging
    fn _show_set(&self, set: &Combination, start: VertexId) -> String {
        let set_vertices: Vec<_> = (0..set.size)
            .map(|i| (self.untranslate(start, set.elements[i])+1).to_string())
            .collect();
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

    // Number of vertices in our path that would go past the max_ticks (and thus
    // don't count as visits).
    fn num_vertices_after_max(&self, graph: &Graph) -> usize {
        let mut tick = graph.start_tick + 1;  // +1 to dock start
        for i in 1..self.vertices.len() {
            let from = self.vertices[i-1];
            let to = self.vertices[i];
            // +1 to dock port
            let cost = graph.cost(graph.tick_offset(tick), from, to);
            tick += (cost as u16) + 1;
            if tick > graph.max_ticks {
                return self.vertices.len() - i;
            }
        }
        0
    }

    fn score(&self, graph: &Graph) -> i32 {
        if self.vertices.is_empty() {
            return 0;
        }
        let overflow = self.num_vertices_after_max(graph);
        let visits = self.vertices.len() - overflow;
        let did_loop = overflow == 0 &&
            self.vertices.last().unwrap() == self.vertices.first().unwrap();
        let ticks = if did_loop { self.cost } else { graph.max_ticks };
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
