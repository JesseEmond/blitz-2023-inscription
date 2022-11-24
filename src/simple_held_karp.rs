// Implementation of held_karp.rs, without optimizations.

use itertools::Itertools;
use std::sync::{Arc};

use crate::challenge::{Solution, eval_score};
use crate::challenge_consts::{MAX_PORTS};
use crate::simple_graph::{SimpleGraph, VertexId};

const MAX_MASK_ITEMS: usize = MAX_PORTS;
// Size needed for an array indexed by masks.
const NUM_MASKS: usize = 1 << MAX_MASK_ITEMS;

type Cost = u16;

struct Tour {
    cost: Cost,  // in terms of total distance
    vertices: Vec<VertexId>,
}

fn held_karp(graph: &Arc<SimpleGraph>) -> Option<Solution> {
    let mut held_karp = HeldKarp::new();
    let best_tour = (0..graph.ports.len())
        .map(|v| held_karp.traveling_salesman(&graph, v as VertexId))
        .min_by_key(|tour| tour.cost).unwrap();
    best_tour.to_solution(&graph)
}

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
struct Mask(u32);

impl Mask {
    pub fn from_set(set: &Vec<VertexId>) -> Mask {
        let mut mask = 0u32;
        for &v in set {
            mask |= 1 << (v as u32);
        }
        Mask(mask)
    }
}

struct HeldKarp {
    /// g(S, e) min cost of going through all nodes in 'S', ending in 'e'.
    /// g[mask][e]
    g: Vec<Vec<Cost>>,
    /// p(S, e) predecessor to 'e' of going through nodes in 'S', ending in 'e'.
    /// Used when backtracking.
    /// p[mask][e]
    p: Vec<Vec<VertexId>>,
}

impl HeldKarp {
    pub fn new() -> Self {
        HeldKarp { g: Vec::new(), p: Vec::new() }
    }

    pub fn traveling_salesman(
        &mut self, graph: &SimpleGraph, start: VertexId) -> Tour {
        let start_tick = graph.start_tick + 1;  // time to dock spawning port
        self.g = vec![vec![Cost::MAX; graph.ports.len()]; NUM_MASKS];
        self.p = vec![vec![VertexId::MAX; graph.ports.len()]; NUM_MASKS];

        let others: Vec<VertexId> = (0..graph.ports.len())
            .map(|v| v as VertexId).filter(|&v| v != start).collect();

        // For |S|=1 (S={k}), smallest cost is the cost of start->k.
        for k in &others {
            let cost = graph.cost(graph.tick_offset(start_tick), start, *k) as Cost;
            let mask = Mask::from_set(&vec![*k]);
            // +1 to dock
            self.g[mask.0 as usize][*k as usize] = start_tick + cost + 1;
            self.p[mask.0 as usize][*k as usize] = *k;
        }

        // For |S|=s, smallest cost depends on |S'|=s-1 values of g(S', k).
        let max_set_items = graph.ports.len() - 1;
        for s in 2..=max_set_items {
            for set in others.iter().cloned().combinations(s) {
                let mask = Mask::from_set(&set);
                for k in &set {
                    let set_minus_k = set.iter().cloned().filter(|&v| v != *k).collect();
                    let mask_minus_k = Mask::from_set(&set_minus_k);
                    let (min_cost, min_vertex) = set_minus_k.iter().cloned().map(|m| {
                        let current_cost = self.g[mask_minus_k.0 as usize][m as usize];
                        let m_k_cost = graph.cost(
                            graph.tick_offset(current_cost), m, *k) as Cost;
                        let cost = current_cost + m_k_cost + 1;  // +1 to dock
                        (cost, m)
                    }).min_by_key(|&(cost, _)| cost).unwrap();
                    self.g[mask.0 as usize][*k as usize] = min_cost;
                    self.p[mask.0 as usize][*k as usize] = min_vertex;
                }
            }
        }

        // Find the best tour by checking paths back to the start.
        let mask_all = Mask::from_set(&others);
        let (total_cost, last_city) = others.iter().cloned().map(|k| {
            let current_cost = self.g[mask_all.0 as usize][k as usize];
            let k_start_cost = graph.cost(
                graph.tick_offset(current_cost), k, start) as Cost;
            // Note: no +1 for docking, the last dock tick doesn't count.
            let cost = current_cost + k_start_cost;
            (cost, k)
        }).min_by_key(|&(cost, _)| cost).unwrap();

        let vertices = self.backtrack(start, last_city, others);
        Tour { cost: total_cost, vertices }
    }

    fn backtrack(&self, start: VertexId, last: VertexId,
                 set: Vec<VertexId>) -> Vec<VertexId> {
        let mut set = set;
        let mut vertices = Vec::with_capacity(set.len() + 2);
        vertices.push(start);
        let mut vertex = last;
        while !set.is_empty() {
            vertices.push(vertex);
            let mask = Mask::from_set(&set);
            let next = self.p[mask.0 as usize][vertex as usize];
            set.retain(|&v| v != vertex);
            vertex = next;
        }
        vertices.push(start);
        vertices.reverse();
        vertices
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
