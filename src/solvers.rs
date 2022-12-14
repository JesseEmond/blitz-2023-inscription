// Different solver implementations to plan a macro sequence of paths to follow.

// On a dev set with 64 20-ports games with a >= 3800 optimal score:
// (note: for max_starts: 5, runs <1s locally, but not always on server)
// Solver                                  | Average |  Max  | Runs |
//                                         |  Score  | Score | <1s? |
// ------------------------------------------------------------------
// OptimalSolver                           |  3835.9 |  3896 |   N  |
// ExactTspSolver                          |  3835.9 |  3896 |   N  |
// ExactTspSomeStartsSolver{max_starts: 5} |  3827.6 |  3896 |   N  |
// ExactTspSomeStartsSolver{max_starts: 4} |  3826.1 |  3896 |   Y  |
// AntColonyOptimizationSolver             |  3824.2 |  3896 |   Y  |
// NearestNeighborSolver                   |  3747.2|  3836 |   Y  |

use log::{info, warn};
use std::sync::Arc;
use std::time::{Instant};

use crate::ant_colony_optimization::{Colony, HyperParams};
use crate::challenge::{Solution, eval_score};
use crate::graph::{Graph, VertexId};
use crate::held_karp::{held_karp};

pub fn verify_solution(graph: &Graph, solution: &Solution) {
    let spawn = graph.vertex_id(&solution.spawn);
    let mut tick = graph.start_tick + 1;  // +1 to dock start
    let mut current = spawn;
    for path in &solution.paths {
        let next = graph.vertex_id(&path.goal);
        let real_path = graph.path(graph.tick_offset(tick), current, next);
        assert!(*path == *real_path,
                "At tick {}, expected path {:?} to {:?}, got {:?}",
                tick, real_path, real_path.goal, path);
        let dock_tick = if next == spawn { 0 } else { 1 };
        tick += (path.cost as u16) + dock_tick;
        current = next;
    }
    let looped = current == spawn;
    let visits = solution.paths.len() + 1;
    let score = eval_score(visits as u32, tick, looped);
    assert!(solution.score == score,
            "{} visits, {} ticks, expected score of {}, got {}",
            visits, tick, score, solution.score);
}

pub trait Solver {
    // Name to display for this solver.
    fn name(&self) -> &str;

    // Implementation of the solver.
    fn do_solve(&mut self, graph: &Arc<Graph>) -> Option<Solution>;

    // Wrapper to do_solve, to log timing and score information.
    fn solve(&mut self, graph: &Arc<Graph>) -> Option<Solution> {
        let start = Instant::now();
        let solution = self.do_solve(graph);
        info!("Solver {} took {:?}", self.name(), start.elapsed());
        match &solution {
            Some(solution) => info!(
                "Solver {} would get us a score of {}, with {} ports",
                self.name(), solution.score, solution.paths.len()),
            None => warn!("Solver {} did NOT find a solution.", self.name()),
        };
        solution
    }
}

// Greedy algorithm that tries each possible starting point and either goes to
// the nearest possible unvisited port, or goes home, keeping track of the best
// observed score so far.
// This is similar to a Nearest Neighbor solver for the TSP, but with the
// challenge logic baked in to consider shorter tours.
pub struct NearestNeighborSolver {
    best_solution: Option<Solution>,
}

// Solver that solves the exact Traveling Salesman Problem on the graph using
// Held-Karp for each possible starting point, going to *all* ports.
// For games where the optimal score has all ports, this will find the optimal
// possible score.
// With 4 physical cores, this can fit under ~1s based on local tests, but this
// is not the case on the server.
pub struct ExactTspSolver;

// Solver that solves the Traveling Salesman Problem on the graph using
// Held-Karp for a max number of possible starting point, going to all ports.
// On the server, this can run ~4 start options. This will sometimes miss
// optimal starting points, see ExactTspSolver for that.
pub struct ExactTspSomeStartsSolver {
    // Max number of starts to try.
    pub max_starts: usize
}

// Solve using Ant Colony Optimization to find a solution.
// It runs iterations of sampling and simulating "ants" that leave pheromones on
// good paths.
pub struct AntColonyOptimizationSolver {
    pub hyperparams: HyperParams,
}

// Solver that solves the TSP on the graph using Held-Karp for each possible
// starting point, then considers every possible port subset S, looping to
// start.
// Note that this solver is slow and should only be used for upper bound offline
// evals.
pub struct OptimalSolver;

impl Solver for NearestNeighborSolver {
    fn name(&self) -> &str {
        "nearest-neighbor"
    }

    fn do_solve(&mut self, graph: &Arc<Graph>) -> Option<Solution> {
        for start in 0..graph.ports.len() {
            self.solve_from_start(graph, start as VertexId);
        }
        self.best_solution.clone()
    }
}

impl NearestNeighborSolver {
    pub fn new() -> Self {
        NearestNeighborSolver { best_solution: None }
    }

    fn solve_from_start(&mut self, graph: &Arc<Graph>, start: VertexId) {
        let mut seen = 1u64 << start;
        let mut tick = graph.start_tick + 1;  // time to dock spawn
        let mut current = start;
        let mut paths = Vec::new();

        loop {
            let tick_offset = graph.tick_offset(tick);
            let options = graph.others(current).filter(|&other| {
                let mask = 1u64 << other;
                let unseen = seen & mask == 0;
                let cost = graph.cost(tick_offset, current, other) as u16;
                let have_time = tick + cost + 1 < graph.max_ticks;
                unseen && have_time
            });
            let closest = match options.min_by_key(
                |&option| graph.cost(tick_offset, current, option)) {
                Some(option_id) => option_id,
                None => break,  // No more options! We are done.
            };
            paths.push((tick_offset, current, closest));
            // +1 for time to dock port
            tick += (graph.cost(tick_offset, current, closest) + 1) as u16;
            current = closest;
            seen |= 1u64 << current;
            let visits = seen.count_ones();
            // First compute score if we stay here until the end of the game
            let score = eval_score(visits, graph.max_ticks, /*looped=*/false);
            let best_score = match &self.best_solution {
                Some(solution) => solution.score,
                None => i32::MIN,
            };
            if score > best_score {
                self.best_solution = Some(Solution {
                    score,
                    spawn: graph.ports[start as usize],
                    paths: paths.iter()
                        .map(|&(tick_offset, from, to)| graph.path(tick_offset,
                                                                   from,
                                                                   to).clone())
                        .collect(),
                });
            }

            // Consider going home from there, if we have time.
            let tick_offset = graph.tick_offset(tick);
            let home_cost = graph.cost(tick_offset, current, start) as u16;
            if tick + home_cost < graph.max_ticks {
                let home_score = eval_score(visits + 1,
                                            tick + home_cost,
                                            /*looped=*/true);
                if home_score > best_score {
                    let mut paths = paths.clone();
                    paths.push((tick_offset, current, start));
                    self.best_solution = Some(Solution {
                        score: home_score,
                        spawn: graph.ports[start as usize],
                        paths: paths.iter().map(
                            |&(tick_offset, from, to)| graph.path(tick_offset,
                                                                  from,
                                                                  to).clone())
                            .collect(),
                    });
                }
            }
        }
    }
}

impl Solver for ExactTspSolver {
    fn name(&self) -> &str {
        "exact-tsp"
    }

    fn do_solve(&mut self, graph: &Arc<Graph>) -> Option<Solution> {
        let max_starts = graph.ports.len();  // try all starts
        held_karp(&graph, max_starts, /*check_shorter_tours=*/false)
    }
}

impl Solver for ExactTspSomeStartsSolver {
    fn name(&self) -> &str {
        "exact-tsp-some-starts"
    }

    fn do_solve(&mut self, graph: &Arc<Graph>) -> Option<Solution> {
        held_karp(&graph, self.max_starts, /*check_shorter_tours=*/false)
    }
}

impl AntColonyOptimizationSolver {
    pub fn new(hyperparams: HyperParams) -> Self {
        AntColonyOptimizationSolver { hyperparams }
    }
}

impl Default for AntColonyOptimizationSolver {
    fn default() -> Self {
        // From sweep (15 rounds of 5 iters, only on game #10589):
        // Point(
        //   hyperparams=Hyperparams(iterations=469,
        //                           ants=72,
        //                           evaporation_rate=0.7165626675063436,
        //                           exploitation_probability=0.2681295531581947,
        //                           heuristic_power=3.5305661701563418,
        //                           local_evaporation_rate=0.7544669066991354,
        //                           min_pheromones=0.18408083030232747,
        //                           max_pheromones=5.243022596076836,
        //                           pheromones_init_ratio=0.7213026402824096,
        //                           seed=42),
        //   score=3839.601370607376)
        let hyperparams = HyperParams {
            iterations: 469,
            ants: 72,
            evaporation_rate: 0.7165626675063436,
            exploitation_probability: 0.2681295531581947,
            heuristic_power: 3.5305661701563418,
            local_evaporation_rate: 0.7544669066991354,
            min_pheromones: 0.18408083030232747,
            max_pheromones: 5.243022596076836,
            pheromones_init_ratio: 0.7213026402824096,
            seed: 42,
        };
        AntColonyOptimizationSolver::new(hyperparams)
    }
}

impl Solver for AntColonyOptimizationSolver {
    fn name(&self) -> &str {
        "ant-colony-optimization"
    }

    fn do_solve(&mut self, graph: &Arc<Graph>) -> Option<Solution> {
        let mut colony = Colony::new(graph, self.hyperparams.clone());
        Some(colony.run())
    }
}

impl Solver for OptimalSolver {
    fn name(&self) -> &str {
        "optimal"
    }

    fn do_solve(&mut self, graph: &Arc<Graph>) -> Option<Solution> {
        let max_starts = graph.ports.len();  // try all starts
        held_karp(&graph, max_starts, /*check_shorter_tours=*/true)
    }
}
