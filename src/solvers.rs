// Different solver implementations to plan a macro sequence of paths to follow.

// On a dev set with 123 games:
// Solver                                  | Average |  Max | <1s? |
// -----------------------------------------------------------------
// ExactTspSolver                          |  3479.1 | 3722 |   N  |
// ExactTspSomeStartsSolver{max_starts: 5} |  3471.8 | 3722 |   N  |
// AntColonyOptimizationSolver             |  3470.2 | 3722 |   Y  |
// ExactTspSomeStartsSolver{max_starts: 4} |  3469.8 | 3722 |   Y  |
// NearestNeighborSolver                   |  3360.9 | 3704 |   Y  |

use log::{info, warn};
use std::sync::Arc;
use std::time::{Instant};

use crate::ant_colony_optimization::{Colony, HyperParams};
use crate::challenge::{Solution, eval_score};
use crate::graph::{Graph, VertexId};
use crate::held_karp::{held_karp};

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
// Held-Karp for each possible starting point, going to all ports.
// For games where the optimal score has all ports, this will find the optimal
// possible score.
// With 4 physical cores, this can fit under ~1s based on local tests, but this
// is not the case on the server.
// TODO: for a fully optimal solver, we could consider going home after each set
// size |S| and pick the best observed score, but this is likely not worth it
// for top leaderboard potential games.
pub struct ExactTspSolver {
}

// Solver that solves the Traveling Salesman Problem on the graph using
// Held-Karp for a max number of possible starting point, going to all ports.
// On the server, this can run ~4 start options. This will sometimes miss
// optimal starting points, see ExactTspSolver for that.
pub struct ExactTspSomeStartsSolver {
    // Max number of starts to try.
    max_starts: usize
}

// Solve using Ant Colony Optimization to find a solution.
// It runs iterations of sampling and simulating "ants" that leave pheromones on
// good paths.
pub struct AntColonyOptimizationSolver {
    seed: u64,
    hyperparams: HyperParams,
}

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
        held_karp(&graph, max_starts)
    }
}

impl Solver for ExactTspSomeStartsSolver {
    fn name(&self) -> &str {
        "exact-tsp-some-starts"
    }

    fn do_solve(&mut self, graph: &Arc<Graph>) -> Option<Solution> {
        held_karp(&graph, self.max_starts)
    }
}

impl AntColonyOptimizationSolver {
    pub fn new(hyperparams: HyperParams, seed: u64) -> Self {
        AntColonyOptimizationSolver { hyperparams, seed }
    }
}

impl Solver for AntColonyOptimizationSolver {
    fn name(&self) -> &str {
        "ant-colony-optimization"
    }

    fn do_solve(&mut self, graph: &Arc<Graph>) -> Option<Solution> {
        let mut colony = Colony::new(graph, self.hyperparams.clone(), self.seed);
        Some(colony.run())
    }
}
