use log::{error, info, warn};
use std::sync::Arc;
use std::time::{Instant};

use crate::challenge::{Solution};
use crate::game_interface::{GameTick};
use crate::graph::{Graph};
use crate::micro_ai::{Micro, State};
use crate::solvers::{ExactTspSolver, NearestNeighborSolver, Solver};

// If using ACO:
use crate::ant_colony_optimization::{Colony, HyperParams};
use std::fs;
use serde_json::Value;

pub struct Macro {
    solution: Option<Solution>,
    solution_idx: usize,
    pub give_up: bool,
}

impl Default for Macro {
    fn default() -> Self {
        Self::new()
    }
}

impl Macro {
    pub fn new() -> Self {
        Macro {
            solution: None,
            solution_idx: 0,
            give_up: false,
        }
    }

    pub fn init(&mut self, game_tick: Arc<GameTick>) {
        let macro_start = Instant::now();

        // This is a bit verbose, but we always want this on server.
        info!("--- TICK DUMP BEGIN ---");
        info!("{game_tick:?}");
        info!("--- TICK DUMP END ---");

        if game_tick.map.ports.len() < 20 {
            error!("Only {} ports!? Why would I bother!", game_tick.map.ports.len());
            self.give_up = true;
            return;
        }

        let graph_start = Instant::now();
        let graph: Arc<Graph> = Arc::new(Graph::new(&game_tick));
        info!("Graph was built in {:?}", graph_start.elapsed());

        let greedy_sln = NearestNeighborSolver::new().solve(&graph)
            .expect("Greedy nearest neighbor did not find a solution.");
        info!("Greedy bot summary");
        summarize_solution(&greedy_sln, &graph);

        // To use an Ant Colony Optimization, use the following:
        let hyperparams = if let Ok(hyperparam_data) = fs::read_to_string("hyperparams.json") {
            info!("[MACRO] Loading hyperparams from hyperparams.json.");
            let parsed: Value = serde_json::from_str(&hyperparam_data).expect("invalid json");
            serde_json::from_value(parsed).expect("invalid hyperparams")
        } else {
            info!("[MACRO] Using default params.");
            // Point(hyperparams=Hyperparams(iterations=464, ants=63, evaporation_rate=0.7800131108465345, exploitation_probability=0.3642226425600267, heuristic_power=2.5583485993720037, base_pheromones=2.0097671658359686, local_evaporation_rate=0.7523178610770483), score=3078.174695652174)
            HyperParams {
                iterations: 464,
                ants: 63,
                evaporation_rate: 0.7800131108465345,
                exploitation_probability: 0.3642226425600267,
                heuristic_power: 2.5583485993720037,
                base_pheromones: 2.0097671658359686,
                local_evaporation_rate: 0.7523178610770483
            }
        };
        info!("[MACRO] Hyperparams: {hyperparams:?}");
        let mut colony = Colony::new(&graph, hyperparams, /*seed=*/42);
        let colony_start = Instant::now();
        let colony_sln = colony.run();
        info!("Colony solution was found in {:?}", colony_start.elapsed());
        info!("[MACRO] Solution found has a score of {}, with {} ports",
              colony_sln.score, colony_sln.paths.len());
        info!("Colony solution summary:");
        summarize_solution(&colony_sln, &graph);

        let tsp_sln = ExactTspSolver{}.solve(&graph)
            .expect("No exact TSP possible on this map");
        info!("Here is the TSP solution:");
        summarize_solution(&tsp_sln, &graph);

        self.solution_idx = 0;

        // If using an Ant Colony Optimization solver:
        self.solution = Some(colony_sln);
        // if greedy_sln.score > self.solution.as_ref().unwrap().score {
        //     warn!("A greedy solution is better {} > {}, using it.",
        //           greedy_sln.score, self.solution.as_ref().unwrap().score);
        //     self.solution = Some(greedy_sln);
        // }
        // self.solution = Some(greedy_sln);
        if tsp_sln.score > self.solution.as_ref().unwrap().score {
            info!("A TSP solution is better (duh!) {} > {}, using it.",
                  tsp_sln.score, self.solution.as_ref().unwrap().score);
            self.solution = Some(tsp_sln);
        } else if tsp_sln.score < self.solution.as_ref().unwrap().score {
            assert!(tsp_sln.paths.len() > self.solution.as_ref().unwrap().paths.len(),
                    "Ran an exact TSP gives a worse solution for a full tour. That's a bug.");
            warn!("TSP solution is worse, because a tour with <20 cities is better (or sub-optimal TSP settings).");
        }

        info!("[MACRO] Our plan is the following: ");
        info!("[MACRO]   spawn on {spawn:?}", spawn = self.solution.as_ref().unwrap().spawn);
        for path in &self.solution.as_ref().unwrap().paths {
            info!("[MACRO]   go to {goal:?} in {cost:?} steps (+1 dock)",
                  goal = path.goal, cost = path.cost);
        }
        info!("Macro took {:?}", macro_start.elapsed());
    }

    pub fn assign_state(&mut self, micro: &mut Micro, game_tick: &GameTick) {
        if self.give_up {
            return
        }
        if game_tick.spawn_location.is_none() {
            let spawn = self.solution.as_ref().unwrap().spawn;
            info!("[MACRO] Will spawn on {spawn:?}");
            micro.state = State::Spawning { position: spawn };
        } else if game_tick.is_over {
            info!("[MACRO] Game over! Waiting.");
            micro.state = State::Waiting;
        } else if let State::Waiting = micro.state {
            let next_port_path = self.solution.as_ref().unwrap().paths[self.solution_idx].clone();
            info!("[MACRO] Will go to this port next: {port:?}, in {steps} steps.",
                  port = next_port_path.goal, steps = next_port_path.cost);
            info!("[MACRO] Path: {path:?}", path = next_port_path.steps);
            micro.state = State::Following {
                path: next_port_path,
                path_index: 0,
            };
        } else if let State::Docking { port } = micro.state {
            let solution = self.solution.as_ref().unwrap();
            assert!(self.solution_idx < solution.paths.len());
            let was_spawn = port == solution.spawn;
            if !was_spawn {  // if this was the spawn, we want to stay on step 0
                self.solution_idx += 1;
            }
            let ports_left = solution.paths.len() - self.solution_idx;
            info!("[MACRO] Docked port: {port:?}, {ports_left} ports left in solution.");
        }
        // else, no-op, micro is no a task.
    }
}

fn summarize_solution(solution: &Solution, graph: &Graph) {
    let mut tick = graph.start_tick + 1;  // Time to dock spawn
    info!("Will spawn on {:?}, start moving on tick {}", solution.spawn, tick);
    for path in &solution.paths {
        tick += path.cost + 1;
        info!("Will go to {:?} in {} steps, then dock. Next starts at tick {}",
              path.goal, path.cost, tick);
    }
    info!("Final dock on tick {}, score of {}", tick - 1, solution.score);
}
