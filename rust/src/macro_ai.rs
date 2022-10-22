use log::{debug, info};
use serde_json::{Value};
use std::fs;

use crate::ant_colony_optimization::{Colony, HyperParams, Solution};
use crate::game_interface::{GameTick};
use crate::graph::{Graph};
use crate::micro_ai::{Micro, State};
use crate::pathfinding::{Pathfinder, Pos};

pub struct Macro {
    pathfinder: Pathfinder,
    solution: Option<Solution>,
    solution_idx: usize,
}

impl Default for Macro {
    fn default() -> Self {
        Self::new()
    }
}

impl Macro {
    pub fn new() -> Self {
        Macro {
            pathfinder: Pathfinder::new(),
            solution: None,
            solution_idx: 0,
        }
    }

    pub fn init(&mut self, game_tick: &GameTick) {
        let schedule: Vec<u8> = game_tick.tide_schedule.iter().map(|&e| e as u8).collect();
        self.pathfinder.grid.init(&game_tick.map, &schedule, game_tick.current_tick);

        // This is a bit verbose, but we always want this on server.
        info!("--- TICK DUMP BEGIN ---");
        info!("{game_tick:?}");
        info!("--- TICK DUMP END ---");

        let graph = Graph::new(&mut self.pathfinder, game_tick);
        let hyperparams = if let Ok(hyperparam_data) = fs::read_to_string("hyperparams.json") {
            info!("[MACRO] Loading hyperparams from hyperparams.json.");
            let parsed: Value = serde_json::from_str(&hyperparam_data).expect("invalid json");
            serde_json::from_value(parsed).expect("invalid hyperparams")
        } else {
            info!("[MACRO] Using default params.");
            // After RANDOM 13 rounds 2 suggestions, 5 suggestions/round
            // Measurement(metrics={'maximize_score': Metric(value=3059.909090909091, std=None)}, elapsed_secs=0.0, steps=0) params: ParameterDict(_items={'iterations': 342.0, 'ants': 461.0, 'evaporation_rate': 0.591452311949127, 'exploitation_probability': 0.07706219963475536, 'pheromone_trail_power': 4.728800687880105, 'heuristic_power': 3.3351427312956146, 'base_pheromones': 0.7635218746185567, 'local_evaporation_rate': 0.4955295476638974})
            HyperParams {
                iterations: 342,
                ants: 461,
                evaporation_rate: 0.6,
                exploitation_probability: 0.077,
                heuristic_power: 3.34,
                base_pheromones: 0.7635,
                local_evaporation_rate: 0.50,
            }
        };
        info!("[MACRO] Hyperparams: {hyperparams:?}");
        let mut colony = Colony::new(graph, hyperparams, /*seed=*/42);
        self.solution = Some(colony.run());
        self.solution_idx = 0;
        info!("[MACRO] Solution found has a score of {score}",
              score = self.solution.as_ref().unwrap().score);
        info!("[MACRO] Our plan is the following: ");
        info!("[MACRO]   spawn on {spawn:?}", spawn = self.solution.as_ref().unwrap().spawn);
        for path in &self.solution.as_ref().unwrap().paths {
            info!("[MACRO]   go to {goal:?} in {cost:?} steps (+1 dock)",
                  goal = path.goal, cost = path.cost);
        }
    }

    pub fn assign_state(&mut self, micro: &mut Micro, game_tick: &GameTick) {
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
            debug!("[MACRO] Path: {path:?}", path = next_port_path.steps);
            micro.state = State::Following {
                path: next_port_path,
                path_index: 0,
            };
        } else if let State::Docking = micro.state {
            let current = Pos::from_position(&game_tick.current_location.unwrap());
            let solution = self.solution.as_ref().unwrap();
            assert!(self.solution_idx < solution.paths.len());
            let was_spawn = current == solution.spawn;
            if !was_spawn {  // if this was the spawn, we want to stay on step 0
                self.solution_idx += 1;
            }
            let ports_left = solution.paths.len() - self.solution_idx;
            info!("[MACRO] Docked port: {current:?}, {ports_left} ports left in solution.");
        }
        // else, no-op, micro is no a task.
    }
}
