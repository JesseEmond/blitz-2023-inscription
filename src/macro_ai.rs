// Executes a given solution by mutating Micro states at a macro level.
use log::{error, info};

use crate::challenge::{Solution};
use crate::game_interface::{GameTick};
use crate::graph::{Graph};
use crate::micro_ai::{Micro, State};

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

    pub fn init(&mut self, graph: &Graph, solution: Solution) {
        if graph.ports.len() < 20 {
            error!("Only {} ports!? Why would I bother!", graph.ports.len());
            self.give_up = true;
            return;
        }

        summarize_solution(&solution, &graph);

        info!("[MACRO] Our plan is the following: ");
        info!("[MACRO]   spawn on {spawn:?}", spawn = solution.spawn);
        for path in &solution.paths {
            info!("[MACRO]   go to {goal:?} in {cost:?} steps (+1 dock)",
                  goal = path.goal, cost = path.cost);
        }

        self.solution_idx = 0;
        self.solution = Some(solution);
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
