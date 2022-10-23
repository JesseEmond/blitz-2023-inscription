use log::{info, warn};
use serde_json::{Value};
use std::fs;
use std::time::{Instant};

use crate::ant_colony_optimization::{Colony, HyperParams, Solution};
use crate::game_interface::{GameTick, eval_score};
use crate::graph::{Graph, VertexId};
use crate::micro_ai::{Micro, State};
use crate::pathfinding::{Pathfinder};

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
        let macro_start = Instant::now();
        let schedule: Vec<u8> = game_tick.tide_schedule.iter().map(|&e| e as u8).collect();
        self.pathfinder.grid.init(&game_tick.map, &schedule, game_tick.current_tick);

        // This is a bit verbose, but we always want this on server.
        info!("--- TICK DUMP BEGIN ---");
        info!("{game_tick:?}");
        info!("--- TICK DUMP END ---");

        let graph_start = Instant::now();
        let graph = Graph::new(&mut self.pathfinder, game_tick);
        info!("Graph was built in {:?}", graph_start.elapsed());

        let greedy_start = Instant::now();
        let greedy_sln = greedy_bot(&graph);
        info!("A greedy bot would get us a score of {score}", score = greedy_sln.score);
        info!("Greedy solution found in {:?}", greedy_start.elapsed());

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
        let mut colony = Colony::new(graph, hyperparams, /*seed=*/42);
        let colony_start = Instant::now();
        self.solution = Some(colony.run());
        info!("Colony solution was found in {:?}", colony_start.elapsed());
        self.solution_idx = 0;
        info!("[MACRO] Solution found has a score of {score}",
              score = self.solution.as_ref().unwrap().score);
        if greedy_sln.score > self.solution.as_ref().unwrap().score {
            warn!("A greedy solution is better... {} > {}, using it.",
                  greedy_sln.score, self.solution.as_ref().unwrap().score);
            self.solution = Some(greedy_sln);
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

pub fn greedy_bot(graph: &Graph) -> Solution {
    let mut best_sln: Option<Solution> = None;
    for start_vertex_id in 0..graph.vertices.len() {
        let start_vertex_id = start_vertex_id as VertexId;
        let mut seen = 1u64 << start_vertex_id;
        let mut tick = graph.start_tick + 1;  // time to dock spawn
        let mut current_id = start_vertex_id;
        let mut paths = Vec::new();

        loop {
            let vertex = graph.vertex(current_id);
            let options = vertex.edges.iter().filter(|&edge_id| {
                let edge = graph.edge(*edge_id);
                let mask = 1u64 << edge.to;
                let unseen = seen & mask == 0;
                let have_time = tick + edge.path(tick).cost + 1 < graph.max_ticks;
                unseen && have_time
            });
            let closest = match options.min_by_key(|&edge_id| {
                graph.edge(*edge_id).path(tick).cost }) {
                Some(&edge_id) => edge_id,
                None => break,
            };
            let edge = graph.edge(closest);
            current_id = edge.to;
            seen |= 1u64 << current_id;
            paths.push(edge.path(tick).path);
            tick += edge.path(tick).cost + 1;
            let visits = seen.count_ones();
            // First compute score if we stay here until the end of the game
            let score = eval_score(visits, graph.max_ticks, /*looped=*/false);
            let best_score = match &best_sln {
                Some(sln) => sln.score,
                None => score - 1,
            };
            if score > best_score {
                best_sln = Some(Solution {
                    score,
                    spawn: graph.vertex(start_vertex_id).position,
                    paths: paths.iter().map(|&path_id| graph.paths[path_id as usize].clone()).collect(),
                });
            }

            // Consider going home from there, if we have time.
            let home_edge_id = graph.vertex_edge_to(
                current_id, start_vertex_id).expect("No way home?");
            let home_edge = graph.edge(home_edge_id);
            if tick + home_edge.path(tick).cost < graph.max_ticks {
                let home_score = eval_score(visits + 1,
                                            tick + home_edge.path(tick).cost,
                                            /*looped=*/true);
                if home_score > best_score {
                    let mut paths = paths.clone();
                    paths.push(home_edge.path(tick).path);
                    best_sln = Some(Solution {
                        score: home_score,
                        spawn: graph.vertex(start_vertex_id).position,
                        paths: paths.iter().map(|&path_id| graph.paths[path_id as usize].clone()).collect(),
                    });
                }
            }
        }
    }
    let best_sln = best_sln.unwrap();
    info!("Greedy bot summary");
    let mut tick = graph.start_tick;
    info!("Will spawn on {:?}, start moving on tick {}", best_sln.spawn, tick);
    for path in &best_sln.paths {
        tick += path.cost + 1;
        info!("Will go to {:?} in {} steps, then dock. Next starts at tick {}",
              path.goal, path.cost, tick);
    }
    info!("Final dock on tick {}, score of {}", tick - 1, best_sln.score);
    best_sln
}
