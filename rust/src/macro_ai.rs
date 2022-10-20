use log::{info};
use std::collections::HashSet;

use crate::ant_colony_optimization::{Colony, HyperParams};
use crate::game_interface::{GameTick, eval_score};
use crate::graph::{Graph};
use crate::micro_ai::{Micro, State};
use crate::pathfinding::{Path, Pathfinder, Pos};

#[derive(Debug)]
pub struct Simulation {
    current: Pos,
    home: Pos,
    ports_visited: u32,
    missing_ports: HashSet<Pos>,
    tick: u32,
    max_tick: u32,
}

pub struct Macro {
    pathfinder: Pathfinder,
    missing_ports: HashSet<Pos>,
}

impl Macro {
    pub fn new() -> Self {
        Macro {
            pathfinder: Pathfinder::new(),
            missing_ports: HashSet::new(),
        }
    }

    pub fn init(&mut self, game_tick: &GameTick) {
        let schedule: Vec<u8> = game_tick.tide_schedule.iter().map(|&e| e as u8).collect();
        self.pathfinder.grid.init(&game_tick.map, &schedule,
                                  game_tick.current_tick.into());
        self.missing_ports = HashSet::from_iter(
            game_tick.map.ports.iter().map(Pos::from_position));

        info!("--- TICK DUMP BEGIN ---");
        info!("{game_tick:?}");
        info!("--- TICK DUMP END ---");

        let graph = Graph::new(&mut self.pathfinder, game_tick);
        let mut hyperparams = HyperParams::default_params(/*iterations=*/1000);
        hyperparams.ants = 100;
        hyperparams.pheromone_trail_power = 0.0;
        hyperparams.heuristic_power = 5.0;
        let mut colony = Colony::new(graph, hyperparams, /*seed=*/42);
        let solution = colony.run();
        info!("Solution found has a score of {score}", score = solution.score);
    }

    pub fn assign_state(&mut self, micro: &mut Micro, game_tick: &GameTick) {
        if game_tick.spawn_location.is_none() {
            let spawn = self.pick_spawn(game_tick);
            info!("[MACRO] Will spawn on {spawn:?}");
            micro.state = State::Spawning { position: spawn };
        } else if game_tick.is_over {
            info!("[MACRO] Game over! Waiting.");
            micro.state = State::Waiting;
        } else if let State::Waiting = micro.state {
            if let Some(next_port_path) = self.pick_next_port(game_tick) {
                info!("[MACRO] Will go to this port next: {port:?}, in {steps} steps.",
                      port = next_port_path.goal, steps = next_port_path.cost);
                info!("[MACRO] Path: {path:?}", path = next_port_path.steps);
                micro.state = State::Following {
                    path: next_port_path,
                    path_index: 0,
                };
            } else {
                let home_path = self.path_home(game_tick)
                    .expect("No path home at the end...?");
                info!("[MACRO] No more ports to dock, going to home port: {port:?}, in {steps} steps",
                      port = home_path.goal, steps = home_path.cost);
                micro.state = State::Following {
                    path: home_path,
                    path_index: 0,
                };
            }
        } else if let State::Docking = micro.state {
            let port = Pos::from_position(&game_tick.current_location.unwrap());
            self.missing_ports.remove(&port);
            info!("[MACRO] Docked port: {port:?}, {left} ports left!",
                  left = self.missing_ports.len());
        }
        // else, no-op, micro is no a task.
    }

    pub fn pick_next_port(&mut self, game_tick: &GameTick) -> Option<Path> {
        // TODO: more advanced TSP than greedy nearest neighbor with one spawn port
        if self.missing_ports.is_empty() { return None; }
        // TODO: consider overall gains from going vs. costs
        let tick = game_tick.current_tick.into();
        let current = Pos::from_position(&game_tick.current_location.unwrap());
        let closest = match self.pathfinder.path_to_closest(
            &current, &self.missing_ports, tick) {
            Some(path) => path,
            None => {
                return None;
            },
        };

        if self.worth_visiting(&closest, game_tick) { Some(closest) } else { None }
    }

    pub fn pick_spawn(&mut self, game_tick: &GameTick) -> Pos {
        game_tick.map.ports.iter().map(Pos::from_position).max_by_key(
            |spawn| self.spawn_score(game_tick, spawn)).expect("No ports...?")
    }

    pub fn spawn_score(&mut self, game_tick: &GameTick, spawn: &Pos) -> i32 {
        let mut sim = Simulation::from_spawn(game_tick, spawn);
        let score = sim.expected_final_score(&mut self.pathfinder);
        info!("[MACRO] Spawn {spawn:?} score: {score}");
        score
    }

    pub fn worth_visiting(
        &mut self, target: &Path, game_tick: &GameTick
        ) -> bool {
        let mut sim = Simulation::new(game_tick, &self);
        if sim.ports_visited == 1 {
            // At least want 2 ports!
            return true;
        }
        let go_home_score = sim.score_if_go_home(&mut self.pathfinder);
        sim.visit(target);
        let final_score = sim.expected_final_score(&mut self.pathfinder);
        let worth = final_score > go_home_score;
        info!(concat!("[MACRO] {goal:?} is {worth_it} visiting. Expected ",
                      "final score: {final_score} vs. going home ",
                      "{go_home_score}"),
              goal = target.goal, final_score = final_score,
              go_home_score = go_home_score,
              worth_it = if worth { "worth" } else { "not worth" });
        worth
    }

    pub fn path_home(&mut self, game_tick: &GameTick) -> Option<Path> {
        let current = Pos::from_position(&game_tick.current_location.unwrap());
        let home = Pos::from_position(&game_tick.spawn_location.unwrap());
        self.pathfinder.shortest_path(&current, &home,
                                      game_tick.current_tick.into())
    }
}

impl Simulation {
    pub fn new(game_tick: &GameTick, ai_macro: &Macro) -> Self {
        Simulation {
            current: Pos::from_position(&game_tick.current_location.unwrap()),
            home: Pos::from_position(&game_tick.spawn_location.unwrap()),
            tick: game_tick.current_tick.into(),
            max_tick: game_tick.total_ticks.into(),
            ports_visited: game_tick.visited_port_indices.len() as u32,
            missing_ports: ai_macro.missing_ports.clone(),
        }
    }

    pub fn from_spawn(game_tick: &GameTick, spawn: &Pos) -> Self {
        let mut missing_ports = HashSet::from_iter(
            game_tick.map.ports.iter().map(Pos::from_position));
        missing_ports.remove(spawn);
        Simulation {
            current: *spawn,
            home: *spawn,
            tick: (game_tick.current_tick + 1) as u32,  // +1 to dock
            max_tick: game_tick.total_ticks.into(),
            ports_visited: 1,  // only home
            missing_ports: missing_ports,
        }
    }

    pub fn done(&self) -> bool {
        let tick_end = self.tick >= self.max_tick;
        let back_home = self.current == self.home && self.ports_visited > 1;
        tick_end || back_home
    }

    pub fn current_score(&self) -> i32 {
        // If our last visit ended after the end, it didn't count.
        let last_visit_counts = self.tick < self.max_tick;
        let looped = self.current == self.home && last_visit_counts;
        let ticks = if looped { self.tick - 1 } else { self.tick };
        let visits = if last_visit_counts { self.ports_visited } else { self.ports_visited - 1 };
        eval_score(visits, ticks, looped)
    }

    pub fn visit(&mut self, target: &Path) {
        assert!(!self.done(), "{self:?}");
        // NOTE: +1 to account for docking.
        self.tick += target.cost + 1;
        self.current = target.goal;
        self.ports_visited += 1;
        self.missing_ports.remove(&target.goal);
    }

    pub fn unvisit(&mut self, target: &Path) {
        // Inverted logic to 'visit'.
        self.missing_ports.insert(target.goal);
        self.ports_visited -= 1;
        self.current = *target.steps.first().unwrap();
        self.tick -= target.cost + 1;
    }

    pub fn score_if_go_home(&mut self, pathfinder: &mut Pathfinder) -> i32 {
        let path_home = pathfinder.shortest_path(
            &self.current, &self.home, self.tick).expect("No path home?");
        self.visit(&path_home);
        let score = self.current_score();
        self.unvisit(&path_home);
        score
    }

    pub fn expected_final_score(&mut self, pathfinder: &mut Pathfinder) -> i32 {
        if self.done() {
            return self.current_score();
        } else if self.missing_ports.is_empty() {
            return self.score_if_go_home(pathfinder);
        }
        let closest = match pathfinder.path_to_closest(
            &self.current, &self.missing_ports, self.tick) {
            Some(path) => path,
            None => {
                return self.score_if_go_home(pathfinder);
            },
        };
        self.visit(&closest);
        let final_score = self.expected_final_score(pathfinder);
        self.unvisit(&closest);
        i32::max(final_score, self.score_if_go_home(pathfinder))
    }
}
