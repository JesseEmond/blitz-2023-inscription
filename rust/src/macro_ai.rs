use log::{info};
use std::collections::HashSet;

use crate::game_interface::{GameTick};
use crate::micro_ai::{Micro, State};
use crate::pathfinding::{Path, Pathfinder, Pos};

fn pick_spawn(game_tick: &GameTick) -> Pos {
    // TODO: smarter spawn logic
    Pos::from_position(game_tick.map.ports.first().expect("No ports...?"))
}

pub struct Macro {
    pathfinder: Pathfinder,
    missing_ports: HashSet<Pos>,
}

impl Macro {
    pub fn new() -> Self {
        Macro { pathfinder: Pathfinder::new(), missing_ports: HashSet::new() }
    }

    // Called on tick 0, before tide schedule is available
    pub fn init_no_tide_info(&mut self, game_tick: &GameTick) {
        // TODO: on tick 0, could precompute path pairs assuming
        // max tide, then on tick 1 only recompute the ones that
        // won't work with tide schedule?
        self.pathfinder.graph.init_map(&game_tick.map);
        self.missing_ports = HashSet::from_iter(
            game_tick.map.ports.iter().map(Pos::from_position));
    }

    // Called on tick 1, after tide schedule is available.
    pub fn init_with_tide_info(&mut self, game_tick: &GameTick) {
        let schedule: Vec<u8> = game_tick.tide_schedule.iter().map(|&e| e as u8).collect();
        self.pathfinder.graph.init_tide_schedule(
            &schedule, game_tick.current_tick.into());
    }

    pub fn assign_state(&mut self, micro: &mut Micro, game_tick: &GameTick) {
        if game_tick.spawn_location.is_none() {
            let spawn = pick_spawn(game_tick);
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
        let home = Pos::from_position(&game_tick.spawn_location.unwrap());
        let cost_go_back = match self.pathfinder.distance(
            &current, &home, tick) {
            Some(cost) => cost,
            None => {
                panic!(concat!(
                        "[MACRO] Could not find a way back home from target ",
                        "{goal:?}, home: {home:?}. Should not happen."),
                        goal = closest.goal, home = home);
            },
        };

        // NOTE: +1s to account for docking
        let total_cost = closest.cost + 1 + cost_go_back + 1;

        if game_tick.current_tick + (total_cost as u16) > game_tick.total_ticks {
            info!(concat!("[MACRO] Closest new goal is {goal:?}, but would ",
                          "cost {cost} to get there, {cost_go_back} to go ",
                          "back, would bring us at tick {tick}/{total}. ",
                          "No time."),
                  goal = closest.goal, cost = closest.cost,
                  cost_go_back = cost_go_back, tick = game_tick.current_tick,
                  total = game_tick.total_ticks);
            None
        } else {
            Some(closest)
        }
    }

    pub fn path_home(&mut self, game_tick: &GameTick) -> Option<Path> {
        let current = Pos::from_position(&game_tick.current_location.unwrap());
        let home = Pos::from_position(&game_tick.spawn_location.unwrap());
        self.pathfinder.shortest_path(&current, &home,
                                      game_tick.current_tick.into())
    }
}
