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
            micro.state = State::Spawning {
                position: pick_spawn(game_tick)
            };
        } else if let State::Waiting = micro.state {
            if let Some(next_port_path) = self.pick_next_port(game_tick) {
                micro.state = State::Following {
                    path: next_port_path,
                    path_index: 0,
                };
            } else {
                micro.state = State::Following {
                    path: self.path_home(game_tick).expect("No path home at the end...?"),
                    path_index: 0,
                };
            }
        } else if let State::Docking = micro.state {
            let port = Pos::from_position(&game_tick.current_location.unwrap());
            self.missing_ports.remove(&port);
        }
        // else, no-op, micro is no a task.
    }

    pub fn pick_next_port(&mut self, game_tick: &GameTick) -> Option<Path> {
        // TODO: consider if we have time to go
        // TODO: consider overall gains from going vs. costs
        let current = Pos::from_position(&game_tick.current_location.unwrap());
        self.pathfinder.path_to_closest(current, &self.missing_ports,
                                        game_tick.current_tick.into())
    }

    pub fn path_home(&mut self, game_tick: &GameTick) -> Option<Path> {
        let current = Pos::from_position(&game_tick.current_location.unwrap());
        let home = Pos::from_position(&game_tick.spawn_location.unwrap());
        self.pathfinder.shortest_path(current, home,
                                      game_tick.current_tick.into())
    }
}
