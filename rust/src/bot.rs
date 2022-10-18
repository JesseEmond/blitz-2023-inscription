use log::info;
use thiserror::Error;

use crate::game_interface::{Action, GameTick};
use crate::micro::{Micro, State};
use crate::pathfinding::{Path, Pos};

#[derive(Error, Debug)]
pub enum Error {
    /// If you want to implement custom errors in your bot you can add them here
    #[error("Something bad happened")]
    UnknownError,
}

pub struct Bot {
    micro: Micro,
}

impl Default for Bot {
    fn default() -> Self {
        Bot { micro: Micro { state: State::Waiting, verbose: false } }
    }
}

impl Bot {
    /// Initialize your bot
    ///
    /// This method should be used to initialize some
    /// variables you will need throughout the challenge.
    pub fn new() -> Self {
        info!("Initializing bot");
        Bot::default()
    }

    /// Make the next move according to current game tick
    ///
    /// This is where the magic happens, it's random but I bet you can do better ;)
    pub fn get_next_move(&mut self, game_tick: GameTick) -> Result<Action, Error> {
        info!("Tick {current}/{total}, pos: {pos:?}",
              current = game_tick.current_tick,
              total = game_tick.total_ticks,
              pos = game_tick.current_location);

        // TODO move logic to macro
        if game_tick.spawn_location.is_none() {
            self.micro.state = State::Spawning {
                position: Pos { x: 0, y: 0 }
            };
        } else if let State::Waiting = self.micro.state {
            self.micro.state = State::Following {
                path: Path {
                    steps: vec![Pos { x: 0, y: 0 },
                                Pos { x: 1, y: 0 },
                                Pos { x: 2, y: 0 },
                                Pos { x: 3, y: 0 },
                                Pos { x: 4, y: 0 },
                                Pos { x: 5, y: 0 }],
                    cost: 5,
                    goal: Pos { x: 5, y: 0 },
                },
                path_index: 0,
            }
        }

        // TODO add some logging of picked action
        Ok(self.micro.get_move(game_tick))
    }
}
