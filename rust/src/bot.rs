// TODOs
// [x] Add timing info logging
// [x] Debug game #2614 -- can reproduce timeout locally?
// [ ] Optimize to have a safer first tick
// [ ] Re-create greedy bot to get eval comparison
// [ ] Add more sweepable hyperparameters/features
use log::info;
use std::time::{Instant};
use thiserror::Error;

use crate::game_interface::{Action, GameTick};
use crate::micro_ai::{Micro, State};
use crate::macro_ai::{Macro};

#[derive(Error, Debug)]
pub enum Error {
    /// If you want to implement custom errors in your bot you can add them here
    #[error("Something bad happened")]
    UnknownError,
}

pub struct Bot {
    ai_micro: Micro,
    ai_macro: Macro,
}

impl Default for Bot {
    fn default() -> Self {
        Bot {
            ai_micro: Micro { state: State::Waiting },
            ai_macro: Macro::new(),
        }
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
    pub fn get_next_move(&mut self, game_tick: &GameTick) -> Result<Action, Error> {
        let start = Instant::now();
        info!("Tick {current}/{total}, pos: {pos:?}",
              current = game_tick.current_tick,
              total = game_tick.total_ticks,
              pos = game_tick.current_location);

        if game_tick.current_tick == 0 {
            self.ai_macro.init(game_tick);
        }

        self.ai_macro.assign_state(&mut self.ai_micro, game_tick);
        let game_move = self.ai_micro.get_move(game_tick);
        info!("Tick overall time: {:?}", start.elapsed());
        Ok(game_move)
    }

    pub fn is_done(&self, game_tick: &GameTick) -> bool {
        game_tick.is_over
    }
}
