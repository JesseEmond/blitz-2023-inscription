// TODOs
// [x] A* that returns paths to all goals
// [x] Precompute all edges (+all possible initial tide offsets)
// [x] Graph representation
// [ ] Ant Colony Optimization for full cycles
// [ ] Follow best seen ant path
// [ ] Need to optimize to fit in 1 turn?
// [ ] Clip ant paths based on going-home options at each stage
// [ ] Mine X real old games from logs
// [ ] Make server eval setup, give score distribution on X games
// [ ] Make random hyperparam sweep
// [ ] Integrate with vizier?
use log::info;
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
            ai_micro: Micro { state: State::Waiting, verbose: true },
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
    pub fn get_next_move(&mut self, game_tick: GameTick) -> Result<Action, Error> {
        info!("Tick {current}/{total}, pos: {pos:?}",
              current = game_tick.current_tick,
              total = game_tick.total_ticks,
              pos = game_tick.current_location);

        if game_tick.current_tick == 0 {
            self.ai_macro.init_no_tide_info(&game_tick);
        } else if game_tick.current_tick == 1 {
            self.ai_macro.init_with_tide_info(&game_tick);
        }

        self.ai_macro.assign_state(&mut self.ai_micro, &game_tick);
        Ok(self.ai_micro.get_move(&game_tick))
    }
}
