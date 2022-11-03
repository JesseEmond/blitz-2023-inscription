// TODOs
// - pass solver from bot creation
// - determine solver from args cmdline
// - ACO parse hyperparams based on arg, sweep using that (in /tmp/)
// - improve/sweep ACO on game #10589, to get closer to 3836
// - implement "slow" pathfinding
// - implement "slow" held-karp
// - write-up outline
// - pathfinding optimizations ablation
// - graph optimizations ablation
// - held-karp optimizations ablation
// - write-up
use log::info;
use std::time::{Instant};
use std::sync::Arc;
use thiserror::Error;

use crate::game_interface::{Action, GameTick};
use crate::graph::{Graph};
use crate::micro_ai::{Micro, State};
use crate::macro_ai::{Macro};
use crate::solvers::{Solver};

#[derive(Error, Debug)]
pub enum Error {
    /// If you want to implement custom errors in your bot you can add them here
    #[error("Something bad happened")]
    UnknownError,
}

pub struct Bot {
    ai_micro: Micro,
    ai_macro: Macro,
    solver: Box<dyn Solver>,
}

impl Bot {
    /// Initialize your bot
    ///
    /// This method should be used to initialize some
    /// variables you will need throughout the challenge.
    pub fn new(solver: Box<dyn Solver>) -> Self {
        info!("Initializing bot");
        Bot {
            ai_micro: Micro { state: State::Waiting },
            ai_macro: Macro::new(),
            solver,
        }
    }

    fn init(&mut self, game_tick: Arc<GameTick>) {
        let planning_start = Instant::now();

        // This is a bit verbose, but we always want this on server.
        info!("--- TICK DUMP BEGIN ---");
        info!("{game_tick:?}");
        info!("--- TICK DUMP END ---");

        let graph_start = Instant::now();
        let graph: Arc<Graph> = Arc::new(Graph::new(&game_tick));
        info!("Graph was built in {:?}", graph_start.elapsed());

        let solution = self.solver.solve(&graph).expect("no solution found");
        self.ai_macro.init(&graph, solution);

        info!("Planning took {:?}", planning_start.elapsed());
    }

    /// Make the next move according to current game tick
    ///
    /// This is where the magic happens, it's random but I bet you can do better ;)
    pub fn get_next_move(&mut self, game_tick: Arc<GameTick>) -> Result<Action, Error> {
        let start = Instant::now();
        info!("Tick {current}/{total}, pos: {pos:?}",
              current = game_tick.current_tick,
              total = game_tick.total_ticks,
              pos = game_tick.current_location);

        if game_tick.current_tick == 0 {
            self.init(game_tick.clone());
        }
        if self.ai_macro.give_up {
            // Get a score of 0 by docking twice on the first port.
            if game_tick.current_tick == 0 {
                return Ok(Action::Spawn { position: game_tick.map.ports[0] });
            } else {
                return Ok(Action::Dock {});
            }
        }

        self.ai_macro.assign_state(&mut self.ai_micro, &game_tick);
        let game_move = self.ai_micro.get_move(&game_tick);
        info!("Tick overall time: {:?}", start.elapsed());
        Ok(game_move)
    }

    pub fn is_done(&self, game_tick: &GameTick) -> bool {
        game_tick.is_over
    }
}
