use log::{error, info};

use crate::pathfinding::{Path, Pos};
use crate::game_interface::{Action, Direction, GameTick};

fn get_direction(from: Pos, to: Pos) -> Direction {
    if from.x == to.x && from.y > to.y { Direction::N }
    else if from.x < to.x && from.y > to.y { Direction::NE }
    else if from.x < to.x && from.y == to.y { Direction::E }
    else if from.x < to.x && from.y < to.y { Direction::SE }
    else if from.x == to.x && from.y < to.y { Direction::S }
    else if from.x > to.x && from.y < to.y { Direction::SW }
    else if from.x > to.x && from.y == to.y { Direction::W }
    else if from.x > to.x && from.y > to.y { Direction::N }
    else { panic!("Bad direction logic, from: {from:?} to: {to:?}") }
}

fn moving_action(current: Pos, target: Pos) -> Action {
    if current == target {
        Action::Anchor
    } else {
        Action::Sail { direction: get_direction(current, target) }
    }
}

#[derive(Debug)]
pub enum State {
    Waiting,
    Spawning { position: Pos },
    Following {
        path: Path,
        // Index to the move we just made.
        path_index: usize,
    },
    Docking,
}

// Micro-management of our boat.
pub struct Micro {
    pub state: State,
    pub verbose: bool,
}

impl Micro {
    pub fn get_move(&mut self, game_tick: &GameTick) -> Action {
        if self.verbose {
            info!("State (before): {state:?}", state = self.state);
        }
        let mut action: Option<Action> = None;
        self.state = match &self.state {
            State::Waiting => {
                action = Some(Action::Anchor);
                State::Waiting
            }
            State::Spawning { position } => {
                let position = position.to_position();
                action = Some(Action::Spawn { position: position });
                State::Docking
            },
            State::Docking => {
                action = Some(Action::Dock);
                State::Waiting
            },
            State::Following { path, path_index } => {
                let current = Pos::from_position(&game_tick.current_location.unwrap());
                assert!(!path.steps.is_empty(),
                        "Path should never end up empty!");
                assert!(path_index < &path.steps.len(),
                        "Should never get past the end of our path!");
                let expected = &path.steps[*path_index];
                if *expected != current {
                    // TODO: multiline string in error?
                    error!("[!!!] Did not make the expected move to {expected:?} (current: {current:?}, path_idx: {path_index}). Trying again.");
                    action = Some(moving_action(current, *expected));
                    State::Following { path: path.clone(), path_index: *path_index }
                } else {
                    let target = path.steps[path_index + 1];
                    action = Some(moving_action(current, target));
                    if path_index + 1 >= path.steps.len() {
                        // Our next step will bring us on the last step of the path,
                        // we're ready to dock.
                        State::Docking
                    } else {
                        State::Following { path: path.clone(), path_index: path_index + 1 }
                    }
                }
            },
        };
        if self.verbose {
            info!("State (after): {state:?}, action: {action:?}", state = self.state);
        }
        action.expect("Action should have been set.")
    }
}
