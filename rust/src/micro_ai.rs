use log::{debug};

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
    else if from.x > to.x && from.y > to.y { Direction::NW }
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

fn state_short(state: &State) -> String {
    match &state {
        State::Waiting => "Waiting".to_string(),
        State::Spawning { position } => format!("Spawning ({position:?})"),
        State::Following { path, path_index } => {
            format!("Following (goal={goal:?}, idx={path_index}", goal = path.goal)
        },
        State::Docking => "Docking".to_string(),
    }
}

// Micro-management of our boat.
pub struct Micro {
    pub state: State,
}

impl Micro {
    pub fn get_move(&mut self, game_tick: &GameTick) -> Action {
        debug!("State (before): {state:?}",
              state = state_short(&self.state));
        let action: Option<Action>;
        self.state = match &self.state {
            State::Waiting => {
                action = Some(Action::Anchor);
                State::Waiting
            }
            State::Spawning { position } => {
                let position = position.to_position();
                action = Some(Action::Spawn { position });
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
                    panic!(concat!(
                            "[!!!] Did not make the expected move to ",
                            "{expected:?} (current: {current:?}, ",
                            "path_idx: {path_index})."),
                            expected = expected, current = current,
                            path_index = path_index);
                } else {
                    let target = path.steps[path_index + 1];
                    action = Some(moving_action(current, target));
                    if target == path.goal {
                        // Our next step will bring us on the last step of the path,
                        // we're ready to dock.
                        State::Docking
                    } else {
                        State::Following { path: path.clone(), path_index: path_index + 1 }
                    }
                }
            },
        };
        debug!("State (after): {state:?}, action: {action:?}",
              state = state_short(&self.state));
        action.expect("Action should have been set.")
    }
}
