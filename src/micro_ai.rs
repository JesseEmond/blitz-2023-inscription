// State machine that produces bot actions to achieve micro-management goals.
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
    Docking { port: Pos },
}

fn state_short(state: &State) -> String {
    match &state {
        State::Waiting => "Waiting".to_string(),
        State::Spawning { position } => format!("Spawning ({position:?})"),
        State::Following { path, path_index } => {
            format!("Following (goal={goal:?}, idx={path_index}", goal = path.goal)
        },
        State::Docking { port } => format!("Docking ({port:?})"),
    }
}

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
                action = Some(Action::Spawn { position: position.to_position() });
                State::Docking { port: *position }
            },
            State::Docking { port } => {
                action = Some(Action::Dock);
                let current = Pos::from_position(&game_tick.current_location.unwrap());
                assert!(current == *port,
                        "Did not dock the expected port! pos:{:?} port:{:?}",
                        current, *port);
                State::Waiting
            },
            State::Following { path, path_index } => {
                let current = Pos::from_position(&game_tick.current_location.unwrap());
                assert!(!path.steps.is_empty(),
                        "Path should never end up empty!");
                assert!(path_index < &path.steps.len(),
                        "Should never get past the end of our path!");
                let expected = &path.steps[*path_index];
                assert!(*expected == current,
                        concat!(
                            "[!!!] Did not make the expected move to ",
                            "{expected:?} (current: {current:?}, ",
                            "path_idx: {path_index})."),
                        expected = expected, current = current,
                        path_index = path_index);
                let target = path.steps[path_index + 1];
                action = Some(moving_action(current, target));
                if path_index + 2 == path.steps.len() {
                    // Our next step will bring us on the last step of the path,
                    // we're ready to dock.
                    State::Docking { port: path.goal }
                } else {
                    State::Following { path: path.clone(), path_index: path_index + 1 }
                }
            },
        };
        debug!("State (after): {state:?}, action: {action:?}",
              state = state_short(&self.state));
        action.expect("Action should have been set.")
    }
}
