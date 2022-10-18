// Don't cluter game_interface with implementations

use std::{array::IntoIter, iter::Cycle};

use crate::game_interface::Direction;

impl Direction {
    pub fn iter() -> Cycle<IntoIter<Direction, 8>> {
        let list = [
            Direction::N,
            Direction::NE,
            Direction::E,
            Direction::SE,
            Direction::S,
            Direction::SW,
            Direction::W,
            Direction::NW,
        ];

        list.into_iter().cycle()
    }
}
