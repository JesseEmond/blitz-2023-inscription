use crate::game_interface::Position;

// TODO: consider packing in one int (how many bits do we need? check old games)
#[derive(Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Copy, Clone)]
pub struct Pos {
    pub x: u16,
    pub y: u16,
}

impl Pos {
    pub fn to_position(&self) -> Position {
        Position { row: self.y, column: self.x }
    }

    pub fn from_position(position: Position) -> Pos {
        Pos { x: position.column, y: position.row }
    }
}

#[derive(Debug, Clone)]
pub struct Path {
    pub steps: Vec<Pos>,
    pub cost: i32,
    pub goal: Pos,
}
