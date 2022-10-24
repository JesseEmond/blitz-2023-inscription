use arrayvec::ArrayVec;
use log::{debug, info};
use std::collections::{HashSet};

use crate::challenge::{MAX_PORTS, MAX_TICK_OFFSETS};
use crate::game_interface::{GameTick};
use crate::pathfinding::{Path, Pathfinder, Pos};

pub type Cost = u8;
pub type VertexId = u8;

type Adjacency = [[Cost; MAX_PORTS]; MAX_PORTS];
type TickOffsetsAdjacency = Vec<Adjacency>;
type Paths = Vec<Vec<Vec<Path>>>;

#[derive(Clone, Debug)]
pub struct Graph {
    // adjacency[tick_offset][from][to]
    pub adjacency: TickOffsetsAdjacency,
    // paths[tick_offset][from][to]
    pub paths: Paths,
    pub tick_offsets: usize,
    pub ports: ArrayVec<Pos, MAX_PORTS>,
    // Tick where paths were computed. Offsets must be computed off of this.
    pub start_tick: u16,
    pub max_ticks: u16,
}

impl Graph {
    pub fn new(pathfinder: &mut Pathfinder, game_tick: &GameTick) -> Self {
        let tick = game_tick.current_tick as u16;
        let tick_offsets = game_tick.tide_schedule.len();
        assert!(tick_offsets <= MAX_TICK_OFFSETS,
                "Bigger tick schedule that we support: {}. Update max.",
                tick_offsets);
        assert!(game_tick.map.ports.len() <= MAX_PORTS,
                "More ports than we support: {}. Update max.",
                game_tick.map.ports.len());
        let all_ports: ArrayVec<Pos, MAX_PORTS> = ArrayVec::from_iter(
            game_tick.map.ports.iter().map(Pos::from_position));
        let mut adjacency: TickOffsetsAdjacency = vec![[[0; MAX_PORTS]; MAX_PORTS]];
        let placeholder_path = Path {
            steps: Vec::new(), cost: 0, goal: Pos { x: 0, y: 0 }
        };
        let mut paths: Paths = vec![vec![
            vec![placeholder_path; all_ports.len()];
            all_ports.len()]; tick_offsets];
        for (source_idx, port) in all_ports.iter().enumerate() {
            let mut targets = HashSet::from_iter(all_ports.iter().cloned());
            targets.remove(port);
            debug!("Pathfinding from port {port:?}");
            let all_offsets_paths = pathfinder.paths_to_all_targets_by_offset(
                port, &targets, tick);
            debug!(concat!("Computed {num_paths} groups of paths ({size} ",
                          "options each). Here they are:"),
                   num_paths = all_offsets_paths.len(),
                   size = game_tick.tide_schedule.len());
            for (target_idx, target) in all_ports.iter().enumerate() {
                if target_idx != source_idx {
                    let offset_paths = all_offsets_paths.get(target)
                        .expect("No path between 2 ports, won't score high -- skip.");
                    for (offset, path) in offset_paths.iter().enumerate() {
                        assert!(path.cost < 256, "Path cost too high for u8.");
                        adjacency[offset][source_idx][target_idx] = path.cost as Cost;
                        paths[offset][source_idx][target_idx] = path.clone();
                        // Verify individual paths -- for debugging purposes only.
                        // let dist = pathfinder.distance(port, target, tick + (offset as u16));
                        // assert!(dist.unwrap() == path.cost,
                        //         "from {:?} to {:?} get direct cost of {}, but computed {}",
                        //         port, target, dist.unwrap(), path.cost);
                    }
                }
                // For source_idx == target_idx (diagonal), leave the defaults
            }
        }

        info!("Graph created: {} vertices", all_ports.len());
        Graph {
            adjacency,
            paths,
            tick_offsets,
            ports: all_ports,
            start_tick: tick + 1,  // time to spawn
            max_ticks: game_tick.total_ticks as u16,
        }
    }

    pub fn cost(&self, tick_offset: u8, from: VertexId, to: VertexId) -> Cost {
        self.adjacency[tick_offset as usize][from as usize][to as usize]
    }

    pub fn others(&self, from: VertexId) -> impl Iterator<Item=VertexId> + '_ {
        (0..self.ports.len()).map(|v| v as VertexId).filter(move |&v| v != from)
    }

    pub fn path(&self, tick_offset: u8, from: VertexId, to: VertexId) -> &Path {
        &self.paths[tick_offset as usize][from as usize][to as usize]
    }

    pub fn tick_offset(&self, tick: u16) -> u8 {
        (tick % (self.tick_offsets as u16)) as u8
    }
}
