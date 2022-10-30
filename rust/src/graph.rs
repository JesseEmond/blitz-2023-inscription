use arrayvec::ArrayVec;
use log::{debug, info};
use std::collections::{HashSet};
use std::sync::{mpsc, Arc};
use std::thread;

use crate::challenge_consts::{MAX_PORTS, NUM_THREADS, TICK_OFFSETS};
use crate::game_interface::{GameTick};
use crate::pathfinding::{Path, Pathfinder, Pos};

pub type Cost = u8;
pub type VertexId = u8;

type Paths = Vec<Vec<Vec<Path>>>;

#[derive(Clone, Debug)]
pub struct Graph {
    // adjacency[to][tick_offset][from]
    pub adjacency: Vec<[[Cost; MAX_PORTS]; TICK_OFFSETS]>,
    // paths[tick_offset][from][to]
    pub paths: Paths,
    pub ports: ArrayVec<Pos, MAX_PORTS>,
    // Tick where paths were computed. Offsets must be computed off of this.
    pub start_tick: u16,
    pub max_ticks: u16,
}

impl Graph {
    pub fn new(game_tick: &Arc<GameTick>) -> Self {
        let tick = game_tick.current_tick as u16;
        let tick_offsets = game_tick.tide_schedule.len();
        assert!(tick_offsets == TICK_OFFSETS,
                "Expected a fixed tick schedule, got: {}",
                tick_offsets);
        assert!(game_tick.map.ports.len() <= MAX_PORTS,
                "More ports than we support: {}. Update max.",
                game_tick.map.ports.len());
        let all_ports: ArrayVec<Pos, MAX_PORTS> = ArrayVec::from_iter(
            game_tick.map.ports.iter().map(Pos::from_position));
        let mut adjacency: Vec<_> = vec![[[0; MAX_PORTS]; TICK_OFFSETS]; MAX_PORTS];
        let placeholder_path = Path {
            steps: Vec::new(), cost: 0, goal: Pos { x: 0, y: 0 }
        };
        let mut paths: Paths = vec![vec![
            vec![placeholder_path; all_ports.len()];
            all_ports.len()]; tick_offsets];
        let mut handles = vec![];
        let (tx, rx) = mpsc::channel();
        for i in 0..NUM_THREADS {
            let tx = tx.clone();
            let game_tick = game_tick.clone();
            let all_ports = all_ports.clone();
            handles.push(thread::spawn(move || {
                let mut pathfinder = Pathfinder::new();
                let schedule: Vec<u8> = game_tick.tide_schedule.iter()
                    .map(|&e| e as u8).collect();
                pathfinder.grid.init(&game_tick.map, &schedule);

                for j in (i..all_ports.len()).step_by(NUM_THREADS) {
                    let source_idx = j;
                    let mut targets = HashSet::from_iter(all_ports.iter().cloned());
                    targets.remove(&all_ports[source_idx]);
                    let all_offsets_paths = pathfinder.paths_to_all_targets_by_offset(
                        &all_ports[source_idx], &targets, tick);
                    tx.send((source_idx, all_offsets_paths)).unwrap();
                }
            }));
        }
        drop(tx);  // Drop the last sender, wait until all threads are done.
        while let Ok((source_idx, all_offsets_paths)) = rx.recv() {
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
                        adjacency[target_idx][offset][source_idx] = path.cost as Cost;
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
        for handle in handles {
            handle.join().unwrap();
        } 

        info!("Graph created: {} vertices", all_ports.len());
        Graph {
            adjacency,
            paths,
            ports: all_ports,
            start_tick: tick + 1,  // time to spawn
            max_ticks: game_tick.total_ticks as u16,
        }
    }

    pub fn cost(&self, tick_offset: u8, from: VertexId, to: VertexId) -> Cost {
        unsafe {
            *self.adjacency.get_unchecked(to as usize).get_unchecked(tick_offset as usize).get_unchecked(from as usize)
        }
    }

    #[inline]
    pub fn others(&self, from: VertexId) -> impl Iterator<Item=VertexId> + '_ {
        (0..self.ports.len()).map(|v| v as VertexId).filter(move |&v| v != from)
    }

    pub fn path(&self, tick_offset: u8, from: VertexId, to: VertexId) -> &Path {
        &self.paths[tick_offset as usize][from as usize][to as usize]
    }

    #[inline]
    pub fn tick_offset(&self, tick: u16) -> u8 {
        (tick % (TICK_OFFSETS as u16)) as u8
    }
}
