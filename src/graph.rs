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

// Pack cost options for each possible tick_offset in a u64 to use up less space
// and reduce cache misses.
// 8 bits for the 'base' cost
// 4 bits per 'offset' added to base, for each tick offset possible
// 8 + 10 * 4 = 48, which fits in 64 bits.
#[derive(Copy, Clone)]
struct EdgeCosts(u64);

impl EdgeCosts {
    pub fn pack(costs: [Cost; TICK_OFFSETS]) -> Self {
        let base = *costs.iter().min().unwrap();
        let mut packed: u64 = base as u64;
        for (t, cost) in costs.iter().enumerate() {
            let diff = cost - base;
            // Note: in reality at worst we'll wait for a full tick offset cycle
            assert!(diff < 16);
            let shift = 8 + 4 * t;
            packed |= (diff as u64) << shift;
        }
        EdgeCosts(packed)
    }

    pub fn cost(&self, tick_offset: u8) -> Cost {
        let base: Cost = (self.0 & 0xFF) as Cost;
        let shift = 8 + 4 * tick_offset;
        let diff = ((self.0 >> shift) & 0x0F) as Cost;
        base + diff
    }
}

#[derive(Clone)]
pub struct Graph {
    // adjacency[to][from]
    // Note, storing [to][from] because Held-Karp iterates over 'to' options in
    // its hot loop.
    adjacency: [[EdgeCosts; MAX_PORTS]; MAX_PORTS],
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
        let mut adjacency = [[EdgeCosts(u64::MAX); MAX_PORTS]; MAX_PORTS];
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
                    // Verify individual paths -- for debugging purposes only.
                    // for target in &targets {
                    //     for offset in 0..TICK_OFFSETS {
                    //         let port = all_ports[source_idx];
                    //         let dist = pathfinder.distance(&port, target, tick + (offset as u16));
                    //         let path = &all_offsets_paths.get(&target).unwrap()[offset];
                    //         assert!(dist.unwrap() == path.cost,
                    //                 "from {:?} to {:?} get direct cost of {}, but computed {}",
                    //                 port, target, dist.unwrap(), path.cost);
                    //     }
                    // }
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
                        paths[offset][source_idx][target_idx] = path.clone();
                    }
                    let all_costs: [Cost; TICK_OFFSETS] = array_init::array_init(
                        |t| offset_paths[t].cost as Cost);
                    adjacency[target_idx][source_idx] = EdgeCosts::pack(all_costs);
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
            self.adjacency.get_unchecked(to as usize).get_unchecked(from as usize).cost(tick_offset)
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

    pub fn vertex_id(&self, pos: &Pos) -> VertexId {
        self.ports.iter().position(|&p| p == *pos).unwrap() as VertexId
    }
}
