use log::{info};
use std::collections::{HashSet};

use crate::game_interface::{GameTick};
use crate::pathfinding::{Path, Pathfinder, Pos};

type VertexId = usize;
type EdgeId = usize;

pub struct Edge {
    pub from: VertexId,
    pub to: VertexId,
    pub paths: Vec<Path>,
}

pub struct Vertex {
    pub edges: Vec<EdgeId>,
    pub position: Pos,
}

pub struct Graph {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    // Tick where paths were computed. Offsets must be computed off of this.
    pub start_tick: u32,
}

impl Edge {
    pub fn new(from: VertexId, to: VertexId, paths: &Vec<Path>) -> Self {
        Edge { from: from, to: to, paths: paths.clone() }
    }

    pub fn path(&self, tick: u32) -> &Path {
        &self.paths[(tick as usize) % self.paths.len()]
    }

    pub fn cost(&self, tick: u32) -> u32 {
        self.path(tick).cost
    }
}

impl Vertex {
    pub fn new(position: &Pos) -> Self {
        Vertex { position: *position, edges: Vec::new() }
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph { vertices: Vec::new(), edges: Vec::new(), start_tick: 0, }
    }

    pub fn init(&mut self, pathfinder: &mut Pathfinder, game_tick: &GameTick) {
        self.vertices.clear();
        self.edges.clear();
        let tick = game_tick.current_tick as u32;
        self.start_tick = tick;
        let all_ports: Vec<Pos> = game_tick.map.ports.iter().map(
            Pos::from_position).collect();
        for port in all_ports.iter() {
            self.vertices.push(Vertex::new(port));
        }
        for (source_idx, port) in all_ports.iter().enumerate() {
            let mut targets = HashSet::from_iter(all_ports.iter().cloned());
            targets.remove(&port);
            info!("Pathfinding from port {port:?}");
            let all_offsets_paths = pathfinder.paths_to_all_targets_by_offset(
                port, &targets, tick);
            info!(concat!("Computed {num_paths} groups of paths ({size} ",
                          "options each). Here they are:"),
                  num_paths = all_offsets_paths.len(),
                  size = game_tick.tide_schedule.len());
            for (target, offset_paths) in &all_offsets_paths {
                let costs: Vec<u32> = offset_paths.iter().map(|path| path.cost).collect();
                info!("  to {target:?}, costs {costs:?}");
            }
            for (target_idx, port) in all_ports.iter().enumerate() {
                if let Some(paths) = all_offsets_paths.get(port) {
                    let edge_id = self.edges.len();
                    self.edges.push(Edge::new(source_idx, target_idx, paths));
                    self.vertices[source_idx].edges.push(edge_id);
                }
            }
        }

        info!("Graph created: {num_vertices} vertices, {num_edges} edges.",
              num_vertices = self.vertices.len(), num_edges = self.edges.len());
    }
}
