use log::{debug, info};
use std::collections::{HashSet};

use crate::game_interface::{GameTick};
use crate::pathfinding::{Path, Pathfinder, Pos};

pub type VertexId = u8;
pub type EdgeId = u16;

#[derive(Clone)]
pub struct Edge {
    pub from: VertexId,
    pub to: VertexId,
    pub paths: Vec<Path>,
}

#[derive(Clone)]
pub struct Vertex {
    pub edges: Vec<EdgeId>,
    pub position: Pos,
    // TODO pack edges here..?
}

#[derive(Clone)]
pub struct Graph {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub tick_offsets: usize,
    // Tick where paths were computed. Offsets must be computed off of this.
    pub start_tick: u16,
    pub max_ticks: u16,
}

impl Edge {
    pub fn new(from: VertexId, to: VertexId, paths: &Vec<Path>) -> Self {
        Edge { from: from, to: to, paths: paths.clone() }
    }

    pub fn path(&self, tick: u16) -> &Path {
        unsafe {
            &self.paths.get_unchecked((tick as usize) % self.paths.len())
        }
    }

    pub fn cost(&self, tick: u16) -> u16 {
        self.path(tick).cost
    }
}

impl Vertex {
    pub fn new(position: &Pos) -> Self {
        Vertex { position: *position, edges: Vec::new() }
    }
}

impl Graph {
    pub fn new(pathfinder: &mut Pathfinder, game_tick: &GameTick) -> Self {
        let tick = game_tick.current_tick as u16;
        let all_ports: Vec<Pos> = game_tick.map.ports.iter().map(
            Pos::from_position).collect();
        let mut vertices: Vec<Vertex> = all_ports.iter().map(Vertex::new).collect();
        let mut edges = Vec::new();
        for (source_idx, port) in all_ports.iter().enumerate() {
            let mut targets = HashSet::from_iter(all_ports.iter().cloned());
            targets.remove(&port);
            debug!("Pathfinding from port {port:?}");
            let all_offsets_paths = pathfinder.paths_to_all_targets_by_offset(
                port, &targets, tick);
            debug!(concat!("Computed {num_paths} groups of paths ({size} ",
                          "options each). Here they are:"),
                   num_paths = all_offsets_paths.len(),
                   size = game_tick.tide_schedule.len());
            for (target, offset_paths) in &all_offsets_paths {
                let costs: Vec<u16> = offset_paths.iter().map(|path| path.cost).collect();
                debug!("  to {target:?}, costs {costs:?}");
            }
            for (target_idx, port) in all_ports.iter().enumerate() {
                if let Some(paths) = all_offsets_paths.get(port) {
                    let edge_id = edges.len();
                    edges.push(Edge::new(source_idx as VertexId, target_idx as VertexId, paths));
                    vertices[source_idx as usize].edges.push(edge_id as EdgeId);
                }
            }
        }

        info!("Graph created: {num_vertices} vertices, {num_edges} edges.",
              num_vertices = vertices.len(), num_edges = edges.len());
        Graph {
            vertices: vertices,
            edges: edges,
            tick_offsets: game_tick.tide_schedule.len(),
            start_tick: tick + 1,  // time to spawn
            max_ticks: game_tick.total_ticks as u16,
        }
    }

    #[inline]
    pub fn edge(&self, edge_id: EdgeId) -> &Edge {
        &self.edges[edge_id as usize]
    }

    #[inline]
    pub fn vertex(&self, vertex_id: VertexId) -> &Vertex {
        &self.vertices[vertex_id as usize]
    }

    pub fn vertex_edge_to(
        &self, vertex_id: VertexId, to: VertexId
        ) -> Option<EdgeId> {
        self.vertex(vertex_id).edges.iter()
            .filter(|&edge_id| self.edge(*edge_id).to == to)
            .take(1).cloned().next()
    }
}
