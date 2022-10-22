use log::{debug, info};
use smallvec::SmallVec;
use std::collections::{HashSet};

use crate::game_interface::{GameTick};
use crate::pathfinding::{Path, Pathfinder, Pos};

pub const MAX_TICK_OFFSETS: usize = 16;
pub const MAX_VERTICES: usize = 16;
// Because it's a complete graph.
pub const MAX_VERTEX_EDGES: usize = MAX_VERTICES - 1;
// Because it's a complete graph.
pub const MAX_EDGES: usize = MAX_VERTEX_EDGES * MAX_VERTICES;

type Vertices = SmallVec<[Vertex; MAX_VERTICES]>;
type Edges = SmallVec<[Edge; 256]>;  // note: 240 not supported
type PathPtrs = SmallVec<[PathPtr; MAX_TICK_OFFSETS]>;

pub type VertexId = u8;
pub type EdgeId = u16;
pub type PathId = u16;

// Small container for path information, for optimization purposes since the
// full list of paths take up a lot of space.
#[derive(Clone, Debug, Copy)]
pub struct PathPtr {
    pub path: PathId,
    pub cost: u16,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub from: VertexId,
    pub to: VertexId,
    pub paths: PathPtrs,
}

#[derive(Clone, Debug)]
pub struct Vertex {
    // TODO: change to fixed array of Options, ordered by vertex (so that it's
    // O(1) to know which edge leads to what vertex)?
    // TODO: remove edgeid and store edges here in-line?
    pub edges: SmallVec<[EdgeId; MAX_VERTEX_EDGES]>,
    pub position: Pos,
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub vertices: Vertices,
    pub edges: Edges,
    pub tick_offsets: usize,
    // Tick where paths were computed. Offsets must be computed off of this.
    pub start_tick: u16,
    pub max_ticks: u16,
    pub paths: Vec<Path>,
}

impl Edge {
    pub fn new(from: VertexId, to: VertexId, paths: &[PathPtr]) -> Self {
        Edge { from, to, paths: SmallVec::from(paths) }
    }

    pub fn path(&self, tick: u16) -> &PathPtr {
        unsafe {
            self.paths.get_unchecked((tick as usize) % self.paths.len())
        }
    }

    pub fn cost(&self, tick: u16) -> u16 {
        self.path(tick).cost
    }
}

impl Vertex {
    pub fn new(position: &Pos) -> Self {
        Vertex { position: *position, edges: SmallVec::new() }
    }
}

impl Graph {
    pub fn new(pathfinder: &mut Pathfinder, game_tick: &GameTick) -> Self {
        assert!(game_tick.tide_schedule.len() <= MAX_TICK_OFFSETS,
            "Can't guarantee inline storage for tide schedule of {count} elements",
            count = game_tick.tide_schedule.len());
        let tick = game_tick.current_tick as u16;
        let all_ports: Vec<Pos> = game_tick.map.ports.iter().map(
            Pos::from_position).collect();
        let mut vertices: Vertices = SmallVec::from_iter(all_ports.iter().map(Vertex::new));
        assert!(vertices.len() <= MAX_VERTICES,
                "Too many vertices to store in-line. {vertices}",
                vertices = vertices.len());
        let mut edges: Edges = SmallVec::new();
        let mut paths: Vec<Path> = Vec::new();
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
            for (target, offset_paths) in &all_offsets_paths {
                let costs: Vec<u16> = offset_paths.iter().map(|path| path.cost).collect();
                debug!("  to {target:?}, costs {costs:?}");
            }
            for (target_idx, port) in all_ports.iter().enumerate() {
                if let Some(edge_paths) = all_offsets_paths.get(port) {
                    let edge_id = edges.len();
                    let mut path_ptrs = PathPtrs::new();
                    for path in edge_paths {
                        let path_id = paths.len();
                        // TODO: could re-use existing IDs for paths that are
                        // the same on this edge.
                        path_ptrs.push(PathPtr {
                            path: path_id as PathId,
                            cost: path.cost
                        });
                        paths.push(path.clone());
                    }
                    edges.push(Edge::new(source_idx as VertexId,
                                         target_idx as VertexId,
                                         &path_ptrs));
                    vertices[source_idx as usize].edges.push(edge_id as EdgeId);
                }
            }
        }

        assert!(!vertices.spilled(), "Vertices spilled: {len}", len = vertices.len());
        assert!(!edges.spilled(), "Edges spilled: {len}", len = edges.len());

        info!("Graph created: {num_vertices} vertices, {num_edges} edges.",
              num_vertices = vertices.len(), num_edges = edges.len());
        Graph {
            vertices,
            edges,
            tick_offsets: game_tick.tide_schedule.len(),
            start_tick: tick + 1,  // time to spawn
            max_ticks: game_tick.total_ticks as u16,
            paths,
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
