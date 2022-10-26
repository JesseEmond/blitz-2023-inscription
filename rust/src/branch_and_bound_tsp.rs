// Exact solver for the Traveling Salesman Problem, using Branch-and-Bound.
// See https://en.wikipedia.org/wiki/Branch_and_bound
//
// At a high level, we treat adding a node to our tour as visiting a node in our
// search space, where we have a lower bound cost for each node. We visit the
// search space one level at a time.
// Once we reach a leaf node (full tour), we get an example cost, which gives us
// an upper bound for our solution. We can use this to cut nodes that have a
// lower bound >= our upper bound.
//
// The lower bound that we use is that each node in our tour will have 2 of its
// edges visited (going in, going out), so the costs to the 2 nearest neighbors
// summed up for all edges (divided by 2, since we're double counting) is a
// lower bound.

// TODO: include start vertex in search decision
// TODO: avoid recursion...?

use arrayvec::ArrayVec;
use log::{info};

use crate::challenge::{Solution, MAX_PORTS};
use crate::graph::{Graph, VertexId};

type Cost = u16;
type Mask = u32;
type Path = ArrayVec<VertexId, { MAX_PORTS + 1 }>;

struct Tour {
    cost: Cost,
    vertices: Path,
}

pub fn branch_and_bound(graph: &Graph) -> Option<Solution> {
    let mut search = SearchSpace::new(graph);
    search.solve(graph);
    search.best_tour.to_solution()
}

struct Node {
    vertex: VertexId,
    lower_bound: Cost,
    tick: u16,
}

struct SearchSpace {
    best_tour: Tour,
    min_outbounds: ArrayVec<u8, MAX_PORTS>,
    min_inbounds: ArrayVec<u8, MAX_PORTS>,
}

impl SearchSpace {
    pub fn new(graph: &Graph) -> Self {
        SearchSpace {
            best_tour: Tour { cost: Cost::MAX, vertices: Path::new() },
            min_outbounds: ArrayVec::from_iter((0..graph.ports.len()).map(|from| {
                let from = from as VertexId;
                graph.others(from as VertexId).map(|to| {
                    let to = to as VertexId;
                    (0..graph.tick_offsets).map(|t| graph.cost(t as u8, from, to))
                        .min().unwrap()
                }).min().unwrap()
            })),
            min_inbounds: ArrayVec::from_iter((0..graph.ports.len()).map(|to| {
                let to = to as VertexId;
                graph.others(to as VertexId).map(|from| {
                    let from = from as VertexId;
                    (0..graph.tick_offsets).map(|t| graph.cost(t as u8, from, to))
                        .min().unwrap()
                }).min().unwrap()
            })),
        }
    }

    pub fn solve(&mut self, graph: &Graph) {
        // TODO: consider starting points in search
        let unseen = ((1 << graph.ports.len()) - 1) as Mask;
        // TODO: loop on start options, pick best one
        let start: VertexId = 0;
        let unseen = unseen & !((1 << start) as Mask);
        let mut path = Path::from_iter([start]);
        let tick = graph.start_tick + 1;  // +1 to dock start
        let root = Node {
            vertex: start,
            lower_bound: self.global_lower_bound(graph),
            tick
        };
        self.expand(graph, unseen, &mut path, &root);
    }

    pub fn expand(
        &mut self, graph: &Graph, unseen: Mask, path: &mut Path, parent: &Node
        ) {
        let offset = graph.tick_offset(parent.tick);
        if unseen == 0 {  // Complete tour!
            let home = path[0];
            // Note: no +1 since the last dock is free
            let cost = parent.tick + graph.cost(offset, parent.vertex, home) as u16;
            if cost < self.best_tour.cost {
                let mut path = path.clone();
                path.push(path[0]);
                self.best_tour.cost = cost;
                self.best_tour.vertices = path;
                info!("New best tour! {:?} cost={}", self.best_tour.vertices,
                      self.best_tour.cost);
            }
        } else {
            let mut nodes = ArrayVec::<Node, { MAX_PORTS - 1 }>::new();
            for k in 0..graph.ports.len() {
                let k = k as VertexId;
                let mask = (1 << k) as Mask;
                if mask & unseen == 0 {
                    continue;
                }
                let cost = graph.cost(offset, parent.vertex, k) as Cost;
                let tick = parent.tick + cost + 1;  // + 1 to dock
                // Note: -1 to floor
                let correction = ((
                    self.min_outbounds[parent.vertex as usize] +
                    self.min_inbounds[k as usize] - 1) / 2) as Cost;
                let lower_bound = parent.lower_bound + cost - correction;
                nodes.push(Node { lower_bound, tick, vertex: k });
            }
            // TODO: helps at all?
            nodes.sort_by_key(|n| n.lower_bound);
            for node in &nodes {
                if node.lower_bound >= self.best_tour.cost {
                    continue;  // can't beat our best -- skip
                }
                let unseen = unseen & !((1 << node.vertex) as Mask);
                path.push(node.vertex);
                self.expand(graph, unseen, path, &node);
                path.pop();
            }
        }
    }

    fn global_lower_bound(&self, graph: &Graph) -> Cost {
        let mut lower_bound: Cost = 0;
        for u in 0..graph.ports.len() {
            lower_bound += (self.min_outbounds[u as usize]
                            + self.min_inbounds[u as usize]) as Cost;
        }
        // +1 to ceil
        (lower_bound + 1) / 2
    }
}

impl Tour {
    pub fn to_solution(&self) -> Option<Solution> {
        None  // TODO impl
    }
}
