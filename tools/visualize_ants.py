import dataclasses
import dataclasses_json
import matplotlib.pyplot as plt
import networkx as nx
import sys
from typing import List, Tuple

assert len(sys.argv) > 2, f'Usage: {sys.argv[0]} <logs_file.txt> <output_prefix>'
logs_file = sys.argv[1]
with open(logs_file, 'r') as f:
  log_lines = [line.strip() for line in f.readlines()]
  # Strip logging prefix
  lines = [line[line.index(']')+2:] for line in log_lines if ']' in line]
out_prefix = sys.argv[2]


def get_logged(tag: str) -> List[str]:
  """Find the line that starts with a given tag"""
  return [line[len(tag):] for line in lines if line.startswith(tag)]


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class Pos:
  x: int
  y: int


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class GraphVisualization:
  vertices: List[Pos]
  # adjacency[from][to][offset]
  adjacency: List[List[List[int]]]


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class AntVisualization:
  start: int
  score: int
  path: List[int]


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class WeightsVisualization:
  # pheromones[from][to]
  pheromones: List[List[float]]
  # heuristics[from][to]
  min_heuristics: List[List[float]]
  max_heuristics: List[List[float]]
  # weights[from][to]
  min_weights: List[List[float]]
  max_weights: List[List[float]]


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ColonyVisualization:
  graph: GraphVisualization
  local_ants: List[AntVisualization]
  global_ants: List[AntVisualization]
  all_ants: List[List[AntVisualization]]
  weights: List[WeightsVisualization]


viz = ColonyVisualization.from_json(get_logged('[VIZ_DATA]')[0])
num_ticks = len(viz.global_ants)


@dataclasses.dataclass
class TickEdge:
  src: int
  dst: int
  pheromones: float
  min_heuristic: float
  max_heuristic: float
  min_weights_prob: float
  max_weights_prob: float
  has_global_ant: bool
  has_local_ant: bool
  ant_visits: int

edge_indices = {}
for src in range(len(viz.graph.vertices)):
  for dst in range(len(viz.graph.vertices)):
    if src == dst:
      continue
    edge_indices[(src, dst)] = len(edge_indices)

tick_edges = []
for tick in range(num_ticks):
  local_ant = viz.local_ants[tick]
  global_ant = viz.global_ants[tick]
  all_ants = viz.all_ants[tick]
  weights = viz.weights[tick]
  edges = []
  for src in range(len(viz.graph.vertices)):
    min_weights_sum = sum(
        weights.min_weights[src][d]
        for d in range(len(viz.graph.vertices))
        if src != d)
    max_weights_sum = sum(
        weights.max_weights[src][d]
        for d in range(len(viz.graph.vertices))
        if src != d)
    for dst in range(len(viz.graph.vertices)):
      if src == dst:
        continue
      edge = TickEdge(
          src=src,
          dst=dst,
          pheromones=weights.pheromones[src][dst],
          min_heuristic=weights.min_heuristics[src][dst],
          max_heuristic=weights.max_heuristics[src][dst],
          min_weights_prob=weights.min_weights[src][dst] / min_weights_sum,
          max_weights_prob=weights.max_weights[src][dst] / max_weights_sum,
          has_global_ant=False,
          has_local_ant=False,
          ant_visits=0)
      edges.append(edge)
  def visit_edges(ant: AntVisualization, fn):
    path = [ant.start] + ant.path
    for i in range(1, len(path)):
      src, dst = path[i-1], path[i]
      edge = edges[edge_indices[(src, dst)]]
      fn(edge)
  for ant in all_ants:
    def _count_visit(e):
      e.ant_visits += 1
    visit_edges(ant, _count_visit)
  def _mark_local(e):
    e.has_local_ant = True
  visit_edges(local_ant, _mark_local)
  def _mark_global(e):
    e.has_global_ant = True
  visit_edges(global_ant, _mark_global)
  tick_edges.append(edges)

# Ranges used to set intensity of colors relative to min/max seen.
all_pheromones = [e.pheromones for edges in tick_edges for e in edges]
all_min_heuristics = [e.min_heuristic for edges in tick_edges for e in edges]
all_max_heuristics = [e.max_heuristic for edges in tick_edges for e in edges]
pheromone_min, pheromone_max = min(all_pheromones), max(all_pheromones)
heuristic_min, heuristic_max = min(all_min_heuristics), max(all_max_heuristics)

# Flip y to match what we see visually -- (0, 0) is top left in the game.
pos = {v_id: (v.x, -v.y) for v_id, v in enumerate(viz.graph.vertices)}
G = nx.DiGraph()
for edge in edges:
  G.add_edge(edge.src, edge.dst)

fig, axs = plt.subplots(3, 2)
dpi = 100
fig.set_dpi(dpi)
fig.set_size_inches(1920/dpi, 1080/dpi)
for i in range(num_ticks):
  edges = tick_edges[i]
  local_ant = viz.local_ants[i]
  global_ant = viz.global_ants[i]
  ants = viz.all_ants[i]
  print(f"Tick #{i}")
  print(f"Local best: {local_ant.score}")
  print(f"Global best: {global_ant.score}")

  # Show the local & global ants
  edge_color = [(0, 1, 0) if e.has_global_ant else (0, 0, 1) if e.has_local_ant else (0, 0, 0, 0)
                for e in edges]
  node_color = ['green' if v == global_ant.start else 'blue' if v == local_ant.start else 'gray'
                for v in range(len(viz.graph.vertices))]
  axs[0,0].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[0,0], edge_color=edge_color, node_color=node_color, with_labels=False)
  axs[0,0].set_title(f"Local(blue)/Global(green) Best Ants (iteration #{i})")
  l = axs[0,0].legend(
      [f'Local best ant ({local_ant.score} points)',
       f'Global best ant ({global_ant.score} points)'])
  l.get_texts()[0].set_color('blue')
  l.get_texts()[1].set_color('green')
  axs[0,0].axis('off')

  # Show the pheromones
  # Display pheromones as a mix of local normalization and global to be able to
  # see _some_ of the movement.
  local_max_pheromones = max(e.pheromones for e in edges)
  edge_color = [(1, 0, 0,
                 0.5 * e.pheromones / pheromone_max + 0.5 * e.pheromones / local_max_pheromones)
                for e in edges]
  axs[0,1].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[0,1], edge_color=edge_color, with_labels=False)
  axs[0,1].set_title(f"Pheromones (iteration #{i})")
  axs[0,1].axis('off')

  # Show the heuristics min
  # edge_color = [(1, 0, 0, (min_ - heuristic_min) / (heuristic_max - heuristic_min)) for min_, _ in heuristics]
  # nx.draw_networkx(G, pos=pos, ax=axs[1,0], edge_color=edge_color, with_labels=False)

  # Show the heuristics max
  edge_color = [(1, 0, 0, (e.max_heuristic - heuristic_min) / (heuristic_max - heuristic_min))
                for e in edges]
  axs[1,0].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[1,0], edge_color=edge_color, with_labels=False)
  axs[1,0].set_title("Distance Heuristics (max)")
  axs[1,0].axis('off')

  # Show the weights min
  edge_color = [(1, 0, 0, e.min_weights_prob) for e in edges]
  axs[1,1].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[1,1], edge_color=edge_color, with_labels=False)
  axs[1,1].set_title(f"Sampling weights (min) (iteration #{i})")
  axs[1,1].axis('off')

  # Show the weights max
  edge_color = [(1, 0, 0, e.max_weights_prob) for e in edges]
  axs[2,1].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[2,1], edge_color=edge_color, with_labels=False)
  axs[2,1].set_title(f"Sampling weights (max) (iteration #{i})")
  axs[2,1].axis('off')

  # Show the ant paths
  def count_node_visits(v: int):
    return sum(1 for ant in ants if v in ant.path)
  num_edge_visits = [e.ant_visits for e in edges]
  num_node_visits = [count_node_visits(v) for v in range(len(viz.graph.vertices))]
  edge_color = [(0, 0, 1, (n - min(num_edge_visits)) / max(max(num_edge_visits) - min(num_edge_visits), 1)) for n in num_edge_visits]
  node_color = [(0, 0, 1, (n - min(num_node_visits)) / max(max(num_node_visits) - min(num_node_visits), 1)) for n in num_node_visits]
  axs[2,0].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[2,0], edge_color=edge_color, node_color=node_color, with_labels=False)
  axs[2,0].set_title(f"Ant paths (iteration #{i})")
  axs[2,0].axis('off')

  out_file = f'{out_prefix}_{i:06}.png'
  plt.savefig(out_file, transparent=False, facecolor='white', dpi=dpi)
  plt.pause(0.01)
