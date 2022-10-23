import dataclasses
import networkx as nx
import matplotlib.pyplot as plt
import sys
from typing import List, Tuple

assert len(sys.argv) > 1, f'Usage: {sys.argv[0]} <logs_file.txt>'
logs_file = sys.argv[1]
with open(logs_file, 'r') as f:
  log_lines = [line.strip() for line in f.readlines()]
  # Strip logging prefix
  lines = [line[line.index(']')+2:] for line in log_lines if ']' in line]


def get_logged(tag: str) -> List[str]:
  return [line[len(tag):] for line in lines if line.startswith(tag)]


@dataclasses.dataclass
class Vertex:
  x: int
  y: int
  edges: List[int]


@dataclasses.dataclass
class Edge:
  from_: int
  to: int
  costs: List[int]


@dataclasses.dataclass
class Graph:
  start_tick: int
  max_ticks: int
  vertices: List[Vertex]
  edges: List[Edge]


@dataclasses.dataclass
class Ant:
  start: int
  edges: List[int]
  score: int


def parse_vertex(line: str) -> Vertex:
  x, y, edges = line.split(' ', 2)
  x, y = int(x), int(y)
  edges = [int(e) for e in edges[1:-1].split(', ')]
  return Vertex(x=x, y=y, edges=edges)


def parse_edge(line: str) -> Edge:
  from_, to, costs = line.split(' ', 2)
  from_, to = int(from_), int(to)
  costs = [int(c) for c in costs[1:-1].split(', ')]
  return Edge(from_=from_, to=to, costs=costs)


def parse_graph() -> Graph:
  start_tick = int(get_logged('[LOGGING_GRAPH_START_TICK]')[0])
  max_ticks = int(get_logged('[LOGGING_GRAPH_MAX_TICK]')[0])
  vertices = [parse_vertex(line)
              for line in get_logged('[LOGGING_GRAPH_VERTICES]')]
  edges = [parse_edge(line) for line in get_logged('[LOGGING_GRAPH_EDGES]')]
  return Graph(start_tick=start_tick, max_ticks=max_ticks, vertices=vertices,
               edges=edges)


def parse_ant(line: str) -> Ant:
  ints = [int(w.replace('[', '').replace(']', '').replace(',', '')) for w in line.split(' ')]
  return Ant(start=ints[0], score=ints[-1], edges=ints[1:-1])


def parse_pheromones() -> List[List[float]]:
  lines = get_logged('[LOGGING_PHEROMONES]')
  pheromones = []
  for i, line in enumerate(lines):
    it, data = line.split(' ', 1)
    assert int(it) == i
    pheromones.append([float(p) for p in data[1:-1].split(', ')])
  return pheromones


def parse_heuristics() -> List[Tuple[float, float]]:
  lines = [line.split(' ') for line in get_logged('[LOGGING_HEURISTIC]')]
  return [(float(a), float(b)) for a, b in lines]


def parse_weights() -> List[List[Tuple[float, float]]]:
  lines = get_logged('[LOGGING_WEIGHTS]')
  weights = []
  for line in lines:
    it, _, e, minw, maxw = line.split()
    it, e = int(it), int(e)
    if len(weights) <= it:
      weights.append([])
    assert len(weights[it]) == e
    weights[it].append((float(minw), float(maxw)))
  return weights


def parse_ants() -> List[List[Ant]]:
  lines = get_logged('[LOGGING_ANT]')
  turns = []
  ants = []
  for line in lines:
    it, ant = line.split(' ', 1)
    if int(it) == 0 and ants:
      turns.append(ants)
      ants = []
    ants.append(parse_ant(ant))
  turns.append(ants)
  return turns


graph = parse_graph()

local_ants = [parse_ant(line) for line in get_logged('[LOGGING_LOCAL_BEST_ANT]')]
global_ants = [parse_ant(line) for line in get_logged('[LOGGING_GLOBAL_BEST_ANT]')]
pheromones = parse_pheromones()
heuristics = parse_heuristics()
weights = parse_weights()
ants = parse_ants()

pheromone_min, pheromone_max = min(p for pheros in pheromones for p in pheros), max(p for pheros in pheromones for p in pheros)
heuristic_min, heuristic_max = min(hmin for hmin, _ in heuristics), max(hmax for _, hmax in heuristics)
weights_min, weights_max = min(wmin for ws in weights for wmin, _ in ws), max(wmax for ws in weights for _, wmax in ws)

# Flip y to match what we see visually -- (0, 0) is top left in the game.
pos = {v_id: (v.x, -v.y) for v_id, v in enumerate(graph.vertices)}
G = nx.DiGraph()
for edge in graph.edges:
  G.add_edge(edge.from_, edge.to)

fig, axs = plt.subplots(3, 2)
plt.get_current_fig_manager().window.showMaximized()
for i in range(len(local_ants)):
  local_ant = local_ants[i]
  global_ant = global_ants[i]
  print(f"Tick #{i}")
  print(f"Local best: {local_ant.score}")
  print(f"Global best: {global_ant.score}")

  # Show the local & global ants
  edge_color = [(0, 1, 0) if e in global_ant.edges else (0, 0, 1) if e in local_ant.edges else (0, 0, 0, 0)
                for e in range(len(graph.edges))]
  node_color = ['green' if v == global_ant.start else 'blue' if v == local_ant.start else 'gray'
                for v in range(len(graph.vertices))]
  axs[0,0].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[0,0], edge_color=edge_color, node_color=node_color, with_labels=False)
  axs[0,0].set_title(f"Local(blue)/Global(green) Best Ants (iteration #{i})")
  axs[0,0].axis('off')

  # Show the pheromones
  edge_color = [(1, 0, 0, (p - pheromone_min) / (pheromone_max - pheromone_min)) for p in pheromones[i]]
  axs[0,1].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[0,1], edge_color=edge_color, with_labels=False)
  axs[0,1].set_title(f"Pheromones (iteration #{i})")
  axs[0,1].axis('off')

  # Show the heuristics min
  # edge_color = [(1, 0, 0, (min_ - heuristic_min) / (heuristic_max - heuristic_min)) for min_, _ in heuristics]
  # nx.draw_networkx(G, pos=pos, ax=axs[1,0], edge_color=edge_color, with_labels=False)

  # Show the heuristics max
  edge_color = [(1, 0, 0, (max_ - heuristic_min) / (heuristic_max - heuristic_min)) for _, max_ in heuristics]
  axs[1,0].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[1,0], edge_color=edge_color, with_labels=False)
  axs[1,0].set_title("Distance Heuristics (max)")
  axs[1,0].axis('off')

  # Show the weights min
  edge_color = [(1, 0, 0, (min_ - weights_min) / (weights_max - weights_min)) for min_, _ in weights[i]]
  axs[1,1].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[1,1], edge_color=edge_color, with_labels=False)
  axs[1,1].set_title(f"Sampling weights (min) (iteration #{i})")
  axs[1,1].axis('off')

  # Show the weights max
  edge_color = [(1, 0, 0, (max_ - weights_min) / (weights_max - weights_min)) for _, max_ in weights[i]]
  axs[2,1].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[2,1], edge_color=edge_color, with_labels=False)
  axs[2,1].set_title(f"Sampling weights (max) (iteration #{i})")
  axs[2,1].axis('off')

  # Show the ant paths
  def count_edge_visits(e: int):
    return sum(1 for ant in ants[i] if e in ant.edges)
  def count_node_visits(v: int):
    return sum(1 for ant in ants[i] if v in [graph.edges[e].to for e in ant.edges])
  num_edge_visits = [count_edge_visits(e) for e in range(len(graph.edges))]
  num_node_visits = [count_node_visits(v) for v in range(len(graph.vertices))]
  edge_color = [(0, 0, 1, (n - min(num_edge_visits)) / max(max(num_edge_visits) - min(num_edge_visits), 1)) for n in num_edge_visits]
  node_color = [(0, 0, 1, (n - min(num_node_visits)) / max(max(num_node_visits) - min(num_node_visits), 1)) for n in num_node_visits]
  axs[2,0].clear()
  nx.draw_networkx(G, pos=pos, ax=axs[2,0], edge_color=edge_color, node_color=node_color, with_labels=False)
  axs[2,0].set_title(f"Ant paths (iteration #{i})")
  axs[2,0].axis('off')

  plt.pause(0.1)
