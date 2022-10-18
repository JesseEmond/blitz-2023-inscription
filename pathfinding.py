# Amazing A* resource there, as always <3:
# https://www.redblobgames.com/pathfinding/a-star/implementation.html#python-dijkstra

import heapq
from typing import Dict, List, Optional, Tuple, TypeVar

from game_message import Position, Map as GameMap


T = TypeVar('T')


def chebyshev_dist(a: Position, b: Position) -> int:
  return max(abs(a.row - b.row), abs(a.column - b.column))


heuristic = chebyshev_dist


class Map:
  def __init__(self, game_map: GameMap, min_tide: int):
    # TODO: tide forecasting with schedule
    self.min_tide = min_tide
    self.topology = game_map.topology
    self.ncols = len(self.topology[0])
    self.nrows = len(self.topology)
    self.ports = game_map.ports

  def navigable(self, p: Position) -> bool:
    return self.topology[p.row][p.column] < self.min_tide

  def neighbors(self, p: Position) -> List[Position]:
    DELTAS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    return [Position(row=p.row + dy, column=p.column + dx)
            for (dx, dy) in DELTAS
            if p.row + dy >= 0 and 
               p.row + dy < self.nrows and
               p.column + dx >= 0 and
               p.column + dx < self.ncols and
               self.navigable(Position(row=p.row + dy, column=p.column + dx))]

  def debug_print(self, highlights: List[Position]):
    for row in range(self.nrows):
      for col in range(self.ncols):
        pos = Position(row=row, column=col)
        if pos in highlights:
          print('H', end='')
        elif pos in self.ports:
          print('P', end='')
        elif self.navigable(pos):
          print('.', end='')
        else:
          print('#', end='')
      print()


class PriorityQueue:
  def __init__(self):
    self.elements: List[Tuple[int, T]] = []
    
  def empty(self) -> bool:
    return not self.elements
    
  def put(self, item: T, priority: int):
    heapq.heappush(self.elements, (priority, item))
    
  def get(self) -> T:
    return heapq.heappop(self.elements)[1]


def reconstruct_path(came_from: Dict[Position, Position],
                     start: Position, goal: Position) -> List[Position]:
  current: Position = goal
  path: List[Position] = []
  if goal not in came_from: # no path was found
    return []
  while current != start:
    path.append(current)
    current = came_from[current]
  path.reverse()
  return path


def a_star_search(graph: Map, start: Position, goal: Position):
  frontier = PriorityQueue()
  frontier.put(start, 0)
  came_from: Dict[Position, Optional[Position]] = {}
  cost_so_far: Dict[Position, int] = {}
  came_from[start] = None
  cost_so_far[start] = 0
  
  while not frontier.empty():
    current: Position = frontier.get()
    
    if current == goal:
      break
    
    for next_ in graph.neighbors(current):
      new_cost = cost_so_far[current] + 1  # graph.cost(current, next_)
      if next_ not in cost_so_far or new_cost < cost_so_far[next_]:
        cost_so_far[next_] = new_cost
        priority = new_cost + heuristic(next_, goal)
        frontier.put(next_, priority)
        came_from[next_] = current
  
  return came_from, cost_so_far


def distance(graph: Map, src: Position, dst: Position) -> Optional[int]:
  _, cost_so_far = a_star_search(graph, src, dst)
  return cost_so_far.get(dst)


def shortest_path(graph: Map, src: Position, dst: Position) -> List[Position]:
  came_from, _ = a_star_search(graph, src, dst)
  return reconstruct_path(came_from, src, dst)
