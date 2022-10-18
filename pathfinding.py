# Amazing A* resource there, as always <3:
# https://www.redblobgames.com/pathfinding/a-star/implementation.html#python-dijkstra

import dataclasses
import heapq
from typing import Dict, List, Optional, Set, Tuple, TypeVar

from game_message import Position, Map as GameMap


T = TypeVar('T')


def chebyshev_dist(a: Position, b: Position) -> int:
  return max(abs(a.row - b.row), abs(a.column - b.column))


def heuristic(a: Position, goals: List[Position]) -> int:
  return min(chebyshev_dist(a, goal) for goal in goals)


class Map:
  def __init__(self, game_map: GameMap, tide_schedule: List[int], tick_offset: int):
    # We want the tick # at the time of this schedule to be able to tell how
    # offset we are withing the schedule at a given tick # in the future.
    self.tick_offset = tick_offset
    self.tide_schedule = tide_schedule
    self.topology = game_map.topology
    self.ncols = len(self.topology[0])
    self.nrows = len(self.topology)
    self.ports = game_map.ports

  def navigable(self, p: Position, tick: int) -> bool:
    return self.topology[p.row][p.column] < self.tide(tick)

  def tide(self, tick: int) -> int:
    idx = (tick - self.tick_offset) % len(self.tide_schedule)
    return self.tide_schedule[idx]

  def neighbors(self, p: Position, tick: int) -> List[Position]:
    DELTAS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    return [Position(row=p.row + dy, column=p.column + dx)
            for (dx, dy) in DELTAS
            if p.row + dy >= 0 and 
               p.row + dy < self.nrows and
               p.column + dx >= 0 and
               p.column + dx < self.ncols and
               self.navigable(Position(row=p.row + dy, column=p.column + dx), tick + 1)]

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


@dataclasses.dataclass(eq=True, frozen=True, order=True)
class State:
  pos: Position
  wait: int


def reconstruct_path(came_from: Dict[State, State],
                     start: Position, goal: Position) -> List[Position]:
  current: State = State(pos=goal, wait=0)
  path: List[Position] = []
  if current not in came_from: # no path was found
    return []
  while current.pos != start or current.wait > 0:
    path.append(current.pos)
    current = came_from[current]
  path.reverse()
  return path


def a_star_search(graph: Map, start: Position, goals: Set[Position], tick: int):
  frontier = PriorityQueue()
  start_state = State(pos=start, wait=0)
  frontier.put(start_state, 0)
  came_from: Dict[State, Optional[State]] = {}
  cost_so_far: Dict[Position, int] = {}
  came_from[start_state] = None
  cost_so_far[start_state] = 0
  goal = None
  start_tick = tick
  
  while not frontier.empty():
    current: State = frontier.get()
    current_tick = start_tick + cost_so_far[current]
    
    if current.pos in goals:
      goal = current.pos
      break
    
    options = [State(pos=n, wait=0)
               for n in graph.neighbors(current.pos, current_tick)]
    if current.wait < len(graph.tide_schedule):
      # No point in waiting longer than a full cycle
      options.append(State(current.pos, current.wait + 1))
    for next_ in options:
      new_cost = cost_so_far[current] + 1  # graph.cost(current, next_)
      if next_ not in cost_so_far or new_cost < cost_so_far[next_]:
        cost_so_far[next_] = new_cost
        priority = new_cost + heuristic(next_.pos, goals)
        frontier.put(next_, priority)
        came_from[next_] = current
  
  return came_from, cost_so_far, goal


def distance(graph: Map, src: Position, dst: Position, tick: int) -> Optional[int]:
  _, cost_so_far, _ = a_star_search(graph, src, set([dst]), tick)
  return cost_so_far.get(State(pos=dst, wait=0))


def shortest_path(graph: Map, src: Position, dst: Position, tick: int) -> List[Position]:
  came_from, _, _ = a_star_search(graph, src, set([dst]), tick)
  return reconstruct_path(came_from, src, dst)


def path_to_closest(
    graph: Map, src: Position, goals: Set[Position], tick: int
) -> Tuple[List[Position], Optional[int]]:
  came_from, cost_so_far, goal = a_star_search(graph, src, goals, tick)
  return (reconstruct_path(came_from, src, goal),
          cost_so_far.get(State(pos=goal, wait=0)))
