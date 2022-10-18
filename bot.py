from game_message import Tick, Position, Action, Spawn, Sail, Dock, Anchor, directions

import pathfinding

# TODO: questions
# - must you be *on* the port to dock?
# - what is 'isOver'?
# - can the tide move by more than 1? Does it loop somewhat? Can it be ~modeled
# - what is our turn time budget?
# - can not get tide min / max in practice?
# - topology is [column][row]?


def direction_towards(src: Position, dst: Position) -> str:
  # x, y
  d = [dst.column - src.column, dst.row - src.row]
  # Normalize to 1/-1
  if d[0] != 0:
    d[0] //= abs(d[0])
  if d[1] != 0:
    d[1] //= abs(d[1])
  d = tuple(d)
  if d == (+0, -1): return 'N'
  if d == (+1, -1): return 'NE'
  if d == (+1, +0): return 'E'
  if d == (+1, +1): return 'SE'
  if d == (+0, +1): return 'S'
  if d == (-1, +1): return 'SW'
  if d == (-1, +0): return 'W'
  if d == (-1, -1): return 'NW'
  print(f"[emond] ERROR, should not try to naviguate to itself: {src}, {dst}, returning 'N'")
  return 'N'


class Bot:
  def __init__(self):
    print("Initializing your super mega duper bot")
    self.path = []
    self.target_port = None
    self.graph = None
    self.unseen_ports = None

  def pick_spawn(self, tick: Tick) -> Position:
    # TODO smarter spawn picking
    min_tide = tick.tideSchedule[0]  # worst case we'll wait a bit on high tide
    self.graph = pathfinding.Map(tick.map, min_tide)
    navigable_ports = [port for port in tick.map.ports
                       if self.graph.navigable(port)]
    self.unseen_ports = set(navigable_ports)
    if not navigable_ports:
      print("[emond] No port is navigable..? Something is off.")
      return tick.map.ports[0]
    return navigable_ports[0]

  def go_home(self, tick: Tick) -> None:
      self.target_port = tick.spawnLocation
      self.path = pathfinding.shortest_path(self.graph, tick.currentLocation,
                                            tick.spawnLocation)

  def pick_next_port(self, tick: Tick) -> None:
    # TODO: this doesn't take into account TSP optimization, just sticks to
    # the given ports ordering.
    # TODO: this assumes we can freely go between ports at the starting tide,
    # which won't be true sometimes (depending on tide schedule)
    if not self.unseen_ports:  # No more ports to see!
      print("[emond] No more ports to see! Going home.")
      self.go_home(tick)
      return
    path, cost = pathfinding.path_to_closest(self.graph, tick.currentLocation,
                                             self.unseen_ports)
    if not path:
      print("[emond] No reachable ports from here. Going home")
      self.go_home(tick)  # Can't reach any
      return

    goal = path[-1]
    cost_go_back = pathfinding.distance(self.graph, goal, tick.spawnLocation)
    if cost_go_back is None:
      print(f"[emond] Can't go home from {goal}... This is weird. Going home")
      # Wouldn't be able to come back from this one
      self.go_home(tick)
      return

    # +1s to account for docking
    total_cost = cost + 1 + cost_go_back + 1
    if tick.currentTick + total_cost > tick.totalTicks:
      print(f"[emond] Closest new goal is {goal}, but would cost {cost} to "
            f"get there, {cost_go_back} to go back, would bring us at tick "
            f"{tick.currentTick + total_cost}/tick.totalTicks. No time.")
      self.go_home(tick)
      return

    self.path = path
    self.target_port = goal
        
  def get_next_move(self, tick: Tick) -> Action:
    """
    Here is where the magic happens, for now the move is random. I bet you can do better ;)
    """
    print(f"[emond] tick {tick.currentTick}/{tick.totalTicks}. Position: {tick.currentLocation}. Target: {self.target_port}")
    print(f"[emond] tide schedule: {tick.tideSchedule}")
    print(f"[emond] path: {self.path}")

    # Skip the first tick -- no tide information then. Skip this turn.
    if tick.currentTick == 0:
      # TODO: ideally don't do that..?
      return Anchor()  

    # Haven't spawned yet?
    if tick.currentLocation is None:
      print("[emond] here is a sample map:")
      print("[emond] --- START ---")
      print(tick)
      print("[emond] --- END ---")
      self.target_port = self.pick_spawn(tick)
      print(f"[emond] Spawing on {self.target_port}")
      return Spawn(self.target_port)

    # Reached our target port?
    if tick.currentLocation == self.target_port:
      self.unseen_ports.discard(self.target_port)
      print(f"[emond] Reached port {self.target_port}, docking.")
      print(f"[emond] Visited so far: {tick.visitedPortIndices}, {len(self.unseen_ports)} left that are reachable")
      # Not our very first port in our journey
      if not tick.visitedPortIndices or tick.currentLocation != tick.spawnLocation:
        self.pick_next_port(tick)
        print(f"[emond] Next, going to: {self.target_port}, path: {self.path}")
      return Dock()

    # Successfully got to our next path step? Remove it from our path!
    # TODO: not necessary if we model tides
    if tick.currentLocation == self.path[0]:
      self.path.pop(0)

    # Navigate towards our next path step.
    direction = direction_towards(tick.currentLocation, self.path[0])
    print(f"[emond] Sailing {direction}, to go from {tick.currentLocation} to {self.path[0]}")
    return Sail(direction)
