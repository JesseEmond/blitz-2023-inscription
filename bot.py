from game_message import Tick, Position, Action, Spawn, Sail, Dock, Anchor, directions

import pathfinding

# TODO: questions
# - rules of movement: related to source tide-topology? What about dest?
# - must you be *on* the port to dock?
# - can the tide move by more than 1? Does it loop somewhat? Can it be ~modeled
# - what is our turn compute time budget?
# - can not get tide min / max in practice?


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
  def __init__(self, verbose=True):
    verbose = False  # TODO: REMOVE THIS
    print("Initializing your super mega duper bot")
    self.verbose = verbose
    self.path = []
    self.target_port = None
    self.graph = None
    self.unseen_ports = set()

  def pick_spawn(self, tick: Tick) -> Position:
    # TODO smarter spawn picking
    self.graph = pathfinding.Map(tick.map, tick.tideSchedule, tick.currentTick)
    self.unseen_ports = set(tick.map.ports)
    return tick.map.ports[0]

  def go_home(self, tick: Tick) -> None:
      self.target_port = tick.spawnLocation
      self.path = pathfinding.shortest_path(
          self.graph, tick.currentLocation, tick.spawnLocation,
          tick.currentTick)

  def pick_next_port(self, tick: Tick) -> None:
    # TODO: this doesn't take into account TSP optimization, just sticks to
    # the given ports ordering.
    if not self.unseen_ports:  # No more ports to see!
      print("[emond] No more ports to see! Going home.")
      self.go_home(tick)
      return
    path, cost = pathfinding.path_to_closest(
        self.graph, tick.currentLocation, self.unseen_ports, tick.currentTick)
    if not path:
      print("[emond] No reachable ports from here. Going home")
      self.go_home(tick)  # Can't reach any
      return

    goal = path[-1]
    cost_go_back = pathfinding.distance(
        self.graph, goal, tick.spawnLocation, tick.currentTick)
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

    # TODO: that's not quite right, we can benefit from other ports that are
    # close after. We need to simulate up to N ports to know if it's worth it.
    # if total_cost * 3 >= 125:
    #   print(f"[emond] Closest new goal is {goal}, but would cost {cost} to "
    #         f"get there, {cost_go_back} to go back, {total_cost} total. Would "
    #         f"give use 125 base points, but take away {3 * total_cost}, not "
    #         "worth it. Going home.")
    #   self.go_home(tick)
    #   return

    self.path = path
    self.target_port = goal
        
  def get_next_move(self, tick: Tick) -> Action:
    """
    Here is where the magic happens, for now the move is random. I bet you can do better ;)
    """
    print(f"[emond] tick {tick.currentTick}/{tick.totalTicks}. Position: {tick.currentLocation}. Target: {self.target_port}")
    print(f"[emond] tide schedule: {tick.tideSchedule}")
    if self.verbose: print(f"[emond] path: {self.path}")

    # Skip the first tick -- no tide information then. Skip this turn.
    if tick.currentTick == 0:
      # TODO: ideally don't do that..?
      return Anchor()  

    # Haven't spawned yet?
    if tick.currentLocation is None:
      if self.verbose:
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
      print(f"[emond] Visited so far: {tick.visitedPortIndices}, {len(self.unseen_ports)} left unseen")
      # Not our very first port in our journey
      if not tick.visitedPortIndices or tick.currentLocation != tick.spawnLocation:
        self.pick_next_port(tick)
        print(f"[emond] Next, going to: {self.target_port}, path: {self.path}")
      return Dock()

    if self.path:
      # Successfully got to our next path step? Remove it from our path!
      # TODO: should not be necessary if we model tides perfectly
      if tick.currentLocation == self.path[0]:
        self.path.pop(0)
    else:
      print(f"[emond] !!! Path to {self.target_port} is empty. Going home")
      self.go_home(tick)
      if not self.path:
        print(f"[emond] !!! PATH TO HOME EMPTY")

    # Navigate towards our next path step.
    target = self.path[0] if self.path else self.target_port
    if tick.currentLocation != target:
      direction = direction_towards(tick.currentLocation, target)
      if self.verbose:
        print(f"[emond] Sailing {direction}, to go from {tick.currentLocation} to {self.path[0]}")
      return Sail(direction)
    print(f"[emond] Anchoring.")
    return Anchor()
