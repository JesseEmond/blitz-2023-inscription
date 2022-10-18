from game_message import Tick, Position, Action, Spawn, Sail, Dock, Anchor, directions

import pathfinding

# TODO: questions
# - must you be *on* the port to dock?
# - what is 'isOver'?
# - can the tide move by more than 1? Does it loop somewhat? Can it be ~modeled
# - what is our turn time budget?
# - can not get tide min / max in practice?
# - topology is [column][row]?


def naviguate_towards(src: Position, dst: Position) -> str:
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
    self.ports_plan = None
    self.graph = None

  def make_plan(self, tick: Tick) -> None:
    min_tide = tick.tideSchedule[0]  # worst case we'll wait a bit on high tide
    self.graph = pathfinding.Map(tick.map, min_tide)
    # TODO: this doesn't take into account TSP optimization, just sticks to
    # the given ports ordering.
    # TODO: this isn't doing any pathing, it assumes we can freely go
    # between ports, which won't be true sometimes (depending on tides) or
    # ever (depending on map topology), need pathfinding.
    cost = 1  # Start at 1 for cost of spawning
    navigable_ports = [port for port in tick.map.ports
                       if self.graph.navigable(port)]
    if not navigable_ports:
      print("[emond] No port is navigable..? Something is off.")
    print(f"[emond] skipping {len(tick.map.ports) - len(navigable_ports)} because they're unnavigable. Navigable: {navigable_ports}")
    spawn_port = navigable_ports[0]
    self.ports_plan = [spawn_port]
    self.target_port = self.ports_plan[0]

    for port in navigable_ports[1:]:
      # Do we have time to get there and back to the start?
      cost_get_there = pathfinding.distance(self.graph, self.ports_plan[-1], port)
      cost_get_back = pathfinding.distance(self.graph, port, self.ports_plan[0])
      if cost_get_there is None or cost_get_back is None:
        print(f"[emond] {port} is unreachable from {self.ports_plan[-1]}, skipping. cost_get_there={cost_get_there} cost_get_back={cost_get_back}")
        continue  # Unreachable.
      dock_costs = 2
      additional_cost = cost_get_there + cost_get_back + dock_costs
      if cost + additional_cost >= tick.totalTicks:
        continue  # Can't afford it! Skip it.
      self.ports_plan.append(port)
      cost += additional_cost

    # Go back to the initial port!
    self.ports_plan.append(self.ports_plan[0])
    print("[emond] our plan is the following: ", self.ports_plan)
    print("[emond] estimating a total cost of ", cost, " ticks, out of budget of ", tick.totalTicks)

  def plan_next_port(self, pos: Position, plan_idx: int):
    self.target_port = self.ports_plan[plan_idx]
    self.path = pathfinding.shortest_path(self.graph, pos, self.target_port)
        
  def get_next_move(self, tick: Tick) -> Action:
    """
    Here is where the magic happens, for now the move is random. I bet you can do better ;)
    """
    print(f"[emond] tick {tick.currentTick}/{tick.totalTicks}. Position: {tick.currentLocation}. Target: {self.target_port}")
    print(f"[emond] tide schedule: {tick.tideSchedule}")
    print(f"[emond] path: {self.path}")

    # Skip the first tick -- no tide information then.
    if tick.currentTick == 0:
      # TODO: ideally don't do that..?
      return Anchor()  


    # Haven't spawned yet?
    if tick.currentLocation is None:
      print("[emond] here is a sample map:")
      print("[emond] --- START ---")
      print(tick)
      print("[emond] --- END ---")
      self.make_plan(tick)
      print(f"[emond] Spawing on {self.target_port}")
      return Spawn(self.target_port)

    # Reached our target port?
    if tick.currentLocation == self.target_port:
      print(f"[emond] Reached port {self.target_port}, docking.")
      plan_idx = len(tick.visitedPortIndices) + 1
      print(f"[emond] Visited so far: {tick.visitedPortIndices}, plan_idx: {plan_idx}")
      if plan_idx < len(self.ports_plan):
        self.plan_next_port(tick.currentLocation, plan_idx)
        print(f"[emond] Next, going to: {self.target_port}, path: {self.path}")
      return Dock()

    # Successfully got to our next path step? Remove it from our path!
    # TODO: not necessary if we model tides
    if tick.currentLocation == self.path[0]:
      self.path.pop(0)

    # Navigate towards our next path step.
    direction = naviguate_towards(tick.currentLocation, self.path[0])
    print(f"[emond] Sailing {direction}, to go from {tick.currentLocation} to {self.path[0]}")
    return Sail(direction)
