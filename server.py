#!/usr/bin/env python

import asyncio
import json
import sys
import time
import websockets

from game_message import Action, Anchor, Dock, Position, Sail, Spawn, Tick, directions
import seen_games


class Game:
  def __init__(self):
    # Hacky copy
    tick = Tick.from_dict(json.loads(json.dumps(seen_games.GAME2.to_dict())))
    self.tick = tick
    self.tick.currentTick = 0  # seen games come from logs at tick=1, reset.
    self.schedule = self.tick.tideSchedule
    self.tick.tideSchedule = [0]  # Start with no info on initial tick (like server)
    self.nrows = len(self.tick.map.topology)
    self.ncols = len(self.tick.map.topology[0])

  def apply(self, action: Action) -> None:
    if self.tick.currentTick == 0:
      self.tick.tideSchedule = self.schedule

    if action.kind == 'spawn':
      self.spawn(action.position)
    elif action.kind == 'anchor':
      pass
    elif action.kind == 'dock':
      self.dock()
    elif action.kind == 'sail':
      self.sail(action.direction)
    else:
      raise NotImplementedError(action.kind)

    self.tick.currentTick += 1
    self.tick.tideSchedule.append(self.tick.tideSchedule.pop(0))
    if self.tick.currentTick >= self.tick.totalTicks:
      self.tick.isOver = True

  def spawn(self, position: Position) -> None:
    if self.tick.spawnLocation:
      print("[ERROR] Already spawned!")
      return
    self.tick.spawnLocation = position
    self.tick.currentLocation = position

  def dock(self) -> None:
    if self.tick.currentLocation not in self.tick.map.ports:
      print("[ERROR] No port there!")
      return
    dock_idx = self.tick.map.ports.index(self.tick.currentLocation)
    if dock_idx in self.tick.visitedPortIndices:
      if self.tick.visitedPortIndices[0] == dock_idx:
        self.tick.isOver = True
      else:
        print("[ERROR] Already visited this port!")
        return
    self.tick.visitedPortIndices.append(dock_idx)

  def sail(self, direction: str) -> None:
    assert direction in directions
    drow, dcol = 0, 0
    if 'W' in direction:
      if self.tick.currentLocation.column == 0:
        print("[ERROR] Already at the west-most point!")
        return
      dcol = -1
    elif 'E' in direction:
      if self.tick.currentLocation.column == self.ncols - 1:
        print("[ERROR] Already at the east-most point!")
        return
      dcol = 1
    if 'N' in direction:
      if self.tick.currentLocation.row == 0:
        print("[ERROR] Already at the north-most point!")
        return
      drow = -1
    elif 'S' in direction:
      if self.tick.currentLocation.row == self.nrows - 1:
        print("[ERROR] Already at the south-most point!")
        return
      drow = 1
    pos = Position(row=self.tick.currentLocation.row + drow,
                   column=self.tick.currentLocation.column + dcol)
    if not self.is_navigable(self.tick.currentLocation):
      print("[ERROR] Can't sail while on ground!")
      return
    if not self.is_navigable(pos):
      print("[ERROR] Can't sail to ground!")
      return
    self.tick.currentLocation = pos

  def height_at(self, pos: Position) -> int:
    return self.tick.map.topology[pos.row][pos.column]

  def current_tide(self) -> int:
    return self.tick.tideSchedule[0]

  def score(self) -> int:
    visits = self.tick.visitedPortIndices
    bonus = 1 if not visits or visits[0] != visits[-1] else 2
    base_score = (len(visits) * 125) - (self.tick.currentTick * 3)
    return base_score * bonus

  def is_navigable(self, pos: Position) -> bool:
    return self.height_at(pos) < self.current_tide()

  def show(self) -> None:
    class bcolors:
      HEADER = '\033[95m'
      OKBLUE = '\033[94m'
      OKCYAN = '\033[96m'
      OKGREEN = '\033[92m'
      WARNING = '\033[93m'
      FAIL = '\033[91m'
      ENDC = '\033[0m'
      BOLD = '\033[1m'
      UNDERLINE = '\033[4m'
    out = ''
    out += f'Tick {self.tick.currentTick} / {self.tick.totalTicks}\n\n'
    for row in range(self.nrows):
      for col in range(self.ncols):
        pos = Position(row=row, column=col)
        if pos == self.tick.currentLocation:
          out += bcolors.OKGREEN + 'X' + bcolors.ENDC
        elif not self.is_navigable(pos):
          out += '#'
        elif pos in self.tick.map.ports:
          idx = self.tick.map.ports.index(pos)
          visited = idx in self.tick.visitedPortIndices
          color = bcolors.OKCYAN if visited else bcolors.OKBLUE
          out += color + 'P' + bcolors.ENDC
        else:
          out += '.'
      out += '\n'
    out += '\n'
    print(out)



async def run():
  async with websockets.serve(handler, "", 8765):
    await asyncio.Future()


async def handler(websocket):
  slow = '--slow' in sys.argv
  fast = '--fast' in sys.argv
  game = None
  while not game or not game.tick.isOver:
    if slow:
      time.sleep(1)
    try:
      message = await websocket.recv()
    except websockets.exceptions.ConnectionClosed:
      print("Websocket was closed.")
      break
    data = json.loads(message)
    if data.get('type') == 'REGISTER':
      print("Started new game.")
      game = Game()
      if not fast: game.show()
      await websocket.send(json.dumps(game.tick.to_dict()))
    elif data.get('type') == 'COMMAND':
      if not game:
        print("Game not started.")
        continue
      data_action = data.get('action', {})
      kind = data_action.get('kind')
      if not kind:
        print("Missing action kind: ", data)
        continue
      if kind == 'sail':
        d = data_action.get('direction')
        if d not in directions:
          print('Invalid direction: ', d)
          continue
        action = Sail(direction=d)
      elif kind == 'spawn':
        p = data_action.get('position', {})
        p = Position.from_dict(p)
        action = Spawn(position=p)
      elif kind == 'anchor':
        action = Anchor()
      elif kind == 'dock':
        action = Dock()
      else:
        print('Unknown action kind: ', kind, data_action)
      if not fast: print(f"Action: {action}")
      game.apply(action)
      if not fast: game.show()
      if game.tick.isOver:
        print(f"Game is done! Score: {game.score()}")
      await websocket.send(json.dumps(game.tick.to_dict()))


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(run())
