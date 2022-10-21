#!/usr/bin/env python

import asyncio
import glob
import json
import os
import sys
import subprocess
import time
import websockets
from typing import List

from game_message import Action, Anchor, Dock, Position, Sail, Spawn, Tick, directions
import seen_games


class Game:
  def __init__(self, tick: Tick):
    # Hacky copy
    tick = Tick.from_dict(json.loads(json.dumps(tick.to_dict())))
    self.tick = tick
    self.nrows = len(self.tick.map.topology)
    self.ncols = len(self.tick.map.topology[0])

  def apply(self, action: Action) -> None:
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

    if not self.tick.isOver:
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
    back_home = visits and visits[0] == visits[-1]
    bonus = 2 if back_home else 1
    num_visits = len(visits) - 1 if back_home else len(visits)
    base_score = (num_visits * 125) - (self.tick.currentTick * 3)
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


def show_scores(scores: List[int]):
  print(f'All scores: {list(sorted(scores))}')
  print(f'Min: {min(scores)}')
  print(f'Max: {max(scores)}')
  print(f'Average: {sum(scores) / len(scores):.1f}')


is_eval = '--eval' in sys.argv
slow = '--slow' in sys.argv
fast = '--fast' in sys.argv or is_eval

all_games = []
game_index = None
game_scores = []
client_process = None


def start_game():
  global client_process
  if client_process:
    client_process.kill()
    waited = False
    printed = False
    while client_process.poll() is None:
      if waited:
        print('.', end='', flush=True)
        printed = True
      time.sleep(0.2)
      waited = True
    if printed: print(' Ready!')
  env = os.environ.copy()
  env['RUST_LOG'] = 'warn'
  client_process = subprocess.Popen(['./rust/target/release/application'], env=env)


async def run():
  global game_index
  if is_eval:
    for path in glob.glob('games/*.json'):
      with open(path, 'r') as f:
        all_games.append(Tick.from_dict(json.load(f)))
    game_index = 0
    print(f'{len(all_games)} games in our dev set')

  async with websockets.serve(handler, "", 8765):
    if is_eval:
      start_game()
    await asyncio.Future()



async def handler(websocket):
  global game_index
  game = None
  while not game or not game.tick.isOver:
    reset_game = False
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
      if game_index is None:
        game = Game(seen_games.GAME1)
      else:
        game = Game(all_games[game_index])
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
        score = game.score()
        print(f"Game is done! Score: {score}")
        if is_eval:
          game_index += 1
          game_scores.append(score)
          if game_index == len(all_games):
            show_stats(game_scores)
            exit(0)
          reset_game = True
      await websocket.send(json.dumps(game.tick.to_dict()))
      if reset_game:
        await websocket.close()
        start_game()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(run())
