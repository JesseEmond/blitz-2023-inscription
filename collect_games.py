import io
import json
import re
import requests
import sys
import time
import zipfile
from typing import Tuple

from game_message import Map, Position, Tick, TideLevels


assert len(sys.argv) > 2, f'Usage: {sys.argv[0]} <num_games> <output_folder>'
num_games = int(sys.argv[1])
output_folder = sys.argv[2]

with open('access_token.priv', 'r') as f:
  access_token = f.read().strip()


def start_game() -> int:
  r = requests.post(f'https://api.blitz.codes/practices/Inscription',
                    data='{}', cookies={'access_token': access_token})
  assert r.status_code // 100 == 2, (r.status_code, r.headers, r.content, r)
  data = json.loads(r.text)
  return data["id"]


def read_game_logs(game_id: int) -> str:
  r = requests.get(f'https://api.blitz.codes/game/{game_id}/debug',
                   cookies={'access_token': access_token})
  assert r.status_code == 200, r
  zip_buffer = io.BytesIO(r.content)
  with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
    return zip_file.read('game_logs.txt').decode('utf-8')


def extract_outer_object(line: str) -> Tuple[str, str]:
  first_bracket = line.index(' { ')
  last_bracket = len(line) - 1 - line[::-1].index('}')
  obj = line[:first_bracket]
  content = line[first_bracket + 3:last_bracket]
  return obj, content


def extract_game(logs: str) -> Tick:
  lines = [line.strip() for line in logs.split('\n')]
  start = next(i for i, line in enumerate(lines) if '--- TICK DUMP BEGIN ---' in line)
  end = next(i for i, line in enumerate(lines) if '--- TICK DUMP END ---' in line)
  assert end - start == 2, (start, end)
  line = lines[start + 1]
  line = line[line.index('] ') + 2:]  # Drop logging prefix
  tick_obj, tick = extract_outer_object(line)
  assert tick_obj == 'GameTick', tick_obj
  current_tick = int(re.search(r'current_tick: (\d+)', tick).group(1))
  total_ticks = int(re.search(r'total_ticks: (\d+)', tick).group(1))
  rows = int(re.search(r'rows: (\d+)', tick).group(1))
  columns = int(re.search(r'columns: (\d+)', tick).group(1))
  topology_str = re.search(r'topology: Topology\((\[\[.*?\]\])\)', tick).group(1)
  topo_lines = topology_str[2:-2].split('], [')
  topology = [[int(d) for d in topo.split(', ')] for topo in topo_lines]
  ports_str = re.search(r'ports: \[(.*?)\]', tick).group(1)
  ports = []
  for port_str in re.findall(r'Position \{.*?\}', ports_str):
    row = int(re.search(r'row: (\d+)', port_str).group(1))
    column = int(re.search(r'column: (\d+)', port_str).group(1))
    ports.append(Position(row=row, column=column))
  depth = int(re.search(r'depth: (\d+)', tick).group(1))
  tide_levels_str = re.search(r'TideLevels \{(.*?)\}', tick).group(1)
  tide_min = int(re.search(r'min: (\d+)', tide_levels_str).group(1))
  tide_max = int(re.search(r'max: (\d+)', tide_levels_str).group(1))
  tide_levels = TideLevels(min=tide_min, max=tide_max)
  assert re.search(r'current_location: (.*?),', tick).group(1) == 'None'
  current_location = None
  assert re.search(r'spawn_location: (.*?),', tick).group(1) == 'None'
  spawn_location = None
  assert re.search(r'visited_port_indices: (.*?),', tick).group(1) == '[]'
  visited_port_indices = []
  tide_schedule_str = re.search(r'tide_schedule: \[(.*?)\]', tick).group(1)
  tide_schedule = [int(t) for t in tide_schedule_str.split(', ')]
  assert re.search(r'is_over: (\w+)', tick).group(1) == 'false'
  is_over = False
  return Tick(
      currentTick=current_tick,
      totalTicks=total_ticks,
      map=Map(rows=rows, columns=columns, topology=topology, ports=ports,
              depth=depth, tideLevels=tide_levels),
      currentLocation=current_location,
      spawnLocation=spawn_location,
      visitedPortIndices=visited_port_indices,
      tideSchedule=tide_schedule,
      isOver=is_over)


def save_game(game: Tick, filename: str):
  with open(filename, 'w') as f:
    json.dump(game.to_dict(), f)


for i in range(num_games):
  print(f'Starting game #{i}...')
  game_id = start_game()
  print(f'  waiting for game #{game_id}...')
  time.sleep(10 * 60)
  print('  downloading game logs...')
  logs = read_game_logs(game_id)
  game = extract_game(logs)
  filename = f'{output_folder}/{game_id}.json'
  print(f'  saving to {filename}...')
  save_game(game, filename)
