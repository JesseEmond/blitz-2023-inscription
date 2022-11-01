import io
import json
import re
import requests
import sys
import time
import zipfile
from typing import Mapping, Optional, Tuple

from game_message import Map, Position, Tick, TideLevels


assert len(sys.argv) > 2, f'Usage: {sys.argv[0]} <num_games> <output_folder>'
num_games = int(sys.argv[1])
output_folder = sys.argv[2]

with open('access_token.priv', 'r') as f:
  access_token = f.read().strip()

MY_TEAM_ID = 58


def attempt_request(make_req_fn):
  r = make_req_fn()
  assert r.status_code != 401, "Update your access_token!"
  if r.status_code == 429:  # HTTP Too Many Requests
    print(f'Oops! We are being throttled: {r.headers}')
    retry = r.headers.get('Retry-After')
    if not retry:
      print(f'Huh, weird. Can\'t find a Retry-After header. Using 10mins to be safe.')
      retry = str(10 * 60)
    print(f'Waiting for {retry} seconds...')
    time.sleep(int(retry))
    r = make_req_fn()
  if not r.ok:
    print(f'Oops. Unexpected error...? Waiting 10 mins and retrying.')
    time.sleep(10 * 60)
    r = make_req_fn()
  assert r.ok, (r.status_code, r.headers, r.content, r)
  return r


def start_game() -> int:
  def req():
    return requests.post(f'https://api.blitz.codes/practices/Inscription',
                         data='{}', cookies={'access_token': access_token})
  r = attempt_request(req)
  data = json.loads(r.text)
  return data["id"]


def get_active_task_states() -> Mapping[int, str]:
  def req():
    query = """
    query getTasks($teamId: smallint) {
      blitz_tasks(
        where: {
          _and: {
            type: {_eq: "game"},
            taskteams: {team_id: {_eq: $teamId}}
          }
        }
        order_by: {id: desc}
        limit: 10) {
        id
        state
      }
    }"""
    data = {
        'operationName': 'getTasks',
        'query': query,
        'variables': {'teamId': MY_TEAM_ID},
    }
    return requests.post('https://api.blitz.codes/graphql', json=data,
                         cookies={'access_token': access_token})
  r = attempt_request(req)
  data = json.loads(r.text)
  assert 'data' in data, data
  return {task['id']: task['state'] for task in data['data']['blitz_tasks']}


def wait_for_game(game_id: int):
  SLEEP_TIME = 5
  started = False
  print(f'  waiting for game #{game_id} to start', end='', flush=True)
  while game_id not in get_active_task_states():
    print('.', end='', flush=True)
    time.sleep(SLEEP_TIME)
  print('  Started!')
  print(f'  waiting for game #{game_id} to complete', end='', flush=True)
  while get_active_task_states()[game_id] != 'completed':
    print('.', end='', flush=True)
    time.sleep(SLEEP_TIME)
  print('  Completed!')



def read_game_logs(game_id: int) -> str:
  def req():
    return requests.get(f'https://api.blitz.codes/game/{game_id}/debug',
                        cookies={'access_token': access_token})
  r = attempt_request(req)
  zip_buffer = io.BytesIO(r.content)
  with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
    return zip_file.read('game_logs.txt').decode('utf-8')


def extract_outer_object(line: str) -> Tuple[str, str]:
  first_bracket = line.index(' { ')
  last_bracket = len(line) - 1 - line[::-1].index('}')
  obj = line[:first_bracket]
  content = line[first_bracket + 3:last_bracket]
  return obj, content


def extract_game(logs: str) -> Optional[Tick]:
  lines = [line.strip() for line in logs.split('\n')]
  start = next(
      (i for i, line in enumerate(lines) if '--- TICK DUMP BEGIN ---' in line),
      None)
  if start is None:
    return None
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


def is_interesting_log_line(line: str) -> bool:
  return ('Graph was built in' in line or
          'greedy bot would get us' in line or
          'Solution found has a score' in line or
          'Colony solution was found in' in line or
          'TSP bot (held-karp)' in line or
          'TSP solution (held-karp)' in line or
          'Macro took ' in line)


def extract_interesting_lines(logs: str) -> str:
  lines = [line.strip() for line in logs.split('\n')]
  interesting = [line for line in lines if is_interesting_log_line(line)]
  return [line[line.index('] ') + 2:] for line in interesting]


def save_game(game: Tick, filename: str):
  with open(filename, 'w') as f:
    json.dump(game.to_dict(), f)


i = 0
while i < num_games:
  print(f'Starting game #{i+1}...')
  game_id = start_game()
  wait_for_game(game_id)
  print('  downloading game logs...')
  logs = read_game_logs(game_id)
  game = extract_game(logs)
  if not game:
    print(f'[!ERROR!] Failed to real logs for game {game_id}. Skipping.')
    continue
  print(f'  this was a {len(game.map.ports)} ports game')
  if len(game.map.ports) < 20:
    print('  skipping!')
    continue
  print('  noteworthy:')
  for line in extract_interesting_lines(logs):
    print(f'  - {line}')
  filename = f'{output_folder}/{game_id}.json'
  print(f'  saving to {filename}...')
  save_game(game, filename)
  i += 1