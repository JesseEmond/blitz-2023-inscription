import dataclasses
import json
import requests
from typing import List, Mapping


with open('access_token.priv', 'r') as f:
  access_token = f.read().strip()

MY_TEAM_ID = 58


@dataclasses.dataclass
class Team:
  id: int
  name: str


class Game:
  score: int
  # TODO: time info


def get_teams() -> List[Team]:
  data = {
      'query': '{ blitz_teams { id, name } }',
  }
  r = requests.post('https://api.blitz.codes/graphql', json=data,
                    cookies={'access_token': access_token})
  assert r.status_code == 200, (r.status_code, r.headers, r.content, r)
  data = json.loads(r.text)
  assert 'data' in data, data
  return [Team(id=team['id'], name=team['name'])
          for team in data['data']['blitz_teams']]


def get_all_teams_task_counts() -> Mapping[int, int]:
  """Returns the count of tasks for each team_id."""
  data = {
      'query': '{ blitz_taskteam { team_id } }',
  }
  r = requests.post('https://api.blitz.codes/graphql', json=data,
                    cookies={'access_token': access_token})
  assert r.status_code == 200, (r.status_code, r.headers, r.content, r)
  data = json.loads(r.text)
  assert 'data' in data, data
  task_teamids = [task['team_id'] for task in data['data']['blitz_taskteam']]
  counts = {}
  for teamid in task_teamids:
    if teamid not in counts: counts[teamid] = 0
    counts[teamid] += 1
  return counts


def get_all_teams_scores() -> Mapping[int, int]:
  pass  # TODO


def get_team_recent_games(team_id: int, num_games: int) -> List[Game]:
  pass  # TODO


def show_game_stats(games: List[Game]) -> None:
  pass  # TODO
  

teams = get_teams()
# It's useful to know how many games each team ran, given the random nature of
# the challenge. This can tell us if we have not played enough games, or if our
# approach is off.
task_counts = get_all_teams_task_counts()
print("Team number of runs:")
for team in sorted(teams, key=lambda t: task_counts.get(t.id, 0), reverse=True):
  print(f'  {team.name}: {task_counts.get(team.id, 0)} runs')

# TODO get leaderboard scores, show recent games stats for top team & me
