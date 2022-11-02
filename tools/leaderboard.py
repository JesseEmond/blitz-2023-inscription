import dataclasses
import json
import requests
import tabulate
from typing import List, Mapping


with open('access_token.priv', 'r') as f:
  access_token = f.read().strip()

MY_TEAM_ID = 58


@dataclasses.dataclass
class Team:
  id: int
  name: str


@dataclasses.dataclass
class Game:
  score: int


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
  data = {
      'query': '{ blitz_scores_aggregate(order_by: {team_id: asc, score: desc} distinct_on: team_id) { nodes { score, team { id } } } }'
  }
  r = requests.post('https://api.blitz.codes/graphql', json=data,
                    cookies={'access_token': access_token})
  assert r.status_code == 200, (r.status_code, r.headers, r.content, r)
  data = json.loads(r.text)
  assert 'data' in data, data
  return {
      score['team']['id']: score['score']
      for score in data['data']['blitz_scores_aggregate']['nodes']
  }


def get_team_recent_games(team_id: int, num_games: int) -> List[Game]:
  query = """
  query RecentGames($teamId: smallint!, $limit: Int!) {
    blitz_scores_aggregate (
      where: {team_id: {_eq: $teamId}}
      order_by: {game_id: desc}
      limit: $limit) {
      nodes { score }
    }
  }"""
  data = {
      'operationName': 'RecentGames',
      'query': query,
      'variables': {'teamId': team_id, 'limit': num_games}
  }
  r = requests.post('https://api.blitz.codes/graphql', json=data,
                    cookies={'access_token': access_token})
  assert r.status_code == 200, (r.status_code, r.headers, r.content, r)
  data = json.loads(r.text)
  assert 'data' in data, data
  return [
      Game(score=game_data['score'])
      for game_data in data['data']['blitz_scores_aggregate']['nodes']
  ]


def show_game_stats(games: List[Game]) -> None:
  scores = list(sorted(game.score for game in games))
  # Note: we only look at the top ~20% games, since we skip games with <20 ports
  # (get a score of 0, which messes with stats) and expect roughly ~25% 20-ports
  # games in practice (from the possible 14, 16, 18, 20).
  num_top_scores = int(0.2 * len(scores))
  top_scores = scores[-num_top_scores:]
  print(f'~# 20-ports games: {len(top_scores)}')
  print(f'              Min: {min(top_scores)}')
  print(f'              Max: {max(top_scores)}')
  print(f'              Avg: {sum(top_scores)/len(top_scores):.1f}')
  print(f'        All games: {top_scores}')
  

teams = get_teams()
# It's useful to know how many games each team ran, given the random nature of
# the challenge. This can tell us if we have not played enough games, or if our
# approach is off.
task_counts = get_all_teams_task_counts()
team_scores = get_all_teams_scores()

ranked_teams = list(sorted(teams, key=lambda t: team_scores.get(t.id, 0),
                           reverse=True))
table = []
for team in ranked_teams:
  table.append([team.name, team_scores.get(team.id, 0), task_counts.get(team.id, 0)])
print(tabulate.tabulate(table, headers=['Team Name', 'Score', '# Runs'],
                        tablefmt='grid'))

print('\n' * 2)
num_games_stats = 200

my_team = next(team for team in teams if team.id == MY_TEAM_ID)
my_rank = ranked_teams.index(my_team) + 1
print(f'My team ID {my_team.id} "{my_team.name}" is at rank #{my_rank} on the '
      f'leaderboard, with a score of {team_scores[my_team.id]}')
print(f'Stats from my {num_games_stats} last games...')
my_recent_games = get_team_recent_games(my_team.id, num_games_stats)
show_game_stats(my_recent_games)

print('\n' * 2)

top_team = ranked_teams[0]
if top_team != my_team:
  print(f'The top team is team ID {top_team.id} "{top_team.name}", with a '
        f'score of {team_scores[top_team.id]}')
  print(f'Stats from their {num_games_stats} last games...')
  their_recent_games = get_team_recent_games(top_team.id, num_games_stats)
  show_game_stats(their_recent_games)
