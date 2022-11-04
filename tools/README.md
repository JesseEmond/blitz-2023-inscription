# Tools

Tooling related to the challenge: local server, online game automation, solution
debugging/tuning.

## Local Server
[`server.py`](server.py) is a local server challenge implementation, used to
play one game while working on a bot, or multiple games to fully eval a bot
or sweep hyperparameters.

By default, it will play some fixed game, with ASCII visualization of the game.

TODO: screencap of a game visualization?

### Options
- `python server.py --fast`: Hide the ASCII visualization, for faster games;
- `python server.py --slow`: Add a sleep between ticks, to inspect a game;
- `python server.py --gameid=5202`: Replay a saved game from `../games/`;
- `python server.py --eval`: Run all saved games from `../games/` (note: only
  20-ports games) to get min, max, average scores;
- `python server.py --sweep`: Sweeps hyperparameters for Ant Colony Optimization
  (e.g. number of ants, iterations, evaporation rate, etc.), using
  [Vizier](https://github.com/google/vizier) and evaluating on all available
  games in `../games`. For each Vizier suggestion, hyperparameters are written
  to disk so that the launched bot picks them up.
  - Can use `python server.py --sweep --gameid=5202` to sweep on a single game.

## Online Server Automation / Games Collection
[`collect_games.py`](collect_games.py) automatically launches games on the
Blitz server, waits for it to complete, downloads & parses the game logs, and
saves them to the given destination.

It's used both to retry games in a loop (for a chance to get a 20-ports game
with high score potential) and collect games for offline evals.

`access_token.priv` must exist and contain your access token, which you can
obtain by looking at HTTP Headers when logged in on the Blitz website (e.g. see
Network tab in Chrome).

Example usage: `python collect_games.py 10 ../games/`

Note that it treats the number as the desired number of 20-ports games and it
does not save <20 ports games.

### Live Evaler

TODO document, uniquely a rust binary

## Visualize Ants
[`visualize_ants.py`](visualize_ants.py) parses debug logs from a run with Ant
Colony Optimization to produce visualizations on how the algorithm behaves at
each iteration.

It shows the local & global best ant paths, the ant paths taken
in aggregate during an iteration, the pheromone trails, the heuristic values,
and the sampling weights.

Note that the debug logs parsing is very very crude, and it would be much better
if the bot wrote the information in e.g. JSON and it got parsed here instead.

TODO: produce gif of iterations and include here?

## Sweep Analysis
[`analyze_sweep.py`](analyze_sweep.py) analyzes logs from a local server sweep
run and plots, for each hyperparameter, how the scores behave in the observed
range of the parameter. This is used to help pick sweep ranges.

Logs from sweeping (needed for `analyze_sweep.py`) can be collected for example
with:

```sh
python -u server.py --sweep | tee /tmp/sweep_logs.txt
```

TODO: include sweep param example screenshot that suggested a narrower range

## Leaderboard Analysis
[`leaderboard.py`](leaderboard.py) interacts with the server's graphql API to
inspect some available information on other teams.

Given that this challenge involves considerable randomness (number of ports, map
generation specifics), it can be hard to tell when we're behind due to a lack of
games played (for a chance for a high potential game), or other issues.
Comparing to other teams on the leaderboard is a helpful proxy.

This shows information about the total number of games played & score per team,
along with some last N games stats for my team & the current top team.

## Other Minor Utils
- [`seen_games.py`](seen_games.py): contains a hard-coded `GameTick` from a
  server game, used by default on the local server.
- [`hyperparams.py`](hyperparams.py): data structure for hyperparameters for
  Ant Colony Optimization. Written to disk when sweeping. Must match the
  expected format by the matching struct in the Rust code.
- [`deduce_score.py`](deduce_score.py): from a given score, deduce the possible
  breakdowns of visited ports vs. ticks.
- [`game_message.py`](game_message.py): from the Python bot starterkit.
