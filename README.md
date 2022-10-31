# Blitz 2023 Registration - /dev/null

TODO desc

## Summary

TODO summarize

## Write-Up

TODO write-up

## Speed Optimizations Ablation

TODO

## Code Overview

TODO rust overview
TODO benchmark overview
- `games/`: folder of games collected while running on the Blitz server. Used to
  reproduce games locally for evaluation/sweeping purposes.
- `tools/`: folder for tools to interact/iterate on the challenge. Here are the
  noteworthy ones (see [`tools/README.md`](tools/README.md) for more info):
  - `server.py`: Local server implementation, used to test the bot locally with
    real games, with ASCII visualization. Has options to do a full eval on all
    saved games, or run a hyperparameter sweep;
  - `collect_games.py`: Launch games on the server and parse their logs to
    extract games and save them locally. Also used to play games in a loop for a
    chance for a better score;
  - `visualize_ants.py`: Plot visualizations of Ant Colony Optimization "ants".
