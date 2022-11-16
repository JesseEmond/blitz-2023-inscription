# Blitz 2023 Registration - /dev/null

TODO prev chals
TODO this chal

## Summary

TODO summarize
TODO challenge
TODO process
TODO results

## Write-Up

TODO write-up

### Challenge

TODO description, visual example

TODO ports distribution

TODO luck factor, score distribution for 20 ports

TODO example map that gives high score vs low score

### Greedy Solver

TODO go in a straight line

TODO A-star ignoring tides (assume top tide)

### Local Server

TODO

### Nearest Neighbor Solver

TODO

### Pathfinding With Tides

TODO how to modify A-star

### Building a Graph

TODO graph (visualization, too?)

TODO mention optimizations (see ablation section)

### ... And Now It's a TSP!

TODO define
TODO how it differs
TODO usual approaches

### Heuristic Solver: Ant System

TODO define / pseudocode
TODO example
TODO visuals
TODO tooling (visualization/sweep)
TODO variants, supported by hyperparams

### Exact Solver: Held-Karp
TODO 20 ports right at the brim of possible in <1s

#### Held-Karp
TODO algo explanation
TODO start vertex vs. our setup

#### Speeding it up
TODO profiling with xprof
TODO masks
TODO sets closer in memory
TODO multithreading

#### ... Ship It?
TODO didn't work
TODO my own AWS server

#### Redeeming this
TODO restrospect: spent too much time on this, should have switched to try
     adapting simplex-based approaches
TODO optimal solver
TODO live evaller

### Final Solver

TODO go back to ant, let it run, reswept on higher scoring map
TODO helper that runs in a loop w/ graphql, collects games, reauths everyday

TODO final score, TODO in how many runs

## Speed Optimizations Ablation

### Benchmarks

TODO which ones

### Pathfinding Optimizations
TODO

### Graph Optimizations
TODO

### Ant System Optimizations
TODO

### Held-Karp Optimizations
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
    - `live_evaler.rs`: Monitors a folder for new stored games, evaluates them
    with a slow but powerful solver to estimate an upper bound of possible score
    out of games seen so far.
  - `visualize_ants.py`: Plot visualizations of Ant Colony Optimization "ants".
