# Blitz 2023 Registration - /dev/null

TODO prev chals
TODO troll last year
TODO this chal

## Summary

TODO summarize
TODO challenge
TODO process
TODO results

## Write-Up

The following is a description of the challenge and how I approached it. For
code structure documentation, jump to the end.

### 🚩 Challenge

For this challenge, we are a salesperson that controls a boat and we want to
visit as many unique ports as possible, as fast as possible, to get the highest
score.

This is a tick-based game, where every tick our bot receives information about
the map and must send the action that our boat will take. We are told we have
1 second maximum per tick to return our action.

#### Details ℹ️
The **map** related information contains the following:

- Dimensions: width/height for the tiles of the map;
- Topology: each tile has a given height and, for a given tick, we can only
  navigate _to or from_ a tile if it is _below_ the water (see tide);
- Tide schedule: tells us how high the water currently is for the current tick,
  as well as what it will be like in the next few ticks;
- Ports: the list of port locations that we can visit to score points.

In practice, we see the following characteristics:
- Maps are 50x50 or 60x60, randomly generated;
- Tide schedules are of length 10;
- Tide schedules **cycle**, meaning that we have perfect game information on the
  first tick;
- We see a mix of 14, 16, 18, or 20 ports, with randomness;
- Games are max 400 ticks.

Note that this means that the navigable tiles are dynamic based on the tide and
that shortest paths can depend on the offset within the tide schedule.

We must give our **action**, out of the following:
- Sail: move in any of the 8 directions (up/down/left/right + diagonals), with
  diagonals having the _same cost_ as horizontal/vertical movements;
- Spawn: only do this once at the start;
- Dock: must be done after reaching a port to count it as visited;
- Anchor: wait and do nothing.

Visually, the game looks like this:

https://user-images.githubusercontent.com/1843555/202345642-41338ea5-d4bc-4326-8b3d-16e859bdec63.mp4

#### Scoring 🧮
The game ends if we dock the first port again (do a full tour) or if 400 ticks
are reached. Then, the score is computed based on the following formula:

```
bonus = 2 if ports_visited[0] == ports_visited[-1] else 1
base = len(ports_visited) * 125 - 3 * total_ticks
score = base * bonus
```

In simpler terms: visit many ports quickly, try to loop back home.

The example above visited 19 ports in 274 ticks, doing a full tour. This gave a
score of 3106 points:

```
bonus = 2  # full loop
base = 19 * 125 - 3 * 274 = 1553
score = 1553 * 2 = 3106
```

Based on the scoring formula, we see that:
- Looping is almost always desired (doubles score + ends game early). Hard to
  imagine cases where visiting `n` ports and forcing 400 ticks total without a
  loop would be better than looping back after `n/2` ports and being able to end
  early;
- Higher number of ports can score us more points. While theoretically games of
  lower ports can match games of higher ones (e.g. 3896 pts can be achieved with
  a 20 ports loop in 184 ticks, the same can be obtained if it would be possible
  to get a 17 ports map, in a 17 ports loop in 59 ticks -- which would be an
  extremely lucky & packed map), our best bet is with a 20-ports game.

#### Roll the Dice 🎲
The randomness of some of the parameters (map generation, tide, # ports) leads
to the optimal possible score on a game depending on luck, and can vary quite a
bit between games.

Here are two games side-by-side with 20 ports that lead to very different
optimal scores (3422 points in 263 ticks vs. 3734 points in 211 ticks), even
though they both visit 20 ports (the optimal solver will be described later):

https://user-images.githubusercontent.com/1843555/202344987-9ca0679c-5819-477a-9ae1-03f6531ca394.mp4

To give an idea of the range, here is the distribution of optimal scores for 100
20-port games assigned to us by the server:

![optimal score distribution](https://user-images.githubusercontent.com/1843555/202337764-ce9662e0-ff0a-424e-9349-cf21eefa25da.png)

Note that this is showing the optimal score, too (best case scenario), and that
we have a 1-second time limit to reply to each tick, which might not allow us to
run an optimal solver. However, this does show that there's a good amount of
variability across games and if we want to get a high score on the leaderboard,
we'll have to roll the dice and rerun games for a chance at a high score towards
the right of this distribution.

### 🤑 Greedy Solver

I started off in Python, and with a simple bot: start on the first port in the
list, then go to the next one in a straight line, disregarding tides. This
allowed me to get used to the challenge & its API, get a score on the board
early so Marc doesn't pull my leg too much, and to design a bot API with:
- "micro" control with a simple state machine that can follow a fixed path by
sending `Sail` actions with the right directions, and `Dock` once the goal is
reached;
- "macro" control to pick goals and give paths (for now visit ports in the
arbitrary input order and generate straight line paths).

As you can imagine, there's no guarantee that a straight line works depending
on the topology, so this would often give a very negative score from getting
to 400 ticks with almost no ports visited. I even got a ping from Andy saying
"I didn't know it went this far into the negatives" after a -575 points game,
shortly after a -950 points one.

Next, I implemented `A*` starting off from the [fantastic pathfinding resource
at Red Blob Games](https://www.redblobgames.com/pathfinding/a-star/implementation.html)
(newer version of the `Amit's A* pages`). For the heuristic, I used the
[Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance) (which
is essentially `max(deltax, deltay)`) to account for diagonal movements of
cost 1. My pathfinding implementation for now ignored tide changes, assuming
that the tide was always at its lowest (most restrictive), meaning that all
tiles that are navigable at that point are _always_ navigable. This restricts
our truly optimal pathfinding opportunities, but gives us correct paths in
the positive points range!

### 🤖 Local Server

Making changes, uploading the code to the server, waiting for a game to run,
then downloading the logs to iterate is quite a long feedback loop to find
trivial bugs. I pivoted to implementing my own version of the server that can
run locally, from game data that I had logged in my previous games on the
server.

This is something that really would have made more sense to do as a very
first step, but hey, I wanted _something_ on the leaderboard
:slightly_smiling_face:.

I made sure that my bot running against my local server on historical games
gave the same score as it did on the server, and it took some iterations to
iron out the exact ordering of operations executed on the server for
movements and tide updates. Eventually though, I ended with a much faster
feedback loop.

### 🧭 Nearest Neighbor Solver

TODO

### 🌊 Pathfinding With Tides

TODO how to modify A-star

### 🕸️ Building a Graph

TODO graph (visualization, too?)

TODO mention optimizations (see ablation section)

### 🕴️ ... And Now It's a TSP!

TODO define
TODO how it differs
TODO usual approaches

### 🐜 Heuristic Solver: Ant System

TODO define / pseudocode
TODO example
TODO visuals
TODO tooling (visualization/sweep)
TODO variants, supported by hyperparams

### ✍️ Exact Solver: Held-Karp
TODO 20 ports right at the brim of possible in <1s

#### Held-Karp 📋
TODO algo explanation
TODO start vertex vs. our setup

#### Speeding it up ⏩
TODO profiling with xprof
TODO masks
TODO sets closer in memory
TODO multithreading

#### ... Ship It? 🚢
TODO didn't work
TODO my own AWS server

#### Redeeming this 🩹
TODO restrospect: spent too much time on this, should have switched to try
     adapting simplex-based approaches
TODO optimal solver
TODO live evaller

### 🦾 Final Solver

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
