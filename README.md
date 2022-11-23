# Blitz 2023 Registration - /dev/null

Two years ago, we were [hardcoding an HTTP server in C++ and reading/writing ints
_fast_](https://github.com/JesseEmond/blitz-2021-chal). Last year, we were
[packing tetrominoes](https://github.com/JesseEmond/blitz-2022-inscription) with
the added fun of msanfacon@ pretending he was participating and beating us on the
leaderboard (spoiler alert -- he was changing his score in the database).

For this year's [Coveo Blitz](https://2023.blitz.codes/) registration challenge,
the [NP-hard](https://en.wikipedia.org/wiki/NP-hardness) problem of choice was
the [Traveling Salesperson Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem)!

## Summary

Our task was to write a bot that controls a boat that goes from port to port to
sell things. It is a tick-based game where we control the individual movements
of our boat, with the added navigation challenge that the tides change, making
the navigability of tiles change through time (and so also the shortest paths
between our ports!) We get points for visiting more ports, a big bonus for
looping back to our origin port, and lose points the longer we take.

Visiting as many (all?) ports (cities?), as fast as possible, returning to the
first one, selling things...
[Isn't there a movie about this?](https://en.wikipedia.org/wiki/Travelling_Salesman_(2012_film))

Here are the high level steps of how I approached this year's Blitz:
- Started in Python, with a greedy bot that tries to go in a straight line
  to the next port in a random order, with no pathfinding, ignoring the map
  & tides, while getting used to the challenge;
- This often would just get stuck, unable to move, and give _very_ negative
  scores:
  > I didn't know it went this far into the negatives
  >
  > -- Andy de Montréal
- Implemented pathfinding, ignoring tides for now, to get _some_ positive
  score;
- Wrote a local version of the server to iterate faster;
- Changed the bot to pick the nearest neighbor unseen port, also picking
  the best starting port by simulating that strategy for each one;
- Modified my pathfinding to take the tide schedule into account to unlock
  cool shortcuts.

That gave a decent score on the leaderboard. It's around that time that I
noticed the team `Roach` quickly matching and passing my scores, and I have
to admit that with
[Marc's trolling from last year](https://github.com/JesseEmond/blitz-2022-inscription),
I had _multiple_ moments of paranoia where I was questioning whether this was
Marc playing in the database once more... :) But he didn't do that this year!

I continued working on the bot:
- Rewrote it in Rust;
- Changed my pathfinding to find the shortest path from one port to _all_
  other ports, for _every_ possible tick offset in the tide schedule;
- Used this to represent the problem as a graph instead, to treat this more
  as a traditional Traveling Salesperson Problem;
- Implemented solvers that solve the _TSP_ exactly, and also heuristic
  approaches based on... ants!?
- Ran a _lot_ of games to try and get a luckier map to score higher points.
  
I ended in first place on the leaderboard with a score of **3896** points,
very closely followed by `Roach` with 3884 points, which is a mere 2 ticks
difference in our respective highest scoring game -- effectively a few
rolls of the dice apart.

I learned a lot about solving TSPs and effective optimizations to apply
to the algorithms I used, and the following write-up documents that.

## Write-Up

_For code structure documentation, jump to the end._

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
  navigate _to or from_ a tile if it is _below_ the water (see tide schedule);
- Tide schedule: tells us how high the water currently is for the current tick,
  as well as what it will be like in the next few ticks;
- Ports: the list of the port locations that we can visit to score points.

In practice, we see the following characteristics:
- Maps are 60x60 (later also 50x50 for smaller ports games), randomly generated;
- Tide schedules are of length 10 always;
- Tide schedules **cycle**, meaning that we have perfect game information on the
  first tick and can plan our entire game strategy right then;
- We see a mix of 14, 16, 18, or 20 ports, with randomness;
- Games are max 400 ticks.

Note that this means that the navigable tiles are dynamic based on the current
tide and so shortest paths can depend on the offset within the tide schedule we
are at.

We must give our **action** for each tick, out of the following:
- Sail: move in any of the 8 directions (up/down/left/right + diagonals), with
  diagonals having the _same cost_ as horizontal/vertical movements;
- Spawn: only do this once to pick where we start;
- Dock: must be done after reaching a port to count it as visited;
- Anchor: wait on the tile and do nothing.

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
  loop would be better than looping back after only `n/2` ports and being able
  to end early;
- Higher number of ports can score us more points. While theoretically games of
  lower ports can match games of higher ones (e.g. 3896 pts can be achieved with
  a 20 ports loop in 184 ticks, the same can be obtained in a 17 ports loop in
  59 ticks -- which would be an extremely lucky & packed map), our best bet is
  with a 20-ports game.

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
our truly optimal pathfinding opportunities, but gives us correct paths to
score points in the positive range!

### 🤖 Local Server

Making changes, uploading the code to the server, waiting for a game to run,
then downloading the logs to iterate is quite a long feedback loop to find
trivial bugs. I pivoted to implementing my own version of the server that can
run locally, from game data that I had logged in my previous games on the
server.

This is something that really would have made more sense to do as a very
first step, but hey, I wanted _something_ on the
leaderboard. :slightly_smiling_face:

I made sure that my bot running against my local server on historical games
gave the same score as it did on the server, and it took some iterations to
iron out the exact ordering of operations executed on the server for
movements and tide updates. Eventually though, I ended with a much faster
feedback loop.

### 🧭 Nearest Neighbor Solver

Following the input port list blindly (spawn on the first one, go to the
next, etc.) is not a good strategy -- we end up needlessly taking very
long paths when we could visit some ports along the way.

Instead, we can prioritize the next unvisited port with the shortest
distance. On the
[Traveling Salesperson Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem),
this is essentially the
[nearest neighbour algorithm](https://en.wikipedia.org/wiki/Nearest_neighbour_algorithm).

To find the shortest path to _any_ goals (the closest unvisited port),
I changed my `A*` implementation to early exit if any position in a
list of goals is found, instead of a single position.

Additionally, since our ports might not all be reachable from each other
(since we're doing `A*` assuming the lowest possible tide), we also
simulate this strategy for each possible starting point, to pick which
spawning position would give the best score. This idea is also
beneficial if we take tides into account -- some starting positions
might be better than others because they start us at a favorable tide
in the schedule.

As part of this planning/simulation, I also consider the option of
"going home" (back to the first port) early in the search (vs.
continuing to the rest of the ports), since it can sometimes be best to
end the game earlier vs. visit more ports for the amount of ticks
needed to get there.

### 🌊 Pathfinding With Tides

Assuming the lowest tide is fairly restrictive, we might instead
unlock shortcuts when the tide is high if we time our movements right.

To adjust pathfinding to take into account tides, we can change
the neighbor generation in `A*` to dynamically lookup tiles that
are navigable based on the tide for the current tick. This current
tide value can be found by looking up
`tide_schedule[tick % len(tide_schedule)]` (since it cycles), and by
using the `A*` `g` score as the `tick` value (the cost so far).

Then, we also want to allow our boat to "wait" for a tick (send
an `anchor` action) in case we can take advantage of a
tide change in the next few ticks. To do so, we change our `A*`
like this:
- The state we push in the priority queue is no longer just a
  position on the grid, it is now a tuple `(position, wait)`,
  where "wait" is the amount of ticks we have waited for so far.
- When adding `A*` neighbors:
  - Add horizontal/vertical/diagonal move actions as
    `(new_position, 0)` with cost 1;
  - Add "wait" actions as `(position, wait+1)` with cost 1;
  - Only add "wait" sometimes: do not bother waiting more than
    `len(tide_schedule)` ticks, there's no point since it cycles
    after that;
  - Only add "move" sometimes: do not consider move actions if
    we are on an unnavigable tile (e.g. we waited and the tide
    went down), we are not allowed to move then.

With this, we start getting a boat that moves efficiently
between ports and takes some pretty cool shortcuts!

https://user-images.githubusercontent.com/1843555/202583986-708dc8a8-f441-420f-810f-d34f33e25ef9.mp4

_Note that diagonal movements of cost 1 are visually
unintuitive -- it often takes paths that look slower, but are
equivalent to going in straight lines. I didn't do anything to
avoid
[ugly paths](https://www.redblobgames.com/pathfinding/a-star/implementation.html#troubleshooting-ugly-path)._

### 🕸️ Building a Graph

When we get the game information on the first tick, we can
precompute all of the following:
- For each port, shortest path to all other ports;
- Do that for each possible tick offset in the tide schedule.

The reason for doing so is that we can then create a graph
from our ports:
- Each port is a **vertex**;
- Each port connects to other ports via a precomputed shortest
  path -- that's an **edge**;
- The **cost** of this edge (and the exact path details)
  depends on the tick offset we are at on the source vertex.

But this gets expensive to build: 10 tick offsets, 20
vertices, 380 edges (fully connected directed graph), that's
3800 total shortest paths if we do them pairwise. If we
want to fit this in 1 second and do processing on that
graph afterwards, we need to speed things up:
- I rewrote the bot in Rust;
- I changed my `A*` implementation to support finding the
  shortest paths to _all_ targets at once, instead of the
  closest one:
  - Switch the heuristic to the smallest
    [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance)
    in the list of remaining ports;
  - When a goal is found, we remove it from the remainder
    list and reprioritize the priority queue with updated
    heuristic values;
- I did a couple more optimizations, outlined in the
  `Speed Optimization Breakdown` section.

But we end up with a graph representation of our problem:

![Graph](https://user-images.githubusercontent.com/1843555/202889263-24fd69af-5ed8-4d00-923e-91d908343e32.png)

Around this time, I added a "give up" mode to my bot that
quickly docks the initial port twice to get 0 points and
early-exit. Because this graph processing is getting a bit
compute-heavy and we are mostly interested in 20-ports
games to get a higher score on the leaderboard, we can
save time (and minimize our impact on Coveo infrastructure
when possible) by just skipping games that are < 20 ports.


### 🕴️ ... And Now It's a TSP!

Now we have a graph and we're effectively trying to find
the shortest possible tour that visits each "city" (port)
once, returning to the origin. That's a
[Traveling Salesperson Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem)
(TSP)! A famously NP-hard problem.

One distinction, however, is that our edge costs are
dynamic, depending on the cost of our tour _so far_
(changes the tick offset we'd be at, and thus the
shortest path & cost to other ports). This means that
traditional approaches to solving TSPs have to be
adapted.

One example of this is that TSP
solutions often assume that we are starting on the
first node without loss of generality -- it doesn't
matter what the origin node is. For us, however, this
impacts tide offsets thorought the tour, which can
make a spawning node better than others.

We can still explore TSP approaches, though: we can
use a heuristic solver that can give suboptimal
tours in reasonable time, or consider exact solvers
that find the optimal tour in exponential algorithmic
time.

Our bot's architecture now becomes:
- On the first tick, a **solver** plans a solution to
  the game (after building the graph);
- A **macro** struct executes the plan by keeping
  track of the current step in the solution,
  adjusting the micro state machine to lookup and
  provide exact paths to follow;
- A **micro** struct produces actions based on the
  state we are in.

### 🐜 Heuristic Solver: Ant Colony Optimization

I thought this would be a very good opportunity to
try a fun algorithm I recently learned about in one
of
[Sebastian Lague's excellent videos](https://www.youtube.com/watch?v=X-iSQQgOd1A)
(amazing channel -- highly recommend it): `Ant Colony
Optimization`
([widipedia](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms)).
The idea is to simulate ants that leave _pheromone
trails_ on paths they visit. Each ant samples its
actions through a mix of heuristics and pheromones
left by other ants. Repeat for multiple ants, for
multiple iterations, and leave pheromone trails for
better solutions to encourage exploitation of "good"
edges.

At a high level, the ant colony optimization (ACO)
algorithm looks like this:

```python
def ACO():
  for _ in range(ITERATIONS):
    ants = ConstructSolutions()  # Simulate ants
    LocalSearch(ants)  # Optionally locally improve solutions
    UpdateTrails(ants)
  return best_ant
```

For us, this looks something like this (this is a bit like
pseudo-code, assuming some global variables exist for
simplicity):

```python
def ConstructSolutions():
  [SimulateAnt() for _ in range(ANTS)]
  
def SimulateAnt():
  spawn = RandomChoice(ports)
  ant = Ant(spawn)
  unvisited = set(ports) - set([start])
  while unvisited:
    choice = SampleChoice(ant, unvisited)
    unvisited.remove(choice)
    ant.Visit(choice)
  return ant
  
def SampleChoice(ant, options):
  weights = []
  for option in options:
    pheromone = pheromones[ant.position][option]
    heuristic = 1 / graph.cost(ant.position, ant.current_tick, option)
    weight = pow(pheromone, ALPHA) * pow(heuristic, BETA)
    weights.append(weight)
  return Sample(weights)
  
def LocalSearch(ants):
  for ant in ants:
    ant.GoHomeEarlyIfBetter()  # end tour earlier if gives a better score
    
def UpdateTrails(ants):
  for source in ports:
    for dest in ports:
      pheromones[source][dest] *= (1 - EVAPORATION_RATE)  # evporate pheromones
  for ant in ants:
    pheromone_add = 1 / ant.tick
    for source, dest in ant.path:
      pheromones[source][dest] += pheromone_add
```

From there, a lot of variations are possible. I found the
following links super useful when trying ideas:
[[1]](https://staff.washington.edu/paymana/swarm/stutzle99-eaecs.pdf)
[[2]](http://www.scholarpedia.org/article/Ant_colony_optimization)
[[3]](https://www.researchgate.net/publication/277284831_MAX-MIN_ant_system).

Some noteworthy variants:
- **Ant System** (AS): as shown above;
- **Ant Colony System** (ACS):
  - `SampleChoice`: with some probability, greedily pick the highest
    weight option (exploration vs. exploitation);
  - `UpdateTrails`: only add pheromones for the global best ant seen
    so far, with the evaporation multiplier applied to the pheromone
    add, too;
  - `SimulateAnt`: when visiting a port, apply a local pheromone
    trail update to make it less desirable _during_ iteration before
    next ants are simulated (promote exploration).
- **MAX-MIN Ant System** (MMAS):
  - `UpdateTrails`: like _ACS_, only the global best ant adds
    pheromones;
  - `UpdateTrails`: pheromones are clamped to a min/max to avoid
    search stagnation;
  - Pheromones are initially set to the max to promote early exploration.
- And many others

To support a range of variants, I implemented _ACO_ with multiple
hyperparameters that control behavior:
- _iterations_: total number of ant simulations for the algorithm;
- _ants_: number of ants simulated per iteration;
- _evaporation_rate_: `EVAPORATION_RATE` above -- used when
  evaporation pheromones, and used as a multiplier for added
  pheromones;
- _exploitation_probability_: probability of taking the max-weight
  option instead of sampling;
- **not** _alpha_: power for the pheromone -- I removed this from
  computations to speed up processing (see the `Speed Optimization
  Breakdown` section), which is not strictly equivalent to having it
  if we resweep _beta_, but this gave me decent results while being
  faster (allowing more iterations/ants).
- _beta_: power for the heuristic when computing sampling weights;
- _local_evaporation_rate_: used in local updates (from _ACO_
  definition) to disincentivize other ants from the same iteration
  to promote exploration;
- _min_pheromones_: min value that pheromones can have, from _MMAS_.
- _max_pheromones_: max value that pheromones can have, from _MMAS_.
- _pheromone_init_ratio_: from min to max pheromones, what value
  should we initialize pheromones with (0 = min, 1 = max).
  
The idea was to make it so that the different variants were
available as different points in the hyperparameter space:
- **AS**: `exploitation_probability=0, local_evaporation_rate=0,
           min_pheromones=0, max_pheromones=high_value,
           pheromone_init_ratio=low_value`
- **ACS**: `min_pheromones=0, max_pheromones=high_value,
            pheromone_init_ratio=low_value`
- **MMAS**: `exploitation_probability=0, local_evaporation_rate=0,
             pheromone_init_ratio=1`

I implemented this and the results were somewhat reasonable, but
the behavior felt very opaque -- when it's doing poorly, is it
because of a bug, or because of bad hyperparameters? So I spent
some time on visualization tools to understand what the ants
behavior was like. Here is what is looks like on an example game,
with some hyperparameters (note: not the exact same ones I ended
up using):

![AntColonyOptimization](https://user-images.githubusercontent.com/1843555/202929678-0232e18e-fa16-4965-b585-666b990c23d7.gif)

What is displayed:
- The _top left_ shows the start & path of the best ants, both locally
  (current iteration, shown in _blue_) and globally (seen so far,
  shown in _green_ and drawn on top of everything);
  - We see that the local best ant tends to not deviate much from
    the global best, maybe this set of hyperparams would benefit
    from more exploration.
- The _middle left_ shows the heuristics (`1/distance`) between
  nodes, this ideologically represents the "smell" of an ant of its
  neighboring options (closer = better heuristic). Because we have
  10 tick offsets and thus 10 path options for each edge, here I'm
  just showing the max heuristic value of possible tide offsets per
  node.
- The _bottom left_ shows what paths the ants in the iteration took,
  with thicker lines representing more frequented edges. We see that
  the exploration is high early on, but then shifts to exploitation
  of the best path found so far;
  - Again, maybe those hyperparameters would benefit from more
    exploration.
- The _top right_ shows the pheromone trails that are left by the
  best ant, that slowly evaporate with time.
- The _two bottom right_ graphs show the combined heuristic and
  pheromone values turned to weights, which are directly used to
  make a probability distribution when sampling moves.

Next, to pick a good set of hyperparameters, I implemented a
couple of extra tools:
- A `collect_games.py` tool that runs games on the server
  in a loop (note: also useful when we're trying to get lucky
  with a high-potential game!), downloads the logs and saves
  the games to disk;
- Extended `server.py` to support an "eval" mode that goes
  through all locally stored games and runs our bots to give
  us some stats (min/max/avg) on our how our bot is doing on
  a range of games.
  
With this, I could then add support for hyperparameter
sweeping plugged in the `server.py` eval logic. I
used [vizier](https://github.com/google/vizier) which I have
some familiarity with. We can define hyperparameter ranges,
pick an optimizer, and let it explore the hyperparameter
space. In the end, I stuck to random search, but this is
easy to change and extend with more advanced search algorithms.

To help out when picking parameter ranges, I also made
a tool to show rudimentary plots of scores obtained vs.
parameter values, to spot ranges that are too narrow or
too wide:

![SweepAnalysis](https://user-images.githubusercontent.com/1843555/202930568-c71cd220-7a3e-4680-8f90-7de2a7799a2a.png)

This didn't end up being super useful, but as you can
see in the image above, small values of `beta` were
hurting, so I was able to narrow that sweep range at
least.

I could have added more tuning here -- there are other
ant system variants or settings that can be useful (e.g.
a schedule between picking the iteration local best ant
vs. global best one) and I made some arbitrary decisions
(e.g. assign pheromone trails to directed edges, I didn't
try undirected ones or maybe instead more granular per-tick-offset ones).

Instead, I got distracted...

### ✍️ Exact Solver: Held-Karp

A 20 "city" TSP is not _that_ big. Even with an exponential
time exact TSP algorithm like
[Held-Karp](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm)
that runs in `O(2^n * n^2)`, that's within a constant factor of
`2^20 * 20^2 ~= 419M`, which is definitely tractable, and
maybe... even doable within 1 second?

#### Held-Karp 📋

A naive approach to solving the TSP would be to consider every
possible permutation of vertices; `n!` of them. For 20 cities,
that's `2,432,902,008,176,640,000` permutations -- a bit much.
Instead, _Held-Karp_ formulates this as a dynamic programming
problem: we solve easier versions of the problem to help us
solver harder versions of it, re-using information along the way.

On a normal graph (ignoring the specifics of the Blitz for now),
_Held-Karp_ defines the problem with a helper `g(S, e)` function.
This function tells us "starting on vertex `1`, going through
all ports of set `S` in some order and _ending_ on vertex `e`, what
is the optimal cost?". It turns out that if we have `g(S', e)`
computed for every possible set `S'` with one less element than
`S`, we can compute `g(S, e)`. Here's how we go about it, by
defining `g(S, e)` recursively:
- `g({k}, k)` is trivially `c(1, k)` -- the only path that starts
  at vertex `1`, goes through all vertices in `{k}` and ends in
  `k`, well is the `1->k` edge of cost `c(1, k)`;
- `g(S, k)` can be determined by looking at every vertex `m` in
  `S\{k}` (`S` without `k`), and picking the one that has the
  lowest value for `g(S\{k}, m) + c(m, k)`. In plain words: of
  this set `S` of nodes (without our target end `k` node), which
  one would be the best second-to-last node to take to then reach
  `k`?
  
With this, our final TSP cost then amounts to checking for what
vertex `k` we get the lowest cost for `g(all_nodes, k) + c(k, 1)`
to complete our tour. While coding this more practically, we check
every combination of 1 element (`n-choose-1`), then 2 elements
(`n-choose-2`), then 3, etc. (this is where the `2^n` in the big-O
time complexity comes from), storing values computed along the way,
and keeping track of decisions made to be able to backtrack like
traditional dynamic programming solutions.

In our case, we have some added complexities:
- We need to dock ports, not just navigate to them, but that's
  just a few `+1`s to add to costs;
- The cost function `c` depends on the current tide offset we're
  at, but we can know this from the cost-so-far we look up in `g`;
- We can't assume without-loss-of-generality that starting at `1`
  is fine, so we need to repeat this for every possible starting
  city.

#### Speeding it up ⏩

We might be coding in Rust, this might be relatively cheap
processing, and we might only have 20 cities, but doing anything
in `O(2^20 * 20^3)` (`20^2` times 20 possible starting cities)
isn't exactly free. If we want this to work in a second, it needs
to go _fast_.

I started writing benchmarks and
[profiling with perf](https://nnethercote.github.io/perf-book/profiling.html)
to iterate on optimizations. The following were the most impactful,
but see the `Speed Optimizations Breakdown` section for details:
- Represent sets of elements as a `u32` mask, with clever
  [bit hacks](https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation)
  to go through all permutations with a fixed amount of bits set
  to 1. We can then use this mask directly as an integer index in
  a contiguous array;
- Hardcode the number of ports and tide schedule --
  `% len(tide_schedule)` is relatively cheap in general, but in
  a tight loop it can become a bottleneck. A hard-coded value
  can lead to the compiler replacing that with a more efficient
  way to compute `% 10`;
- Multithread the processing by trying multiple starting ports
  in parallel. This doesn't require synchronization so we can
  almost divide our processing time by the number cores we
  run on;
  - By running a "print the CPU info" Python bot on the server,
    I could see that there were 4 physical (i.e. without
    hyperthreading, since this is compute-bound processing)
    cores available;
- Speed up graph generation (pathfinding) a bit -- it's not the
  bottleneck, but savings here free up some "time budget" for
  Held-Karp:
  - Added multithreading, generating starting points in parallel;
  - Packed the tuple `(pos, wait)` A-star state as a `u32`:
    `pos.x << 16 | pos.y << 8 | wait` (works for our 60x60 maps
    and max 10 tick offsets), so that hashing can be trivially
    hashing just the `u32`;
  - Switched hashsets to `FxHashSet`, since the std one
    [can be slow](https://nnethercote.github.io/perf-book/hashing.html)
    (and was showing up in profiling);
  - Implemented "early exploration" (thanks to
    [this source](https://takinginitiative.wordpress.com/2011/05/02/optimizing-the-a-algorithm/)),
    where a neighbor node can skip the priority queue entirely
    if its f-score is smaller than or equal to the current one;
  - Changed the storage of the `adjacency` matrix of the graph
    to be `adjacency[to][from][tick_offset]`, to match the
    access patterns of Held-Karp iterating over a fixed `to`
    vertex in its hot loop (have values close in memory for
    better CPU cache usage).
- Place our data in our contiguous array in the order of the length
  of the subset `S`, since that matches the order in which we iterate
  in our dynamic programming solution. This leads to memory
  reads/writes that are closer in memory and take better
  advantage of the CPU cache (instead of very sporadic
  accesses when indexing by the mask as an integer directly).
  This is a bit tricky to do right, requiring some precomputed
  [binomials](https://en.wikipedia.org/wiki/Binomial_coefficient),
  but I found this great idea from 
  [this link](https://www.math.uwaterloo.ca/~bico/papers/comp_chapterDP.pdf)
  which documented it quite well.
  
And with all that... It could barely fit in a second in my local
tests! 🎉 It sometimes would go a bit over when you add up the
graph building costs, but I also noticed before that the server
looked like it allowed a bit over 1 second per tick (closer to 2s?),
so maybe we would be fine?

#### ... Ship It? 🚢

At that point I was really excited that I might be able to
basically "solve" the challenge (get the optimal score) when
the map's optimal solution is to visit 20 ports.

I uploaded to the server, kicked off a game, retried until I
got a 20 ports game and... score of `-1`. Uh oh, that's usually
a crash. I open the game and... it took _5 seconds_ to run our
first "planning" tick!

Well, that's a bummer. I checked a couple of things:
- The CPU on the server had a higher frequency than my
  desktop's, that seemed fine;
- It had a much smaller CPU cache than my desktop's, so I
  made some extra attempts to reduce the memory footprint
  (e.g. pack graph costs for all tide offsets in a `u64`,
  with a 'base' cost and a 'diff' from that base with 4
  bits, times 10 offsets). This helped a bit, but nowhere
  close to enough;
- The server runs on AMD, I have an Intel, maybe some
  critical Intel optimizations are happening...?
  - I deduced the AWS machine it was running on based on
    the specs I was seeing, then **spun up my own AWS
    server** to try and replicate the issue with the same
    specs (I really wanted that optimal solve each game),
    but I still didn't see the same slowdown.
- Spent a lot of time looking for ways to make changes
  to Held-Karp to make the starting port decision a part
  of the search, but the ideas I tried here sometimes gave
  suboptimal outputs, since it might be best to start from
  a node mid-way during the dynamic programming search,
  but at the end (when looping back to the start) the best
  start node might change. I didn't find a way to avoid
  having to do the`20x` processing here.
    
At that point it didn't seem worth it (or much fun) to try to
debug this slowdown blind, without being able to reproduce
& measure the causes of slowdowns. I tested different
number of threads, and it did look like running >3 threads
was slower despite the 4 physical cores, so it seems that
we might be getting a slice of the compute on the machine
we're running on (makes sense, not having the whole
machine to ourselves for each test!) and that we can't
effectively make full use of the physically available parallelism.

#### Redeeming this 🩹

In retrospect here, I spent a _lot_ of time trying to get
Held-Karp to work in 1s. I was having fun optimizing and
was seeing promising results locally. From a purely
"strategic" perspective, however, it would have been best
for me to pivot to either:
- a fully heuristic approach (e.g.
improve my ant solver);
- see if it's possible to adjust existing
  [Simplex algorithm approaches to TSP](https://www.cl.cam.ac.uk/teaching/1718/AdvAlgo/simplex_tsp.pdf)
  to our constraints (where the cost of an edge depends
  on the cost-so-far). For future reference for myself, I
  found
  [this Rust example](https://github.com/ztlpn/minilp/blob/master/examples/tsp.rs)
  of solving the TSP with linear programming.

But this wasn't lost time, this gave me a way to get
an offline **optimal solver** to evaluate how far off
my heuristic-based solver was doing on my collected
set of offline games. To truly get an optimal score,
we also need to consider the possibility of visiting
less than 20 ports. We can do that after running
Held-Karp by going through all subsets of ports once
more and checking if only visiting those ports (+
going home) would give a better score than the full
tour.

I then implemented a `live_evaler` tool that monitors
my `games` folder and evaluates new saved games as
they come in with an optimal solver, so that I can
keep an eye on my "could-have-been" best score. This
was also useful to compare to other teams on the
leaderboard to know if my ants-based bot was worse
than the other team's, or if I had maybe not been
given the chance to get a higher score just yet.

### 🦾 Final Solver

So I went back to an ants-based solver. I also tried
Held-Karp where I only consider a few starting ports
(that fits in a second), but on my offline set of
games this did worse than the ants solver.

I started running games in a loop, with a few minutes
sleep between games, and this turned into a very tight
race with the `Roach` team, and we kept passing each
other. I eventually found maps where the optimal score
was quite a few ticks better than my ants-based solution,
so I reswept my ants hyperparameters on just that
"high score potential" map and after that I found my
ants-based score to be always very close to the optimal
one, thankfully. 

But the `Roach` team kept passing me still, and from
interacting with the GraphQL API of the challenge I
was able to extract how many games each team has run
so far, to find that I was quite a bit behind in that
regard (so less chances for high-potential maps).

I changed my auto-play script to remove the sleep and
instead interact with the GraphQL API to check when
the game is complete (i.e. restart as soon as it
completes), and that allowed me to get a closer score
in the ~3800 points range.

`Roach` kept getting higher scores and I was still
behind on the number of games played, so I switched
to playing 2 games in parallel, respecting whatever
sleep the server suggested when we got throttled
(sorry for any infra trouble caused, Coveo :smile:,
apparently we overflowed the game ID column --
oops!). I also added logic to auto-refresh my
access token, by reusing firefox cookies for the
OAuth2 flow with
[browser_cookie3](https://github.com/borisbabic/browser_cookie3)
to avoid having to manually refresh it every day
(and lose precious hours where my bot could have
been running due to HTTP 401s!)

In the end, I ended up with a winning game:
- with a score of `3896` points;
- visiting 20 ports in 184 ticks;
- luckily enough, this was also the optimal score
  on that map;
- it was the highest possible optimal score I've
  observed in all games I collected.
  
Here is the winning game:

https://user-images.githubusercontent.com/1843555/202878765-ccdfe79c-b98c-46a0-907e-d3f90da75cef.mp4

This barely put me in the first place on the
leaderboard, with `Roach` right after at `3884`
points. To put this in perspective, this is
a difference of 2 ticks. I also went a bit
overboard with the auto-play script (I really
wanted to break the 3900 points range!), since
I played a total of **18576** games in total.
With `Roach` being a bit under 10000 games played,
I have no doubt that they could have gotten the
same score, too.

This was super fun and I learned a ton about
solving TSP problems, thanks Coveo for the great
challenge, as always! (Psst, I also hear that
[they're hiring](https://www.coveo.com/en/company/careers).)

## Speed Optimizations Breakdown

This section breaks down various optimizations that I made
throughout the challenge, and the impact that they had.
Every optimization is incremental to the previous one, and
the relative speed-ups are in relation to the previous row.

I evaluate the optimizations with benchmarks on a fixed
game:
- Creating a graph (pathfinding);
- Running the Ant Colony Optimization solver;
- Running the Held-Karp solver for all starts.

Because these are expensive operations, I override the
benchmark sample size to a small amount so that this can
run in a reasonable time, with the caveat that the
results can be pretty noisy.

The listed optimizations are not in the same order that I
added them during the challenge, but to be able to get an
understanding of how each contributed to the final speed,
I re-implemented "simple" versions of all operations, ~and
gradually re-introduced the same optimizations in branches
called `optimization-ablation-...` after-the-fact, if you'd
like to look at the incremental changes.~ TODO(emond): this
is not done yet, I just have the before/final numbers for
now.

### Pathfinding Optimizations

Comparing incremental improvements on `simple_graph.rs`
and `simple_pathfinding.rs` vs. `graph.rs` and
`pathfinding.rs`, in branch
[`optimization-ablation-pathfinding`](https://github.com/JesseEmond/blitz-2023-inscription/tree/optimization-ablation-pathfinding).

Going from the non-optimized "simple" versions to the
final ones on the graph creation benchmark gives a
**97%** relative improvement in compute time
(1.78s -> 58ms), i.e. it is **~31x faster**.

The optimizations are:
- [Use `FxHashMap`](https://nnethercote.github.io/perf-book/hashing.html)
  for `came_from`/`cost_so_far` storage;
- Hardcode tide schedule len/widths/heights as constants, switch to
  static arrays for topology and tide schedules;
- Precompute tile neighbors for all tick offsets;
- Use a heuristic based on min
  [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance)
  of all targets, reprioritize queue when a goal is found;
- Represent `(position, wait)` state as a packed `u32`;
- Do [early exploration](https://takinginitiative.wordpress.com/2011/05/02/optimizing-the-a-algorithm/),
  where neighbor nodes can skip the priority queue if their
  f-score is <= the current one;
- Multithread on 3 cores the pathfinding from different
  starting positions.

| Optimization | Commit | Benchmark Time | Speedup (relative improvement) |
| --- | --- | --- | --- |
| Base (simple implementation) | [8dea4d6](https://github.com/JesseEmond/blitz-2023-inscription/commit/8dea4d6bee03d593b31b2556e7f55af1146b259c) | 1786ms | _N/A_ |
| + `FxHashMap` | [cb22692](https://github.com/JesseEmond/blitz-2023-inscription/commit/cb226923cd689f26365d30853925a67cc197b65b) | 1359ms | 23.9% |
| + Use constants & static storage | [d8fbaa5](https://github.com/JesseEmond/blitz-2023-inscription/commit/d8fbaa5d6195dd8b702de707495756be9d53ef3d) | 1240ms | 8.8% |
| + Precompute neighbors | [afef254](https://github.com/JesseEmond/blitz-2023-inscription/commit/afef25405bb55c63caf90d07bdf265f9c0de6d5f) | 1185ms | 4.4% |
| + Use heuristic with reprioritization | [8294dbe](https://github.com/JesseEmond/blitz-2023-inscription/commit/8294dbef4cf00bc1f8aeac923bd59ad9679de148) | 198ms | 83.3% |
| + Pack A-star state as u32 | [56fbd48](https://github.com/JesseEmond/blitz-2023-inscription/commit/56fbd486b0bb6b359c4228ac470a362f5da394c3) | 193ms | 2.4% |
| + Early exploration | [bf14332](https://github.com/JesseEmond/blitz-2023-inscription/commit/bf14332cd0819b20ad959dfcc7ea23612ddfff6a) | 154ms | 20.4% |
| + Mutlithreading (3 cores) | [5f275ba](https://github.com/JesseEmond/blitz-2023-inscription/commit/5f275ba13540163a61cb13d2145756f591910510) | 58ms | 62.2% |

### Ant Colony Optimizations
Comparing incremental improvements on `simple_ant_colony_optimization.rs`
vs. `ant_colony_optimization.rs`.

Going from the non-optimized "simple" version to the
final one on the ACO benchmark gives a **73%**
relative improvement in compute time (564ms -> 151ms),
i.e. it is **~3.7x faster**.

The optimizations are:
- Remove `alpha` hyperparameter (power for pheromones), to avoid frequent
  `powf`s. This does change behavior, but after resweeping it looks like
  this was okay to do without losing points;
- Make use of continuous memory (`ArrayVec`s) for pheromones, ant
  storage, etc.;
- Store ant `seen` ports as a `u64` mask;
- Precompute heuristics and `eta ^ beta` once to avoid `powf`s;
- Use a fixed weight array with values forced to 0s for invalid choices,
  instead of building options on the fly and sampling on those.
- Cache weight computations for fast lookup in sampling, update when
  updating trails.
  
TODO(emond): Include table of relative optimization
ablation, adding one at a time incrementally

### Held-Karp Optimizations
Comparing incremental improvements on `simple_held_karp.rs` vs.
`held_karp.rs`.

Going from the non-optimized "simple" version to the final one
on the Held-Karp benchmark gives a **99%** relative
improvement in compute time (110.1s -> 1.1s), i.e. it is
**100x faster**.

The optimizations are:
- Store `g` and `p` values in a contiguous array instead of a
  `HashMap` (index by `mask` treated as an integer);
- Generate mask combinations of same size with
  [bit hacks](https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation);
- In the flattened contiguous array, store masks of similar
  size (number of elements in the set) closer in memory, for
  better caching (optimization
  [from here](https://www.math.uwaterloo.ca/~bico/papers/comp_chapterDP.pdf)).
- Multithreading, computing multiple start options in parallel;
- Change graph layout to be `adjacency[to][from][offset]` to
  follow access patterns from Held-Karp more closely;
- When computing index in flattened array in hot loop, compute
  offset from previous value instead of recomputing full expression;
- Use arrays/`ArrayVec`s where possible, use challenge consts for
  dims/loops;
- Precompute "translation" from vertex index to
  index-without-start;
- Use `get_unchecked` in graph adjacency cost accesses.

TODO(emond): Include table of relative optimization
ablation, adding one at a time incrementally

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
