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
might be better than others because they start us in a favorable tide
point in the schedule.

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
are navigable with the tide for the current tick offset. This current
tide value can be found by looking up
`tide_schedule[tick % len(tide_schedule)]` (since it cycles), by using
the `A*` `g` score as the `tick` value (the cost so far).

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
precompute all of the following at once:
- For each port, shortest path to all other ports;
- Do that for each possible tick offset in the tide schedule.

The reason for doing so is that we can then create a graph
from our ports:
- Each port is a **vertex**;
- Each port connects to other ports via a precomputed shortest
  path -- that's an **edge**;
- The **cost** of this edge (and the exact path details)
  depends on the tick offset we are at on the source vertex.

But this gets expensive: 10 tick offsets, 20 vertices,
190 edges (fully connected graph), that's 1900 total shortest
paths if we do them pairwise. If we want to fit this in 1
second and do processing on that graph afterwards, we need
to speed things up:
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
  `Speed Optimization Ablation` section.

TODO: graph visualization

Around this time, I added a "give up" mode to my bot, that
quickly docks the initial port twice to get 0 points and
early-exit. Because this graph processing is getting a bit
compute-heavy and we are mostly interested in 20-ports
games to get a higher score on the leaderboard, we can
save time (and minimize our impact on Coveo infrastructure
as much as possible) by skipping games that are < 20 ports.


### 🕴️ ... And Now It's a TSP!

Now we have a graph and we're effectively trying to find
the shortest possible tour that visits each "city" (port)
once, returning to the origin. That's a
[Traveling Salesperson Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem)
(TSP)! A famously NP-hard problem.

On distinction, however, is that our edge costs are
dynamic, depending on the cost of our tour _so far_
(changes the tick offset we'd be at, and thus the
shortest path & cost to other ports). This means that
traditional approaches to solving TSPs have to be
adapted.

One simple example of this is that TSP
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
  adjusting the micro state machine to provide exact
  paths to follow;
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
    heuristic = graph.cost(ant.position, ant.current_tick, option)
    weight = pow(pheromone, ALPHA) * pow(heuristic, BETA)
    weights.append(weight)
  return Sample(weights)
  
def LocalSearch(ants):
  for ant in ants:
    ant.GoHomeEarlyIfBetter()  # close loop earlier if gives a better score
    
def UpdateTrails(ants):
  for source in ports:
    for dest in ports:
      pheromones[source][dest] *= (1 - EVAPORATION_RATE)
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
- **Ant System** (AS): as defined above;
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
  - Pheromones are initially set to the max to promot exploration.
- And many others

To support a range of variants, I implemented _ACO_ with multiple
hyperparameters:
- _iterations_: total number of ant simulations for the algorithm;
- _ants_: number of ants simulated per iteration;
- _evaporation_rate_: "EVAPORATION_RATE" above -- used when
  evaporation pheromones, and used as a multiplier for added
  pheromones;
- _exploitation_probability_: probability of taking the max-weight
  option in sampling;
- **not** _alpha_: power for the pheromone -- I removed this from
  computations to speed up processing (see the `Speed Optimization
  Ablation` section), which is not strictly equivalent even
  post-sweeps, but gave me decent results while being faster
  (allowing more iterations/ants).
- _beta_: power for the heuristic when computing sampling weights;
- _local_evaporation_rate_: used in local updates (from _ACO_
  definition) to disincentivize other ants from the same iteration
  to promote exploration;
- _min_pheromones_: min value that pheromones can have, from _MMAS_.
- _max_pheromones_: max value that pheromones can have, from _MMAS_.
- _pheromone_init_ratio_: from min to max pheromones, what value
  should we initialize pheromones with (0 = min, 1 = max).
  
The idea was to make it so that the different variants were
somewhat available as different points in the hyperparameter
space:
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
behavior was like. Here is what is looks like on an example game:

TODO ant visualization

TODO describe the visuals

Next, to pick a good set of hyperparameters, I implemented a
couple of extra tools:
- A `collect_games.py` tool that runs games on the server
  in a loop (note: also useful when we're trying to get lucky
  with a high-potential game!), downloads the logs and saves
  the games to disk;
- Extended `server.py` to support an "eval" mode that goes
  through all locally stored games and runs our bots to give
  us some stats (min/max/avg) on our how our bot is doing on
  a range of games;
  
With this, I could then add support for hyperparameter
sweeping plugged in the `server.py` eval logic. I
used [vizier](https://github.com/google/vizier) which I have
some familiarity with. We can define hyperparameter ranges,
pick an optimizer, and let it explore the hyperparameter
space. In the end, I stuck to random search, but this is
easy to change and extend with your own search algorithm.

TODO sweep viz

I could have added more tuning here -- there are other
ant system variants or settings that can be useful (e.g.
a schedule between picking the iteration local best ant
vs. global best one) and I made some arbitrary decisions
(e.g. assign pheromone trails to directed edges, I didn't
try undirected ones or more granular per-tick-offset ones).

But instead, I got distracted...

### ✍️ Exact Solver: Held-Karp

A 20 "city" TSP is not _that_ big. Even with an exponential
time exact TSP algorithm like
[Held-Karp](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm)
that runs in `O(2^n n^2)`, that's within a constant factor of
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
  
With this, our TSP cost is then as simple as checking for what
vertex `k` we get the lowest cost for `g(all_nodes, k) + c(k, 1)`
to complete our tour. While coding this, we check every
combination of 1 element (`n-choose-1`), then 2 elements
(`n-choose-2`), then 3, etc. (this is where the `2^n` comes from),
storing values computed along the way, and keeping track of
decisions made to be able to backtrack like traditional dynamic
programming solutions.

In our case, we have some added complexities:
- We need to dock ports, not just navigate to them, but that's
  just a few `+1`s to add;
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
but see the `Speed Optimizations Ablation` section for details:
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
    [this source](https://takinginitiative.wordpress.com/2011/05/02/optimizing-the-a-algorithm/),
    where a neighbor node can skip the priority queue entirely
    if its f-score is smaller than or equal to the current one.
- Place our data in our contiguous array in order of length of
  subset `S`, since that matches the order in which we iterate
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
  desktop's;
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
    
At that point it didn't seem worth it (or fun) to try to
debug this slowdown blind, without being able to reproduce
& measure the causes of slowdowns. I tested different
number of threads, and it did look like running >3 threads
was slower despite the 4 physical cores, so it seems that
we might be getting a slice of the compute on the machine
we're running on (makes sense, not having the whole
machine to ourselves for each test!) and that we can't
make full use of the physically available parallelism.

#### Redeeming this 🩹
TODO restrospect: spent too much time on this, should have switched to try
     adapting simplex-based approaches
TODO optimal solver
TODO live evaller

### 🦾 Final Solver

TODO held-karp N starts
TODO go back to ant, let it run, reswept on higher scoring map
TODO helper that runs in a loop w/ graphql, collects games, reauths everyday

TODO final score, TODO in how many runs

## Speed Optimizations Ablation

### Benchmarks

TODO which ones

### Pathfinding Optimizations
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
