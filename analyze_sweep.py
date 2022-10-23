from dataclasses import dataclass
import matplotlib.pyplot as plt
import json
import sys

import hyperparams


assert len(sys.argv) > 1, f'Usage: {sys.argv[0]} <logs_file.txt>'

logs_file = sys.argv[1]
with open(logs_file, 'r') as f:
  log_lines = [line.strip() for line in f.readlines()]
  prefix = '[SCORE]'
  lines = [line[len(prefix):] for line in log_lines if line.startswith(prefix)]


@dataclass
class Point:
  hyperparams: hyperparams.Hyperparams
  score: float


line_parts = [line.split(' ', 1) for line in lines]
points = [
    Point(score=float(score),
          hyperparams=hyperparams.Hyperparams.from_dict(
            json.loads(params)))
    for score, params in line_parts]


# Little helper to find a decent point with reasonable compute
score_pts = sorted(points, key=lambda p: p.score, reverse=True)
for p in score_pts:
  if p.hyperparams.ants * p.hyperparams.iterations < 200 * 200:
    print('Use these params:')
    print(p)
    break


fig, axs = plt.subplots(3, 3)
plt.get_current_fig_manager().window.showMaximized()

axs[0,0].set_title('score per iteration')
pts = {'iter': list(range(len(points))), 'score': [p.score for p in points]}
axs[0,0].plot('iter', 'score', 'o', data=pts)

axs[0,1].set_title('score per total num iterations')
total_iters_pts = sorted(points, key=lambda p: p.hyperparams.iterations)
pts = {'total_iters': [p.hyperparams.iterations for p in total_iters_pts],
       'score': [p.score for p in total_iters_pts]}
axs[0,1].plot('total_iters', 'score', 'o', data=pts)

axs[0,2].set_title('score per num ants')
ants_pts = sorted(points, key=lambda p: p.hyperparams.ants)
pts = {'num_ants': [p.hyperparams.ants for p in ants_pts],
       'score': [p.score for p in ants_pts]}
axs[0,2].plot('num_ants', 'score', 'o', data=pts)

axs[1,0].set_title('score per evaporation rate')
evaporation_rate_pts = sorted(points, key=lambda p: p.hyperparams.evaporation_rate)
pts = {'evaporation_rate': [p.hyperparams.evaporation_rate for p in evaporation_rate_pts],
       'score': [p.score for p in evaporation_rate_pts]}
axs[1,0].plot('evaporation_rate', 'score', 'o', data=pts)

axs[2,0].set_title('score per beta')
beta_pts = sorted(points, key=lambda p: p.hyperparams.heuristic_power)
pts = {'beta': [p.hyperparams.heuristic_power for p in beta_pts],
       'score': [p.score for p in beta_pts]}
axs[2,0].plot('beta', 'score', 'o', data=pts)

axs[1,1].set_title('score per base pheromones')
base_pheromones_pts = sorted(points, key=lambda p: p.hyperparams.base_pheromones)
pts = {'base_pheromones': [p.hyperparams.base_pheromones for p in base_pheromones_pts],
       'score': [p.score for p in base_pheromones_pts]}
axs[1,1].plot('base_pheromones', 'score', 'o', data=pts)

axs[2,1].set_title('score per local evaporation rate')
local_evaporation_rate_pts = sorted(points, key=lambda p: p.hyperparams.local_evaporation_rate)
pts = {'local_evaporation_rate': [p.hyperparams.local_evaporation_rate for p in local_evaporation_rate_pts],
       'score': [p.score for p in local_evaporation_rate_pts]}
axs[2,1].plot('local_evaporation_rate', 'score', 'o', data=pts)

plt.show()
