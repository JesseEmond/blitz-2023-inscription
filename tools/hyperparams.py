from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Hyperparams:
  iterations: int
  ants: int
  evaporation_rate: float
  exploitation_probability: float
  heuristic_power: float
  local_evaporation_rate: float
  min_pheromones: float
  max_pheromones: float
  pheromones_init_ratio: float
  seed: int
