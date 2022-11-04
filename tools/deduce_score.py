import sys


assert len(sys.argv) > 1, f'Usage: {sys.argv[0]} <score>'

score = int(sys.argv[1])


for ports in range(1, 20+1):
  if score % 2 == 0:  # if they maybe looped
    spawn_tick = 1
    dock_ticks = ports  # 1 for each port, last looped dock doesn't count
    base_ticks = spawn_tick + dock_ticks
    min_travel_ticks = ports + 1  # at least need to move off of each port + home
    if (ports * 125 - (base_ticks + min_travel_ticks) * 3) * 2 < score:
      continue  # Can't theoretically accumulate enough points for this score
    base_score = score // 2
    tick_points = ports * 125 - base_score
    if tick_points % 3 != 0:
      continue  # Not a possible number of ports for this score
    assert tick_points % 3 == 0, tick_points
    ticks = tick_points // 3
    travel_ticks = ticks - base_ticks
    assert travel_ticks >= min_travel_ticks, (travel_ticks, min_travel_ticks)
    print(f'{ports} ports, {ticks} total ticks ({base_ticks} base, {travel_ticks} travel), loop')

  total_ticks = 400
  if ports * 125 - total_ticks * 3 == score:
    print(f'{ports} ports, {total_ticks} total ticks, NO loop')
