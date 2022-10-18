from bot import Bot
from game_message import Tick, Map, Position
import pathfinding

import games_seen

# TODO: do a proper test server instead
tick = games_seen.GAME1
bot = Bot()
spawn = bot.get_next_move(tick)
print(spawn)
tick.currentLocation = spawn.position
tick.spawnLocation = spawn.position
# dock
print(bot.get_next_move(tick))
# move
print(bot.get_next_move(tick))


highlights = []#[Position(row=43,column=28), Position(row=42,column=27)]
# src = Position(row=20, column=32)
# dst = Position(row=36, column=36)
# highlights.extend(pathfinding.shortest_path(bot.graph, src, dst))
bot.graph.debug_print(highlights=highlights)


