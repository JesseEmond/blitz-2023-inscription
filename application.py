#!/usr/bin/env python

import asyncio
import os
import websockets
import json

from bot import Bot
from game_message import Tick


async def run():
    uri = "ws://127.0.0.1:8765"

    async with websockets.connect(uri) as websocket:
        bot = Bot()
        if "TOKEN" in os.environ:
            await websocket.send(json.dumps({"type": "REGISTER", "token": os.environ["TOKEN"]}))
        else:
            await websocket.send(json.dumps({"type": "REGISTER", "teamName": "MyPythonicBot"}))

        await game_loop(websocket=websocket, bot=bot)


async def game_loop(websocket: websockets.WebSocketServerProtocol, bot: Bot):
    while True:
        try:
            message = await websocket.recv()
        except websockets.exceptions.ConnectionClosed:
            # Connection is closed, the game is probably over
            print("Websocket was closed.")
            break

        game_message: Tick = Tick.from_dict(json.loads(message))
        print(f"Playing tick {game_message.currentTick} of {game_message.totalTicks}")

        payload = {
            "type": "COMMAND",
            "tick": game_message.currentTick,
            "action": bot.get_next_move(game_message).to_dict()
        }

        await websocket.send(json.dumps(payload))


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(run())
