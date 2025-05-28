from __future__ import annotations

import asyncio
from collections import deque

import pytest
from async_asgi_testclient import TestClient

from backend.app.main import app
from backend.app.schemas.game import (
    EndGame,
    FullSync,
    JoinGame,
    Move,
    StartGame,
)


@pytest.mark.asyncio
async def test_game_flow() -> None:
    client = TestClient(app)

    response = await client.post("/game/create_room")
    assert response.status_code == 200
    room_id = response.json()["room_id"]
    print(f"{room_id=}")

    handles = ["Player1", "Player2", "Player3", "Player4"]

    async def run_client(handle):
        response = await client.post("/game/set_uid")
        assert response.status_code == 200
        uid = response.json()["uid"]

        async with client.websocket_connect(
            f"/game/ws/{room_id}?uid={uid}"
        ) as websocket:
            status = "wait_connect"
            state = None
            message_q = deque()
            while True:
                if message_q and state is not None:
                    message = message_q.popleft()
                else:
                    message = await websocket.receive_json()

                if "seqnum" in message:
                    if state is not None:
                        if state["seqnum"] >= message["seqnum"]:
                            continue
                        else:
                            assert message["seqnum"] == state["seqnum"] + 1
                            state["seqnum"] = message["seqnum"]
                    elif message["type"] != "full_state":
                        message_q.append(message)
                        continue

                if message["type"] == "connect_ack":
                    assert status == "wait_connect"
                    status = "wait_sync"
                    await websocket.send_json(FullSync(room_id=room_id).model_dump())
                elif message["type"] == "full_state":
                    assert status == "wait_sync"
                    state = {
                        k: v for k, v in message.items() if k not in ["type", "room_id"]
                    }
                    assert state["stage"] == "lobby"
                    await websocket.send_json(
                        JoinGame(room_id=room_id, handle=handle).model_dump()
                    )
                    status = "wait_join"
                elif message["type"] == "joined_game":
                    assert status in ["wait_join", "joined"]
                    state["player_uids"].append(message["uid"])
                    state["handles"].append(message["handle"])

                    if uid == message["uid"]:
                        status = "joined"

                    if handle == "Player1" and len(state["handles"]) == len(handles):
                        await websocket.send_json(
                            StartGame(room_id=room_id).model_dump()
                        )
                elif message["type"] in ["started_game", "moved"]:
                    assert status in ["joined", "play"]
                    status = "play"

                    for k in ["players", "cur_uid", "engine_state", "valid_actions"]:
                        if k in message:
                            state[k] = message[k]

                    if state["engine_state"]["phase"] != "end":
                        valid_actions = state["valid_actions"]
                        assert valid_actions
                        assert state["cur_uid"] is not None

                        if uid == state["cur_uid"]:
                            action = valid_actions[0]
                            await websocket.send_json(
                                Move(room_id=room_id, action=action).model_dump()
                            )
                    else:
                        break
                else:
                    raise ValueError(str(message))

            assert state["engine_state"]["phase"] == "end"
            if handle == "Player1":
                await websocket.send_json(EndGame(room_id=room_id).model_dump())
                message = await websocket.receive_json()
                assert message["type"] == "ended_game"

    await asyncio.gather(*(run_client(handle) for handle in handles))
