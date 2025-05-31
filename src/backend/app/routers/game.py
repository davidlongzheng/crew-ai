from __future__ import annotations

import random
import time
import uuid

from fastapi import (
    APIRouter,
    Query,
    Response,
    WebSocket,
    WebSocketDisconnect,
)

from backend.app.schemas.game import ClientMessage, ClientMessageAdapter
from backend.app.services.game import GameRoom

router = APIRouter(prefix="/game")

# Store active game rooms
game_rooms: dict[str, GameRoom] = {}


@router.post("/create_room")
async def create_room():
    for game_room in game_rooms.values():
        if time.time() - game_room.last_update_time > 24 * 60 * 60:
            del game_room

    while True:
        room_id = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(4))
        if room_id not in game_rooms:
            game_rooms[room_id] = GameRoom(room_id)
            break

    return {"room_id": room_id}


@router.post("/set_uid")
def set_uid(response: Response = Response()):
    uid = str(uuid.uuid4())
    response.set_cookie(key="uid", value=uid)
    return {"uid": uid}


@router.websocket("/ws/{room_id}")
async def websocket_endpoint(
    websocket: WebSocket, room_id: str, uid: str = Query(...)
) -> None:
    if room_id not in game_rooms:
        await websocket.close(code=4004, reason="Room not found")
        return
    room = game_rooms[room_id]
    await room.connect(websocket, uid)

    try:
        while True:
            # Receive message from client
            text = await websocket.receive_text()
            msg: ClientMessage = ClientMessageAdapter.validate_json(text)
            await room.process_message(msg, uid, websocket)
    except WebSocketDisconnect:
        room.disconnect(websocket, uid)
