import asyncio
import random
import time
from dataclasses import replace

from fastapi import WebSocket
from loguru import logger

import cpp_game
from ai.ai import AI, get_ai, supports_ai
from backend.app.schemas.game import (
    AddAi,
    ClientMessage,
    ConnectAck,
    EndedGame,
    EndGame,
    Error,
    FullState,
    FullSync,
    JoinedGame,
    JoinGame,
    KickedPlayer,
    KickPlayer,
    Move,
    Moved,
    ServerMessage,
    SetSettings,
    SettingsUpdated,
    StartedGame,
    StartGame,
)
from game.engine import Engine
from game.settings import Settings
from game.tasks import Task
from game.types import Action, Card
from game.utils import to_cpp_action, to_py_action


class GameRoom:
    def __init__(self, room_id: str) -> None:
        self.room_id: str = room_id
        self.websockets: dict[str, set[WebSocket]] = {}
        self.settings: Settings | None = None
        self.engine: Engine | None = None
        self.stage = "lobby"
        self.player_uids: list[str] = []
        self.handles: list[str] = []
        self.players: list[int] = []
        self.seqnum = 0
        self.last_update_time: float = time.time()
        self.num_players: int = 4
        self.difficulty: int = 7
        self.ai: AI | None = None
        self.ai_state: dict | None = None
        self.active_cards: list[tuple[Card, int]] = []

    def get_full_state(self) -> FullState:
        return FullState(
            start_seqnum=self.seqnum,
            room_id=self.room_id,
            stage=self.stage,
            player_uids=self.player_uids,
            handles=self.handles,
            players=self.players,
            engine_state=self.engine.state if self.engine else None,
            cur_uid=self.cur_uid,
            valid_actions=self.valid_actions,
            tasks=self.tasks,
            num_players=self.num_players,
            difficulty=self.difficulty,
            active_cards=self.active_cards,
        )

    @property
    def cur_uid(self) -> str | None:
        if self.stage != "play":
            return None

        assert self.engine is not None

        if self.engine.state.phase == "end":
            return None

        cur_player = self.engine.state.cur_player
        i = self.players.index(cur_player)
        return self.player_uids[i]

    @property
    def valid_actions(self) -> list[Action] | None:
        if self.stage != "play":
            return None

        assert self.engine is not None

        if self.engine.state.phase == "end":
            return None

        return self.engine.valid_actions()

    @property
    def tasks(self) -> dict[int, Task] | None:
        if self.stage != "play":
            return None

        assert self.engine is not None
        assert self.settings is not None

        ret = {}
        for task_idx in self.engine.state.task_idxs:
            formula, desc, difficulty = self.settings.task_defs[task_idx]
            ret[task_idx] = Task(
                formula=formula, desc=desc, difficulty=difficulty, task_idx=task_idx
            )

        return ret

    async def set_settings(
        self, num_players: int | None, difficulty: int | None, websocket: WebSocket
    ):
        if self.stage != "lobby":
            logger.error(f"Trying to set settings but stage {self.stage} is not lobby.")
            await self.send(
                websocket,
                Error(
                    room_id=self.room_id, message="Cannot set settings outside of lobby"
                ),
            )
            return

        if num_players is not None:
            self.num_players = num_players

        if difficulty is not None:
            self.difficulty = difficulty

        self.seqnum += 1
        await self.broadcast(
            SettingsUpdated(
                room_id=self.room_id,
                seqnum=self.seqnum,
                num_players=num_players,
                difficulty=difficulty,
            )
        )

    async def add_ai(self, websocket: WebSocket):
        if self.stage != "lobby":
            logger.error(f"Trying to add ai but stage {self.stage} is not lobby.")
            await self.send(
                websocket,
                Error(room_id=self.room_id, message="Cannot add AI outside of lobby"),
            )
            return

        if len(self.player_uids) >= 5:
            logger.error("Trying to add ai but too many players.")
            await self.send(
                websocket, Error(room_id=self.room_id, message="Too many players")
            )
            return

        if not supports_ai():
            logger.error("Do not support AI.")
            await self.send(
                websocket, Error(room_id=self.room_id, message="AI not supported")
            )
            return

        i = 0
        while True:
            if f"ai_{i}" not in self.player_uids:
                break
            i += 1

        uid = f"ai_{i}"
        handle = f"AI {i + 1}"
        self.player_uids.append(uid)
        self.handles.append(handle)
        self.seqnum += 1
        await self.broadcast(
            JoinedGame(room_id=self.room_id, seqnum=self.seqnum, uid=uid, handle=handle)
        )

    async def join_game(self, uid: str, handle: str, websocket: WebSocket):
        if self.stage != "lobby":
            logger.error(f"Trying to join game but stage {self.stage} is not lobby.")
            await self.send(
                websocket,
                Error(
                    room_id=self.room_id, message="Cannot join game outside of lobby"
                ),
            )
            return

        if uid in self.player_uids:
            logger.error(f"Trying to join game but uid {uid} already joined.")
            await self.send(
                websocket, Error(room_id=self.room_id, message="Already joined")
            )
            return

        if handle in self.handles:
            logger.error(f"Trying to join game but handle {handle} already taken.")
            await self.send(
                websocket, Error(room_id=self.room_id, message="Handle already taken")
            )
            return

        if len(self.player_uids) >= 5:
            logger.error("Trying to join game but too many players.")
            await self.send(
                websocket, Error(room_id=self.room_id, message="Too many players")
            )
            return

        self.player_uids.append(uid)
        self.handles.append(handle)
        self.seqnum += 1
        await self.broadcast(
            JoinedGame(room_id=self.room_id, seqnum=self.seqnum, uid=uid, handle=handle)
        )

    async def kick_player(self, handle: str, websocket: WebSocket):
        if self.stage != "lobby":
            logger.error(f"Trying to kick player but stage {self.stage} is not lobby")
            await self.send(
                websocket,
                Error(
                    room_id=self.room_id, message="Cannot kick player outside of lobby"
                ),
            )
            return

        if handle not in self.handles:
            logger.error(f"Trying to kick player but handle {handle} doesn't exist.")
            await self.send(
                websocket, Error(room_id=self.room_id, message="Player not found")
            )
            return

        i = self.handles.index(handle)
        uid = self.player_uids[i]
        del self.player_uids[i]
        del self.handles[i]
        self.seqnum += 1
        await self.broadcast(
            KickedPlayer(
                room_id=self.room_id, seqnum=self.seqnum, handle=handle, uid=uid
            )
        )

    async def start_game(self, websocket: WebSocket):
        if len(self.player_uids) == 0:
            logger.error("Trying to start game but no players joined.")
            await self.send(
                websocket, Error(room_id=self.room_id, message="No players joined")
            )
            return None

        if len(self.player_uids) != self.num_players:
            logger.error("Trying to start game but the wrong number of players joined.")
            await self.send(
                websocket,
                Error(
                    room_id=self.room_id,
                    message=f"Need {self.num_players} players, have {len(self.player_uids)}",
                ),
            )
            return None

        settings = Settings(
            num_players=self.num_players,
            min_difficulty=self.difficulty,
            max_difficulty=self.difficulty,
            max_num_tasks=8,
            use_signals=True,
            bank="all",
            use_drafting=True,
        )
        use_ai = any(x.startswith("ai_") for x in self.player_uids)
        if use_ai:
            ai_settings = replace(settings, use_signals=False)
            try:
                ai = get_ai(ai_settings)
            except ValueError:
                logger.error(f"Unsupported AI settings: {settings}")
                await self.send(
                    websocket,
                    Error(room_id=self.room_id, message="Unsupported AI settings"),
                )
                return None

        self.settings = settings
        self.stage = "play"
        self.engine = Engine(settings)
        seed = random.randint(1, 1_000_000_000)
        self.engine.reset_state(seed)
        self.players = list(range(self.settings.num_players))
        random.shuffle(self.players)
        self.players = self.players[: len(self.player_uids)]
        self.active_cards = []

        if use_ai:
            self.ai = ai
            self.ai_engine = cpp_game.Engine(ai_settings.to_cpp())
            self.ai_engine.reset_state(seed)
            assert self.engine.state.task_idxs == self.ai_engine.state.task_idxs
            self.ai_state = self.ai.new_rollout()
        else:
            self.ai = None
            self.ai_engine = None
            self.ai_state = None

        self.seqnum += 1
        await self.broadcast(
            StartedGame(
                room_id=self.room_id,
                seqnum=self.seqnum,
                players=self.players,
                cur_uid=self.cur_uid,
                engine_state=self.engine.state,
                valid_actions=self.valid_actions,
                tasks=self.tasks,
            )
        )
        await self.auto_move()

    async def engine_move(self, action):
        is_signal_phase = self.engine.state.phase == "signal"
        if self.ai is not None:
            start_time = time.time()
            if is_signal_phase:
                ai_action = Action(self.engine.state.cur_player, "nosignal")
            else:
                ai_action = to_py_action(
                    self.ai.get_move(self.ai_engine, self.ai_state)
                )

            if action == "ai":
                action = ai_action
                elapsed = time.time() - start_time
                sleep_time = 2
                if elapsed < sleep_time:
                    await asyncio.sleep(sleep_time - elapsed)

        self.engine.move(action)
        if self.ai_engine and not is_signal_phase:
            self.ai_engine.move(to_cpp_action(action))
        if self.engine.state.active_cards:
            self.active_cards = self.engine.state.active_cards

        self.seqnum += 1
        msg = Moved(
            room_id=self.room_id,
            seqnum=self.seqnum,
            action=action,
            cur_uid=self.cur_uid,
            engine_state=self.engine.state,
            valid_actions=self.valid_actions,
            active_cards=self.active_cards,
        )
        await self.broadcast(msg)

    async def auto_move(self):
        assert self.engine is not None

        while self.engine.state.phase != "end":
            valid_actions = self.engine.valid_actions()
            if len(valid_actions) == 1 and valid_actions[0].type in [
                "nodraft",
                "nosignal",
            ]:
                await self.engine_move(valid_actions[0])
            elif self.cur_uid.startswith("ai_"):
                await self.engine_move("ai")
            else:
                break

    async def move(self, action: Action, uid: str, websocket: WebSocket):
        if self.stage != "play":
            logger.error(f"Trying to move but stage {self.stage} is not play")
            await self.send(
                websocket, Error(room_id=self.room_id, message="Game is not in play")
            )
            return None

        assert self.engine is not None

        if self.engine.state.phase == "end":
            logger.error("Trying to move but game has ended")
            await self.send(
                websocket, Error(room_id=self.room_id, message="Game has ended")
            )
            return None

        if uid not in self.player_uids:
            logger.error(f"Trying to move but uid {uid} is not a player")
            await self.send(
                websocket,
                Error(room_id=self.room_id, message="Not a player in this game"),
            )
            return None

        if uid != self.cur_uid:
            logger.error(f"Trying to move but uid {uid} is not cur_uid {self.cur_uid}")
            await self.send(
                websocket, Error(room_id=self.room_id, message="Not your turn")
            )
            return None

        valid_actions = self.engine.valid_actions()
        if action not in valid_actions:
            logger.error(f"Trying to move but action {action} is invalid.")
            await self.send(
                websocket, Error(room_id=self.room_id, message="Invalid action")
            )
            return None

        # Make the move
        await self.engine_move(action)

        # Continue with auto moves
        await self.auto_move()

    async def end_game(self, websocket: WebSocket):
        if self.stage != "play":
            logger.error(f"Trying to end game but stage {self.stage} is not play")
            await self.send(
                websocket, Error(room_id=self.room_id, message="Game is not in play")
            )
            return

        assert self.engine is not None

        self.stage = "lobby"
        self.players = []
        self.seqnum += 1
        await self.broadcast(EndedGame(room_id=self.room_id, seqnum=self.seqnum))

    async def connect(self, websocket: WebSocket, uid: str) -> None:
        await websocket.accept()
        self.websockets.setdefault(uid, set()).add(websocket)
        await self.send(websocket, ConnectAck(uid=uid, room_id=self.room_id))
        self.last_update_time = time.time()

    async def process_message(
        self, message: ClientMessage, uid: str, websocket: WebSocket
    ) -> None:
        match message:
            case FullSync():
                await self.send(websocket, self.get_full_state())
            case JoinGame():
                await self.join_game(uid, message.handle, websocket)
            case KickPlayer():
                await self.kick_player(message.handle, websocket)
            case SetSettings():
                await self.set_settings(
                    message.num_players, message.difficulty, websocket
                )
            case AddAi():
                await self.add_ai(websocket)
            case StartGame():
                await self.start_game(websocket)
            case Move():
                await self.move(message.action, uid, websocket)
            case EndGame():
                await self.end_game(websocket)

        self.last_update_time = time.time()

    def disconnect(self, websocket: WebSocket, uid: str) -> None:
        self.websockets[uid].remove(websocket)
        if not self.websockets[uid]:
            del self.websockets[uid]

    async def send(self, websocket: WebSocket, message: ServerMessage) -> None:
        await websocket.send_text(message.model_dump_json())

    async def broadcast(self, message: ServerMessage) -> None:
        for websockets in self.websockets.values():
            for websocket in websockets:
                await self.send(websocket, message)
