import random
import time

from fastapi import WebSocket
from loguru import logger

from backend.app.schemas.game import (
    AddAi,
    ClientMessage,
    ConnectAck,
    EndedGame,
    EndGame,
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
    TrickWon,
)
from ai.ai import AI, get_ai
from game.engine import Engine
from game.settings import Settings
from game.tasks import Task
from game.types import Action


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

    def get_full_state(self) -> FullState:
        return FullState(
            seqnum=self.seqnum,
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

    def set_settings(self, num_players: int | None, difficulty: int | None):
        if self.stage != "lobby":
            logger.error(f"Trying to set settings but stage {self.stage} is not lobby.")
            return None

        if num_players is not None:
            self.num_players = num_players

        if difficulty is not None:
            self.difficulty = difficulty

        self.seqnum += 1
        return SettingsUpdated(
            room_id=self.room_id,
            seqnum=self.seqnum,
            num_players=num_players,
            difficulty=difficulty,
        )

    def add_ai(self):
        if self.stage != "lobby":
            logger.error(f"Trying to add ai but stage {self.stage} is not lobby.")
            return None

        if len(self.player_uids) >= 5:
            logger.error("Trying to add ai but too many players.")
            return None

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
        return JoinedGame(
            room_id=self.room_id, seqnum=self.seqnum, uid=uid, handle=handle
        )

    def join_game(self, uid: str, handle: str):
        if self.stage != "lobby":
            logger.error(f"Trying to join game but stage {self.stage} is not lobby.")
            return None

        if uid in self.player_uids:
            logger.error(f"Trying to join game but uid {uid} already joined.")
            return None

        if handle in self.handles:
            logger.error(f"Trying to join game but handle {handle} already taken.")
            return None

        if len(self.player_uids) >= 5:
            logger.error("Trying to join game but too many players.")
            return None

        self.player_uids.append(uid)
        self.handles.append(handle)
        self.seqnum += 1
        return JoinedGame(
            room_id=self.room_id, seqnum=self.seqnum, uid=uid, handle=handle
        )

    def kick_player(self, handle: str):
        if self.stage != "lobby":
            logger.error(f"Trying to kick player but stage {self.stage} is not lobby")
            return None

        if handle not in self.handles:
            logger.error(f"Trying to kick player but handle {handle} doesn't exist.")
            return None

        i = self.handles.index(handle)
        uid = self.player_uids[i]
        del self.player_uids[i]
        del self.handles[i]
        self.seqnum += 1
        return KickedPlayer(
            room_id=self.room_id, seqnum=self.seqnum, handle=handle, uid=uid
        )

    def start_game(self):
        if len(self.player_uids) == 0:
            logger.error("Trying to start game but no players joined.")
            return None

        if len(self.player_uids) != self.num_players:
            logger.error("Trying to start game but the wrong number of players joined.")
            return None

        self.settings = Settings(
            num_players=self.num_players,
            min_difficulty=self.difficulty,
            max_difficulty=self.difficulty,
            max_num_tasks=4,
            use_signals=False,
            bank="med",
            use_drafting=True,
        )
        self.stage = "play"
        if any(x.startswith("ai_") for x in self.player_uids):
            self.ai = get_ai(self.settings)
            self.ai_state = self.ai.new_rollout()
        else:
            self.ai = None
            self.ai_state = None
        self.engine = Engine(self.settings)
        self.engine.reset_state()
        self.players = list(range(self.settings.num_players))
        random.shuffle(self.players)
        self.players = self.players[: len(self.player_uids)]

        msgs = []
        self.seqnum += 1
        msgs.append(
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
        msgs.extend(self.auto_move())
        return msgs

    def engine_move(self, action=None):
        if self.ai is not None:
            ai_action = self.ai.get_move(self.engine, self.ai_state)
            if action == "ai":
                action = ai_action
            self.ai.record_move(self.engine, action, self.ai_state)

        msgs = []
        prev_trick = self.engine.state.trick
        prev_phase = self.engine.state.phase
        self.engine.move(action)
        self.seqnum += 1
        msgs.append(
            Moved(
                room_id=self.room_id,
                seqnum=self.seqnum,
                action=action,
                cur_uid=self.cur_uid,
                engine_state=self.engine.state,
                valid_actions=self.valid_actions,
            )
        )
        if self.engine.state.trick > prev_trick and prev_phase == "play":
            self.seqnum += 1
            msgs.append(
                TrickWon(
                    room_id=self.room_id,
                    seqnum=self.seqnum,
                    trick=prev_trick,
                    trick_winner=self.engine.state.past_tricks[-1][1],
                )
            )
        return msgs

    def auto_move(self):
        assert self.engine is not None
        msgs = []

        while self.engine.state.phase != "end":
            valid_actions = self.engine.valid_actions()
            if self.cur_uid.startswith("ai_"):
                msgs.extend(self.engine_move("ai"))
            elif len(valid_actions) == 1 and valid_actions[0].type in [
                "nodraft",
                "nosignal",
            ]:
                msgs.extend(self.engine_move(valid_actions[0]))
            else:
                break
        return msgs

    def move(self, action: Action, uid: str):
        if self.stage != "play":
            logger.error(f"Trying to move but stage {self.stage} is not play")
            return None

        assert self.engine is not None

        if self.engine.state.phase == "end":
            logger.error("Trying to move but game has ended")
            return None

        if uid not in self.player_uids:
            logger.error(f"Trying to move but uid {uid} is not a player")
            return None

        if uid != self.cur_uid:
            logger.error(f"Trying to move but uid {uid} is not cur_uid {self.cur_uid}")
            return None

        valid_actions = self.engine.valid_actions()
        if action not in valid_actions:
            logger.error(f"Trying to move but action {action} is invalid.")
            return None

        msgs = []

        # Make the move
        msgs.extend(self.engine_move(action))

        # Continue with auto moves
        msgs.extend(self.auto_move())

        return msgs

    def end_game(self):
        if self.stage != "play":
            logger.error(f"Trying to end game but stage {self.stage} is not play")
            return None

        assert self.engine is not None

        self.stage = "lobby"
        self.engine.reset_state()
        self.players = []
        self.seqnum += 1
        return EndedGame(room_id=self.room_id, seqnum=self.seqnum)

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
                resp = None
            case JoinGame():
                resp = self.join_game(uid, message.handle)
            case KickPlayer():
                resp = self.kick_player(message.handle)
            case SetSettings():
                resp = self.set_settings(message.num_players, message.difficulty)
            case AddAi():
                resp = self.add_ai()
            case StartGame():
                resp = self.start_game()
            case Move():
                resp = self.move(message.action, uid)
            case EndGame():
                resp = self.end_game()

        if resp is not None:
            if isinstance(resp, list):
                for r in resp:
                    await self.broadcast(r)
            else:
                await self.broadcast(resp)

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
