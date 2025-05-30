from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from game.state import State
from game.tasks import Task
from game.types import Action


class ServerMessageBase(BaseModel):
    type: str
    room_id: str


class ConnectAck(ServerMessageBase):
    type: Literal["connect_ack"] = "connect_ack"
    uid: str


class FullState(ServerMessageBase):
    type: Literal["full_state"] = "full_state"
    seqnum: int
    stage: str
    player_uids: list[str]
    handles: list[str]
    players: list[int]
    cur_uid: str | None
    engine_state: State | None
    valid_actions: list[Action] | None
    tasks: dict[int, Task] | None
    num_players: int
    difficulty: int


class JoinedGame(ServerMessageBase):
    type: Literal["joined_game"] = "joined_game"
    seqnum: int
    uid: str
    handle: str


class KickedPlayer(ServerMessageBase):
    type: Literal["kicked_player"] = "kicked_player"
    seqnum: int
    handle: str
    uid: str


class SettingsUpdated(ServerMessageBase):
    type: Literal["settings_updated"] = "settings_updated"
    seqnum: int
    num_players: int | None = None
    difficulty: int | None = None


class StartedGame(ServerMessageBase):
    type: Literal["started_game"] = "started_game"
    seqnum: int
    players: list[int]
    cur_uid: str | None
    engine_state: State
    valid_actions: list[Action] | None
    tasks: dict[int, Task] | None


class Moved(ServerMessageBase):
    type: Literal["moved"] = "moved"
    seqnum: int
    action: Action
    cur_uid: str | None
    engine_state: State
    valid_actions: list[Action] | None


class EndedGame(ServerMessageBase):
    type: Literal["ended_game"] = "ended_game"
    seqnum: int


class TrickWon(ServerMessageBase):
    type: Literal["trick_won"] = "trick_won"
    room_id: str
    seqnum: int
    trick_winner: int
    trick: int


ServerMessage = Annotated[
    Union[
        ConnectAck,
        FullState,
        JoinedGame,
        KickedPlayer,
        SettingsUpdated,
        StartedGame,
        Moved,
        EndedGame,
        TrickWon,
    ],
    Field(discriminator="type"),
]
ServerMessageAdapter = TypeAdapter(ServerMessage)  # type: ignore


class ClientMessageBase(BaseModel):
    type: str
    room_id: str


class FullSync(ClientMessageBase):
    type: Literal["full_sync"] = "full_sync"


class AddAi(ClientMessageBase):
    type: Literal["add_ai"] = "add_ai"


class JoinGame(ClientMessageBase):
    type: Literal["join_game"] = "join_game"
    handle: str


class SetSettings(ClientMessageBase):
    type: Literal["set_settings"] = "set_settings"
    num_players: int | None = None
    difficulty: int | None = None


class KickPlayer(ClientMessageBase):
    type: Literal["kick_player"] = "kick_player"
    handle: str


class StartGame(ClientMessageBase):
    type: Literal["start_game"] = "start_game"


class Move(ClientMessageBase):
    type: Literal["move"] = "move"
    action: Action


class EndGame(ClientMessageBase):
    type: Literal["end_game"] = "end_game"


ClientMessage = Annotated[
    Union[FullSync, AddAi, JoinGame, SetSettings, KickPlayer, StartGame, Move, EndGame],
    Field(discriminator="type"),
]
ClientMessageAdapter = TypeAdapter(ClientMessage)  # type: ignore
