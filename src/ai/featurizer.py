import torch
from tensordict import TensorDict

from game.settings import Settings
from lib.types import StrMap


def featurize(
    public_history: StrMap | list[StrMap] | list[list[StrMap]],
    private_inputs: StrMap | list[StrMap] | list[list[StrMap]],
    valid_actions: tuple | list[tuple] | list[list[tuple]],
    settings: Settings,
    *,
    non_feature_dims: int,
    device: torch.device | None = None,
) -> TensorDict:
    """Takes in a set of nn inputs and featurizes the inputs.

    Inputs to the func are either F, (T, F), (N, F), or (N, T, F).
    You should set non_feature_dims=0,1,1,2 respectively.
    """

    assert 0 <= non_feature_dims <= 2

    def mapper(func, inp, _non_feature_dims=non_feature_dims):
        if _non_feature_dims == 0:
            ret = func(inp)
        elif _non_feature_dims == 1:
            ret = [func(x) for x in inp]
        else:
            ret = [[func(y) for y in x] for x in inp]

        return ret

    def pad_tasks(x):
        return x + [(-1, -1)] * (settings.get_max_num_tasks() - len(x))

    hist_player_idx = torch.tensor(
        mapper(lambda x: x.get("player_idx", -1), public_history),
        dtype=torch.int8,
    )
    hist_trick = torch.tensor(
        mapper(lambda x: x.get("trick", -1), public_history), dtype=torch.int8
    )
    hist_action = torch.tensor(
        mapper(lambda x: x.get("action", (-1, -1)), public_history), dtype=torch.int8
    )
    hist_turn = torch.tensor(
        mapper(lambda x: x.get("turn", -1), public_history), dtype=torch.int8
    )
    hist_phase = torch.tensor(
        mapper(lambda x: x.get("phase", -1), public_history), dtype=torch.int8
    )

    def pad_hand(x: list[tuple]):
        # +1 to handle nosignal action in signal phase.
        assert len(x) <= settings.max_hand_size
        return x + [(-1, -1)] * (settings.max_hand_size - len(x))

    hand = torch.tensor(
        mapper(lambda x: pad_hand(x["hand"]), private_inputs), dtype=torch.int8
    )
    player_idx = torch.tensor(
        mapper(lambda x: x["player_idx"], private_inputs), dtype=torch.int8
    )
    trick = torch.tensor(mapper(lambda x: x["trick"], private_inputs), dtype=torch.int8)
    turn = torch.tensor(mapper(lambda x: x["turn"], private_inputs), dtype=torch.int8)
    phase = torch.tensor(mapper(lambda x: x["phase"], private_inputs), dtype=torch.int8)
    task_idxs = torch.tensor(
        mapper(lambda x: pad_tasks(x["task_idxs"]), private_inputs), dtype=torch.int8
    )

    def pad_valid_actions(x: list[tuple]):
        # +1 to handle nosignal action in signal phase.
        assert len(x) <= settings.max_num_actions
        return x + [(-1, -1)] * (settings.max_num_actions - len(x))

    valid_actions_arr: torch.Tensor = torch.tensor(
        mapper(pad_valid_actions, valid_actions), dtype=torch.int8
    )

    return TensorDict(
        hist=TensorDict(
            player_idx=hist_player_idx,
            trick=hist_trick,
            action=hist_action,
            turn=hist_turn,
            phase=hist_phase,
        ),
        private=TensorDict(
            hand=hand,
            player_idx=player_idx,
            trick=trick,
            turn=turn,
            phase=phase,
            task_idxs=task_idxs,
        ),
        valid_actions=valid_actions_arr,
    )
