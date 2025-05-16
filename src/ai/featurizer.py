import torch
from tensordict import TensorDict

from ..game.settings import Settings
from ..lib.types import StrMap, TaskIdxs


def featurize(
    public_history: StrMap | list[StrMap] | list[list[StrMap]],
    private_inputs: StrMap | list[StrMap] | list[list[StrMap]],
    valid_actions: tuple | list[tuple] | list[list[tuple]],
    task_idxs: TaskIdxs | list[TaskIdxs] | list[list[TaskIdxs]],
    settings: Settings,
    *,
    non_feature_dims: int,
    device: torch.device | None = None,
) -> StrMap:
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

    hist_player_idxs = torch.tensor(
        mapper(lambda x: x.get("player_idx", -1), public_history),
        dtype=torch.int8,
    )
    hist_tricks = torch.tensor(
        mapper(lambda x: x.get("trick", -1), public_history), dtype=torch.int8
    )
    hist_cards = torch.tensor(
        mapper(lambda x: x.get("card", (-1, -1)), public_history), dtype=torch.int8
    )
    hist_turns = torch.tensor(
        mapper(lambda x: x.get("turn", -1), public_history), dtype=torch.int8
    )
    hist_phases = torch.tensor(
        mapper(lambda x: x.get("phase", -1), public_history), dtype=torch.int8
    )

    def pad_cards(x: list[tuple]):
        # +1 to handle nosignal action in signal phase.
        assert len(x) <= settings.max_hand_size + 1
        return x + [(-1, -1)] * (settings.max_hand_size + 1 - len(x))

    hand = torch.tensor(
        mapper(lambda x: pad_cards(x["hand"]), private_inputs), dtype=torch.int8
    )
    player_idx = torch.tensor(
        mapper(lambda x: x["player_idx"], private_inputs), dtype=torch.int8
    )
    trick = torch.tensor(mapper(lambda x: x["trick"], private_inputs), dtype=torch.int8)
    turn = torch.tensor(mapper(lambda x: x["turn"], private_inputs), dtype=torch.int8)
    phase = torch.tensor(mapper(lambda x: x["phase"], private_inputs), dtype=torch.int8)

    valid_actions_arr: torch.Tensor = torch.tensor(
        mapper(lambda x: pad_cards(x), valid_actions), dtype=torch.int8
    )

    def pad_tasks(x):
        return x + [(-1, -1)] * (settings.get_max_num_tasks() - len(x))

    # task_idxs doesn't vary across time.
    task_idxs_arr = torch.tensor(
        mapper(
            pad_tasks,
            task_idxs,
            _non_feature_dims=(1 if non_feature_dims == 2 else non_feature_dims),
        ),
        dtype=torch.int8,
    )

    return TensorDict(
        hist=TensorDict(
            player_idxs=hist_player_idxs,
            tricks=hist_tricks,
            cards=hist_cards,
            turns=hist_turns,
            phases=hist_phases,
        ),
        private=TensorDict(
            hand=hand,
            player_idx=player_idx,
            trick=trick,
            turn=turn,
            phase=phase,
        ),
        valid_actions=valid_actions_arr,
        task_idxs=task_idxs_arr,
    )
