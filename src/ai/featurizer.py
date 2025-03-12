import torch
from torch.nn.utils.rnn import pad_sequence

from ..game.settings import Settings
from ..lib.types import StrMap


def featurize(
    public_history: StrMap | list[StrMap] | list[list[StrMap]],
    private_inputs: StrMap | list[StrMap] | list[list[StrMap]],
    valid_actions: tuple | list[tuple] | list[list[tuple]],
    settings: Settings,
    *,
    non_feature_dims: int,
    device: torch.device | None = None,
) -> StrMap:
    """Takes in a set of nn inputs and featurizes the inputs.

    Inputs to the func are either F, (T, F), or (N, T, F).
    You should set non_feature_dims=0,1,2 respectively.
    """

    assert 0 <= non_feature_dims <= 2

    def mapper(func, inp):
        if non_feature_dims == 0:
            ret = func(inp)
        elif non_feature_dims == 1:
            ret = [func(x) for x in inp]
        else:
            ret = [[func(y) for y in x] for x in inp]

        return ret

    seq_lengths = (
        torch.tensor([len(x) for x in public_history], dtype=torch.int8, device=device)
        if non_feature_dims == 2
        else None
    )

    def pad_seq(inp, *, dtype):
        if non_feature_dims == 2:
            return pad_sequence(
                [torch.tensor(x, dtype=dtype, device=device) for x in inp],
                batch_first=True,
            )
        return torch.tensor(inp, dtype=dtype, device=device)

    hist_player_idxs = pad_seq(
        mapper(lambda x: x.get("player_idx", -1), public_history),
        dtype=torch.int8,
    )
    hist_tricks = pad_seq(
        mapper(lambda x: x.get("trick", -1), public_history), dtype=torch.int8
    )
    hist_cards = pad_seq(
        mapper(lambda x: x.get("card", (-1, -1)), public_history), dtype=torch.int8
    )
    hist_turns = pad_seq(
        mapper(lambda x: x.get("turn", -1), public_history), dtype=torch.int8
    )

    def pad_cards(x: list[tuple]):
        assert len(x) <= settings.max_hand_size
        return x + [(-1, -1)] * (settings.max_hand_size - len(x))

    hand = pad_seq(
        mapper(lambda x: pad_cards(x["hand"]), private_inputs), dtype=torch.int8
    )
    player_idx = pad_seq(
        mapper(lambda x: x["player_idx"], private_inputs), dtype=torch.int8
    )
    trick = pad_seq(mapper(lambda x: x["trick"], private_inputs), dtype=torch.int8)
    turn = pad_seq(mapper(lambda x: x["turn"], private_inputs), dtype=torch.int8)

    valid_actions_arr: torch.Tensor = pad_seq(
        mapper(lambda x: pad_cards(x), valid_actions), dtype=torch.int8
    )

    return dict(
        hist=dict(
            player_idxs=hist_player_idxs,
            tricks=hist_tricks,
            cards=hist_cards,
            turns=hist_turns,
        ),
        private=dict(
            hand=hand,
            player_idx=player_idx,
            trick=trick,
            turn=turn,
        ),
        valid_actions=valid_actions_arr,
        seq_lengths=seq_lengths,
    )
