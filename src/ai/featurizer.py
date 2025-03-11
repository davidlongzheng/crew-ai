import numpy as np
import torch
from torch import nn

from ..lib.types import StrMap


def featurize(
    embed_models: dict[str, nn.Module],
    *,
    public_history: None | StrMap | list[StrMap] | list[list[StrMap]],
    private_inputs: StrMap | list[StrMap] | list[list[StrMap]],
    valid_actions: tuple | list[tuple] | list[list[tuple]],
    non_feature_dims: int,
) -> tuple[torch.Tensor | None, ...]:
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

    def max_length_per_depth(inp, *, depth=0, depth_counts=None):
        if depth_counts is None:
            depth_counts = {}

        if isinstance(inp, (list, tuple)):
            depth_counts[depth] = max(depth_counts.get(depth, 0), len(inp))
            for item in inp:
                max_length_per_depth(item, depth=depth + 1, depth_counts=depth_counts)

        return [depth_counts[d] for d in sorted(depth_counts)]

    def pad(inp, shape, pad_value, *, depth=0):
        if not isinstance(inp, (list, tuple)):
            return inp

        arr = [pad(x, shape, pad_value, depth=depth + 1) for x in inp]
        assert len(arr) <= shape[depth]
        pad_len = shape[depth] - len(arr)
        if pad_len > 0:
            arr += [
                np.full(shape[depth + 1 :], fill_value=pad_value).tolist()
            ] * pad_len

        return arr

    def padded_tensor(inp, *, dtype, pad_value=-1):
        shape = max_length_per_depth(inp)
        arr = pad(inp, shape, pad_value)
        return torch.tensor(arr, dtype=dtype)

    hist_inp = None
    if public_history is not None:
        hist_player_idxs = padded_tensor(
            mapper(lambda x: x["player_idx"], public_history), dtype=torch.int8
        )
        hist_tricks = padded_tensor(
            mapper(lambda x: x["trick"], public_history), dtype=torch.int8
        )
        hist_cards = padded_tensor(
            mapper(lambda x: x["card"], public_history), dtype=torch.int8
        )
        hist_last_in_trick = padded_tensor(
            mapper(lambda x: x["last_in_trick"], public_history), dtype=torch.float16
        )
        hist_player_embed = embed_models["player"](hist_player_idxs)
        hist_trick_embed = embed_models["trick"](hist_tricks)
        hist_card_embed = embed_models["card"](hist_cards)
        assert (
            hist_player_embed.shape == hist_trick_embed.shape == hist_card_embed.shape
        )
        hist_embed = hist_player_embed + hist_trick_embed + hist_card_embed
        hist_inp = torch.cat([hist_embed, hist_last_in_trick.unsqueeze(-1)], dim=-1)

    hand = padded_tensor(mapper(lambda x: x["hand"], private_inputs), dtype=torch.int8)
    player_idx = padded_tensor(
        mapper(lambda x: x["player_idx"], private_inputs), dtype=torch.int8
    )
    trick = padded_tensor(
        mapper(lambda x: x["trick"], private_inputs), dtype=torch.int8
    )
    hand_embed = embed_models["hand"](hand)
    player_embed = embed_models["player"](player_idx)
    trick_embed = embed_models["player"](trick)
    assert hand_embed.shape == player_embed.shape == trick_embed.shape
    private_inp = hand_embed + player_embed + trick_embed

    valid_actions_inp = padded_tensor(
        mapper(lambda x: x, valid_actions), dtype=torch.int8
    )

    return (hist_inp, private_inp, valid_actions_inp)
