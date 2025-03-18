from collections import OrderedDict

import torch
from torch import nn


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    *,
    dropout=0.0,
    use_layer_norm=False,
    activation=torch.nn.GELU,
) -> nn.Sequential:
    layer_idx = 0
    inps = OrderedDict()
    assert num_hidden_layers >= 1

    def make_layer(inp_dim, out_dim):
        nonlocal layer_idx
        inps[f"fc{layer_idx}"] = nn.Linear(inp_dim, out_dim)
        inps[f"active{layer_idx}"] = activation()
        if dropout:
            inps[f"dropout{layer_idx}"] = nn.Dropout(dropout)
        if use_layer_norm:
            inps[f"layer_norm{layer_idx}"] = nn.LayerNorm(out_dim)
        layer_idx += 1

    make_layer(input_dim, hidden_dim)
    for _ in range(num_hidden_layers - 1):
        make_layer(hidden_dim, hidden_dim)
    inps[f"fc{layer_idx}"] = nn.Linear(hidden_dim, output_dim)

    return nn.Sequential(inps)
