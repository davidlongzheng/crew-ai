import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None,
        num_hidden_layers: int,
        *,
        dropout=0.0,
        use_layer_norm=False,
        activation=torch.nn.GELU,
        use_resid=False,
    ):
        super().__init__()
        self.activation = activation()
        self.dropout = nn.Dropout(dropout) if dropout else None

        fcs = []
        layer_norms = []
        resid_fcs = []

        assert num_hidden_layers >= 1

        def make_layer(inp_dim, out_dim):
            fcs.append(nn.Linear(inp_dim, out_dim))
            if use_layer_norm:
                layer_norms.append(nn.LayerNorm(out_dim))
            if use_resid:
                if inp_dim == out_dim:
                    resid_fcs.append(nn.Identity())
                else:
                    resid_fcs.append(nn.Linear(inp_dim, out_dim))

        make_layer(input_dim, hidden_dim)
        for _ in range(num_hidden_layers - 1):
            make_layer(hidden_dim, hidden_dim)

        self.fcs = nn.ModuleList(fcs)
        self.layer_norms = nn.ModuleList(layer_norms) if use_layer_norm else None
        self.resid_fcs = nn.ModuleList(resid_fcs) if use_resid else None
        self.out_fc = (
            nn.Linear(hidden_dim, output_dim) if output_dim is not None else None
        )

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            y = self.activation(fc(x))
            if self.dropout:
                y = self.dropout(y)
            if self.layer_norms:
                y = self.layer_norms[i](y)
            if self.resid_fcs:
                y = self.resid_fcs[i](x) + y
            x = y

        if self.out_fc:
            x = self.out_fc(x)

        return x


def win_rate_by_difficulty(td):
    win = td["win"].float()
    difficulty = td["difficulty"]
    unique_cats, bin_indices = torch.unique(difficulty, return_inverse=True)

    bin_sums = torch.bincount(bin_indices, weights=win)
    bin_counts = torch.bincount(bin_indices)
    bin_means = bin_sums / bin_counts

    return dict(zip(unique_cats.tolist(), bin_means.tolist()))
