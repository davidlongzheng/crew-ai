import torch
from torch import nn

from ..game.settings import Settings
from .hyperparams import Hyperparams


class HistoryModel(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_dim = hidden_dim

        self.state = None

    def reset_state(self):
        self.state = None
        pass

    def forward(self, x):
        if len(x.shape) != 1 and self.state is not None:
            raise Exception(
                "Forgot to reset_state() before going back to batched mode."
            )

        x, state = self.lstm(x, self.state)
        # Only save state if we are using the model one action
        # at a time.
        if len(x.shape) == 1:
            self.state = state
        return x


class DecisionModel(nn.Module):
    def __init__(
        self,
        hist_model: HistoryModel,
        embed_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
    ):
        super().__init__()
        self.hist_model = hist_model
        self.output_dim = output_dim
        fcs = [
            nn.Linear(self.hist_model.hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 1):
            fcs += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        fcs.append(nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Sequential(*fcs)

    def reset_state(self):
        self.hist_model.reset_state()

    def forward(self, hist_inp, private_inp):
        if hist_inp is None:
            hist_embed = torch.zeros(
                private_inp.shape[:-1] + (self.hist_model.hidden_dim,)
            )
        else:
            hist_embed = self.hist_model(hist_inp)
            if len(hist_embed.shape) > 1:
                hist_embed = nn.functional.pad(hist_embed, (0, 0, 1, 0))

        x = torch.cat([hist_embed, private_inp], axis=-1)
        x = self.fc(x)
        return x


class PolicyModel(nn.Module):
    def __init__(
        self,
        decision_model: DecisionModel,
        embed_dim: int,
        query_dim: int,
    ):
        super().__init__()
        self.decision_model = decision_model
        self.query_model = nn.Linear(self.decision_model.output_dim, query_dim)
        self.key_model = nn.Linear(embed_dim, query_dim)
        self.query_dim = query_dim

    def reset_state(self):
        self.decision_model.reset_state()

    def forward(self, hist_inp, private_inp, valid_actions_inp):
        decision_embed = self.decision_model(hist_inp, private_inp)
        query = self.query_model(decision_embed)
        key = self.key_model(valid_actions_inp)

        attn_score = torch.einsum("...q,...vq->...v", query, key) / self.query_dim**0.5
        probs = nn.functional.softmax(attn_score, dim=-1)

        return probs


class ValueModel(nn.Module):
    def __init__(
        self,
        decision_model: DecisionModel,
    ):
        super().__init__()
        self.decision_model = decision_model
        self.fc = nn.Linear(self.decision_model.output_dim, 1)

    def forward(self, hist_inp, private_inp):
        decision_embed = self.decision_model(hist_inp, private_inp)
        x = self.fc(decision_embed)
        x = x.squeeze(-1)

        return x


def get_models(
    hp: Hyperparams,
    settings: Settings,
) -> dict[str, nn.Module]:
    hist_model = HistoryModel(hp.embed_dim, hp.hist_hidden_dim, hp.hist_num_lstm_layers)
    decision_model = DecisionModel(
        hist_model,
        hp.embed_dim,
        hp.decision_hidden_dim,
        hp.decision_num_hidden_layers,
        hp.decision_output_dim,
    )
    policy_model = PolicyModel(
        decision_model,
        embed_dim=hp.embed_dim,
        query_dim=hp.policy_query_dim,
    )
    value_model = ValueModel(decision_model)
    return {
        "hist": hist_model,
        "decision": decision_model,
        "policy": policy_model,
        "value": value_model,
    }
