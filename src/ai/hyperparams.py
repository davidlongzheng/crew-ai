from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Hyperparams:
    # Training params
    num_epochs: int = 10
    batch_size: int = 8

    # For embeddings
    embed_dim: int = 16

    # For embedding a hand from a set of card embeddings
    hand_hidden_dim: int = 16
    hand_num_hidden_layers: int = 1

    # For public history LSTM
    hist_hidden_dim: int = 16
    hist_num_lstm_layers: int = 1

    # For decision MLP
    decision_hidden_dim: int = 16
    decision_num_hidden_layers: int = 2
    decision_output_dim: int = 16

    # For doing attention on the valid actions in the policy network.
    policy_query_dim: int = 16
