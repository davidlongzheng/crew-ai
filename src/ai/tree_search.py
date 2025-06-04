import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from tensordict import TensorDict

import cpp_game
from ai.utils import get_lstm_state, set_lstm_state

if TYPE_CHECKING:
    from ai.ai import AI


class Timer:
    def __init__(self):
        self.start_times: dict[str, float] = {}
        self.times: dict[str, float | None] = {}
        self.counts: dict[str, int] = {}

    def start(self, key):
        assert key not in self.start_times
        self.start_times[key] = time.time()

    def finish(self, key):
        elapsed = time.time() - self.start_times[key]
        self.times.setdefault(key, 0.0)
        self.times[key] += elapsed
        self.counts.setdefault(key, 0)
        self.counts[key] += 1
        del self.start_times[key]

    def print(self):
        for key, elapsed in sorted(self.times.items()):
            print(
                f"{key}: elapsed={elapsed:.3f} count={self.counts[key]} avg={elapsed / self.counts[key]:.4f}"
            )


@dataclass(frozen=True)
class TreeSearchSettings:
    c_puct_base: float = 19652
    c_puct_init: float = 1.55
    num_parallel: int = 20
    root_noise: bool = True
    all_noise: bool = False
    cheating: bool = True
    noise_eps: float = 0.3
    noise_alpha: float = 0.13698
    skip_thresh: float | None = 0.98
    num_iters: int = 100
    seed: int | None = None


def uct_search(
    tree_search: cpp_game.TreeSearch,
    ts_settings: TreeSearchSettings,
    engines: list[cpp_game.Engine],
    ai: "AI",
    ai_states: list[dict],
    print_=False,
) -> tuple[list[list[cpp_game.Action]], np.ndarray, np.ndarray]:
    timer = Timer()

    assert len(engines) == len(ai_states)

    timer.start("pv")
    valid_actions_li, root_probs, root_values = ai.get_pv(engines, ai_states)
    timer.finish("pv")

    skip_mask = np.array(
        [
            (
                ts_settings.skip_thresh is not None
                and np.max(probs_i) >= ts_settings.skip_thresh
            )
            or len(valid_actions) == 1
            for valid_actions, probs_i in zip(valid_actions_li, root_probs)
        ],
        dtype=np.bool,
    )

    if np.all(skip_mask):
        return valid_actions_li, root_probs, root_values

    root_states = [engine.state for skip, engine in zip(skip_mask, engines) if not skip]
    tree_search.reset(root_states, root_probs, root_values)

    lstm_state_map = {
        i: ai_state["lstm_state"]
        for i, ai_state in enumerate(
            [x for skip, x in zip(skip_mask, ai_states) if not skip]
        )
    }

    for _ in range(ts_settings.num_iters):
        timer.start("select")
        move_inps = tree_search.select_nodes()
        timer.finish("select")
        leaf_nodes = tree_search.leaf_node_idxs
        num_leaves = len(leaf_nodes)
        if num_leaves == 0:
            continue

        timer.start("pv")
        inps = TensorDict(
            hist=TensorDict(
                player_idx=move_inps.hist_player_idx,
                trick=move_inps.hist_trick,
                action=move_inps.hist_action,
                turn=move_inps.hist_turn,
                phase=move_inps.hist_phase,
            ),
            private=TensorDict(
                hand=move_inps.hand,
                player_idx=move_inps.player_idx,
                trick=move_inps.trick,
                turn=move_inps.turn,
                phase=move_inps.phase,
                task_idxs=move_inps.task_idxs,
            ),
            valid_actions=move_inps.valid_actions,
        )
        inps.auto_batch_size_()
        inps = inps[:num_leaves]

        lstm_states = [lstm_state_map[parent_idx] for _, parent_idx in leaf_nodes]
        set_lstm_state(ai.pv_model, lstm_states)

        with torch.no_grad():
            log_probs, values, _ = ai.pv_model(inps)

        lstm_states = get_lstm_state(ai.pv_model)
        for (node_idx, _), lstm_state in zip(leaf_nodes, lstm_states):
            assert node_idx not in lstm_state_map
            lstm_state_map[node_idx] = lstm_state

        probs = np.exp(log_probs.numpy())
        values = values.numpy()
        timer.finish("pv")

        timer.start("expand")
        tree_search.expand_nodes(probs, values)
        timer.finish("expand")

    final_probs, final_values = tree_search.get_final_pv()
    final_probs = final_probs[: len(root_states)]
    final_values = final_values[: len(root_states)]

    root_probs[~skip_mask] = final_probs
    root_values[~skip_mask] = final_values
    final_probs, final_values = root_probs, root_values

    if print_:
        timer.print()

    return valid_actions_li, final_probs, final_values
