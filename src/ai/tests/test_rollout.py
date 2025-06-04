import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import cpp_game
from ai.featurizer import featurize
from ai.rollout import do_batch_rollout, do_batch_rollout_cpp
from game.settings import DEFAULT_PRESET, get_preset


@pytest.mark.parametrize("use_drafting", [False, True])
@pytest.mark.parametrize("signal_mode", ["no", "yes", "single", "cheating"])
@pytest.mark.parametrize("use_difficulty_distro", [True, False])
def test_batch_rollout_cpp(signal_mode, use_drafting, use_difficulty_distro):
    kwargs = {}
    if signal_mode == "no":
        kwargs = {"use_signals": False, "single_signal": False}
    elif signal_mode == "yes":
        kwargs = {"use_signals": True, "single_signal": False}
    elif signal_mode == "single":
        kwargs = {"use_signals": True, "single_signal": True}
    elif signal_mode == "cheating":
        kwargs = {"use_signals": True, "single_signal": False, "cheating_signal": True}

    kwargs["use_drafting"] = use_drafting

    settings = get_preset(DEFAULT_PRESET)
    if use_difficulty_distro:
        num_difficulties = settings.max_difficulty - settings.min_difficulty + 1
        probs = np.arange(num_difficulties)
        probs = probs / probs.sum()
        kwargs["difficulty_distro"] = tuple(probs.tolist())
    settings = replace(settings, **kwargs)
    cpp_settings = settings.to_cpp()
    num_rollouts = 250
    batch_rollout = cpp_game.BatchRollout(cpp_settings, num_rollouts)

    for batch_seed in [42, 43]:
        start_time = time.time()
        rollouts = do_batch_rollout(settings, num_rollouts, batch_seed)
        inps = featurize(
            public_history=[x["public_history"] for x in rollouts],
            private_inputs=[x["private_inputs"] for x in rollouts],
            valid_actions=[x["valid_actions"] for x in rollouts],
            non_feature_dims=2,
            settings=settings,
        )
        actions = torch.tensor([x["actions"] for x in rollouts])
        orig_log_probs = torch.tensor(
            [x["log_probs"] for x in rollouts], dtype=torch.float32
        )
        rewards = torch.tensor([x["rewards"] for x in rollouts], dtype=torch.float32)
        task_idxs_no_pt = torch.tensor(
            [x["task_idxs_no_pt"] for x in rollouts],
            dtype=torch.int8,
        )
        task_success = torch.tensor(
            [x["task_success"] for x in rollouts],
            dtype=torch.bool,
        )
        difficulty = torch.tensor(
            [x["difficulty"] for x in rollouts],
            dtype=torch.int8,
        )
        win = torch.tensor(
            [x["win"] for x in rollouts],
        )
        td = TensorDict(
            inps=inps,
            actions=actions,
            orig_log_probs=orig_log_probs,
            rewards=rewards,
            task_idxs_no_pt=task_idxs_no_pt,
            task_success=task_success,
            difficulty=difficulty,
            win=win,
        )
        td.auto_batch_size_()
        print(f"Took {time.time() - start_time:.3f} to run rollouts in Python")

        start_time = time.time()
        cpp_td = do_batch_rollout_cpp(batch_rollout, batch_seed)
        del cpp_td["aux_info"]
        print(f"Took {time.time() - start_time:.3f} to run rollouts in C++")

        to_process = [(td, cpp_td)]
        while to_process:
            cur_td, cur_cpp_td = to_process.pop()
            assert sorted(cur_td.keys()) == sorted(cur_cpp_td.keys()), (
                sorted(cur_td.keys()),
                sorted(cur_cpp_td.keys()),
            )

            for k, v in cur_td.items():
                cpp_v = cur_cpp_td[k]
                if isinstance(v, TensorDict):
                    to_process.append((v, cpp_v))
                elif isinstance(v, torch.Tensor):
                    assert torch.isclose(v, cpp_v).all()
                else:
                    assert np.allclose(v, cpp_v)


def test_rollout_draft():
    settings = get_preset(DEFAULT_PRESET)
    do_batch_rollout(
        settings,
        num_rollouts=1,
        batch_seed=42,
    )
