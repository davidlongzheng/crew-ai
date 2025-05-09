import time
from dataclasses import replace

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence

import cpp_game

from ...game.settings import get_preset
from ..featurizer import featurize
from ..rollout import do_batch_rollout, do_batch_rollout_cpp


def pad_seq(inp, *, dtype=None):
    return pad_sequence(
        [torch.tensor(x, dtype=dtype) for x in inp],
        batch_first=True,
    )


def test_batch_rollout_cpp():
    settings = get_preset("easy_p4")
    settings = replace(settings, use_signals=True)
    cpp_settings = cpp_game.get_preset("easy_p4")
    cpp_settings.use_signals = True
    num_rollouts = 500
    batch_rollout = cpp_game.BatchRollout(cpp_settings, num_rollouts)

    for batch_seed in [42, 43]:
        start_time = time.time()
        rollouts = do_batch_rollout(settings, num_rollouts, batch_seed)
        inps = featurize(
            public_history=[x["public_history"] for x in rollouts],
            private_inputs=[x["private_inputs"] for x in rollouts],
            valid_actions=[x["valid_actions"] for x in rollouts],
            task_idxs=[x["task_idxs"] for x in rollouts],
            non_feature_dims=2,
            settings=settings,
        )
        actions = pad_seq([x["actions"] for x in rollouts])
        orig_probs = pad_seq([x["probs"] for x in rollouts], dtype=torch.float32)
        orig_log_probs = pad_seq(
            [x["log_probs"] for x in rollouts], dtype=torch.float32
        )
        rewards = pad_seq([x["rewards"] for x in rollouts], dtype=torch.float32)
        frac_success = torch.tensor(
            [
                np.sum([y[0] for y in x["num_success_tasks_pp"]])
                / np.sum([y[1] for y in x["num_success_tasks_pp"]])
                for x in rollouts
            ],
            dtype=torch.float32,
        )
        win = torch.tensor(
            [x["win"] for x in rollouts],
        )
        td = TensorDict(
            inps=inps,
            actions=actions,
            orig_probs=orig_probs,
            orig_log_probs=orig_log_probs,
            rewards=rewards,
            frac_success=frac_success,
            win=win,
        )
        print(f"Took {time.time() - start_time:.3f} to run rollouts in Python")

        start_time = time.time()
        cpp_td = do_batch_rollout_cpp(batch_rollout, batch_seed)
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
