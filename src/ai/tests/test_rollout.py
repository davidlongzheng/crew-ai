import time
from dataclasses import replace

import numpy as np

import cpp_game

from ...game.settings import get_preset
from ..rollout import do_batch_rollout, do_batch_rollout_cpp


def test_batch_rollout_cpp():
    settings = get_preset("easy_p4")
    settings = replace(settings, use_signals=True)
    cpp_settings = cpp_game.get_preset("easy_p4")
    cpp_settings.use_signals = True
    num_rollouts = 1_000
    batch_seed = 42

    start_time = time.time()
    rollouts = do_batch_rollout(settings, num_rollouts, batch_seed)
    print(f"Took {time.time() - start_time:.3f} to run rollouts in Python")
    start_time = time.time()
    cpp_rollouts = do_batch_rollout_cpp(cpp_settings, num_rollouts, batch_seed)
    print(f"Took {time.time() - start_time:.3f} to run rollouts in C++")

    assert len(rollouts) == len(cpp_rollouts)

    for rollout, cpp_rollout in zip(rollouts, cpp_rollouts):
        for k in [
            "public_history",
            "private_inputs",
            "valid_actions",
            "actions",
            "num_success_tasks_pp",
            "task_idxs",
            "win",
        ]:
            assert rollout[k] == cpp_rollout[k], (rollout[k], cpp_rollout[k])

        for k in [
            "probs",
            "log_probs",
            "rewards",
        ]:
            if isinstance(rollout[k][0], list):
                assert len(rollout[k]) == len(cpp_rollout[k])
                for x, y in zip(rollout[k], cpp_rollout[k]):
                    assert np.isclose(x, y).all(), (x, y)
            else:
                assert np.isclose(rollout[k], cpp_rollout[k]).all()
