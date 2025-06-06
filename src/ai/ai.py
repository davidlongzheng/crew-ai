from functools import cache
from pathlib import Path
from typing import cast

import numpy as np
import torch

import cpp_game
from ai.featurizer import featurize_cpp
from ai.models import load_model_for_eval
from ai.tree_search import TreeSearchSettings
from ai.utils import get_lstm_state, set_lstm_state
from game.settings import Settings


class AI:
    def __init__(
        self,
        path: Path,
        ts_settings: TreeSearchSettings | None = None,
        num_rollouts: int = 1,
    ):
        self.pv_model, self.settings = load_model_for_eval(path)
        cpp_settings = self.settings.to_cpp()
        self.num_rollouts = num_rollouts
        self.featurizer = cpp_game.Featurizer(cpp_settings, num_rollouts)
        self.ts_settings = ts_settings or TreeSearchSettings()
        self.tree_search = cpp_game.TreeSearch(
            self.settings.to_cpp(),
            num_rollouts=num_rollouts,
            c_puct_base=self.ts_settings.c_puct_base,
            c_puct_init=self.ts_settings.c_puct_init,
            num_parallel=self.ts_settings.num_parallel,
            root_noise=self.ts_settings.root_noise,
            all_noise=self.ts_settings.all_noise,
            cheating=self.ts_settings.cheating,
            noise_eps=self.ts_settings.noise_eps,
            noise_alpha=self.ts_settings.noise_alpha,
            seed=self.ts_settings.seed or -1,
        )

    def new_rollout(self):
        return {
            "lstm_state": None,
        }

    def get_pv(self, engines: list[cpp_game.Engine], ai_states: list[dict], timer=None):
        assert len(engines) <= self.num_rollouts
        assert len(engines) == len(ai_states)
        inps = featurize_cpp(engines, self.featurizer)
        valid_actions_li = [engine.valid_actions() for engine in engines]
        set_lstm_state(self.pv_model, [x["lstm_state"] for x in ai_states])

        if timer:
            timer.start("forward")
        with torch.no_grad():
            log_probs, values, _ = self.pv_model(inps)
        if timer:
            timer.finish("forward")

        for ai_state, lstm_state in zip(ai_states, get_lstm_state(self.pv_model)):
            ai_state["lstm_state"] = lstm_state

        probs = np.exp(log_probs.numpy())
        values = values.numpy()

        return valid_actions_li, probs, values

    def get_moves(self, engines: list[cpp_game.Engine], ai_states: list[dict]):
        assert len(engines) <= self.num_rollouts
        assert len(engines) == len(ai_states)
        valid_actions_li, probs, _ = self.get_pv(engines, ai_states)

        ret = []
        for valid_actions, probs_i in zip(valid_actions_li, probs):
            action_idx = int(np.argmax(probs_i))
            try:
                action = valid_actions[action_idx]
            except IndexError:
                raise ValueError(f"Invalid action index: {action_idx} {probs}")

            ret.append(action)
        return ret

    def get_move(self, engine: cpp_game.Engine, ai_state: dict):
        return self.get_moves([engine], [ai_state])[0]

    def get_pv_tree_search(self, engines: list[cpp_game.Engine], ai_states: list[dict]):
        assert len(engines) <= self.num_rollouts
        assert len(engines) == len(ai_states)
        from ai.tree_search import uct_search

        valid_actions_li, probs, values = uct_search(
            self.tree_search, self.ts_settings, engines, self, ai_states
        )

        return valid_actions_li, probs, values

    def get_moves_tree_search(
        self, engines: list[cpp_game.Engine], ai_states: list[dict]
    ):
        assert len(engines) <= self.num_rollouts
        assert len(engines) == len(ai_states)
        valid_actions_li, probs, _ = self.get_pv_tree_search(engines, ai_states)

        ret = []
        for valid_actions, probs_i in zip(valid_actions_li, probs):
            action_idx = int(np.argmax(probs_i))
            try:
                action = valid_actions[action_idx]
            except IndexError:
                raise ValueError(f"Invalid action index: {action_idx} {probs}")

            ret.append(action)

        return ret


@cache
def _get_ai_by_path(
    path, ts_settings: TreeSearchSettings | None = None, num_rollouts: int = 1
) -> AI:
    return AI(path, ts_settings=ts_settings, num_rollouts=num_rollouts)


@cache
def get_ai(
    settings: Settings,
    ts_settings: TreeSearchSettings | None = None,
    num_rollouts: int = 1,
) -> AI:
    if (
        settings.num_players == 4
        and settings.use_drafting
        and not settings.use_signals
        and settings.bank == "all"
        and settings.min_difficulty == settings.max_difficulty
        and (4 <= cast(int, settings.min_difficulty) <= 7)
        and settings.max_num_tasks == 8
    ):
        for path in [
            Path("/Users/davidzheng/projects/crew-ai/outdirs/0531/run_16"),
        ]:
            if path.exists():
                return _get_ai_by_path(path, ts_settings, num_rollouts)
        raise ValueError("Paths do not exist")
    else:
        raise ValueError(f"Unsupported settings for AI: {settings}")


def batch_rollout(engines, ai, seeds=None, use_tree_search=False):
    assert seeds is None or len(engines) == len(seeds)
    for i, engine in enumerate(engines):
        engine.reset_state(seeds[i] if seeds else None)

    ai_states = [ai.new_rollout() for _ in engines]

    while engines[0].state.phase != cpp_game.Phase.end:
        assert all(engine.state.phase != cpp_game.Phase.end for engine in engines)
        if use_tree_search:
            actions = ai.get_moves_tree_search(engines, ai_states)
        else:
            actions = ai.get_moves(engines, ai_states)
        for action, engine in zip(actions, engines):
            engine.move(action)

    return np.array(
        [engine.state.status == cpp_game.Status.success for engine in engines]
    )
