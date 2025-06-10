import pickle
from functools import cache
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime

import cpp_game
from ai.tree_search import TreeSearchSettings, uct_search
from game.settings import Settings


def load_ort_model(model_path):
    ort_model = onnxruntime.InferenceSession(model_path / "model.onnx")
    with open(model_path / "settings_dict.pkl", "rb") as f:
        settings_dict = pickle.load(f)
    settings = settings_dict["settings"]
    hp = settings_dict["hp"]
    return ort_model, settings, hp


def init_lstm_state(num_rollouts, hp):
    h = np.zeros(
        (hp.hist_num_layers, num_rollouts, hp.hist_hidden_dim), dtype=np.float32
    )
    c = np.zeros(
        (hp.hist_num_layers, num_rollouts, hp.hist_hidden_dim), dtype=np.float32
    )
    return h, c


def collate_lstm_state(ai_states):
    h = np.concatenate([x["lstm_state"][0] for x in ai_states], axis=1)
    c = np.concatenate([x["lstm_state"][1] for x in ai_states], axis=1)
    return h, c


def featurize(
    engines: list[cpp_game.Engine],
    featurizer: cpp_game.Featurizer,
    ai_states: list[dict],
):
    featurizer.reset()
    assert len(engines) <= featurizer.num_rollouts
    for engine in engines:
        featurizer.record_move_inputs(engine)
    move_inps = featurizer.get_move_inputs()
    h, c = collate_lstm_state(ai_states)

    num_rollouts = len(engines)
    ort_inps = {
        "hist_player_idx": move_inps.hist_player_idx[:num_rollouts],
        "hist_trick": move_inps.hist_trick[:num_rollouts],
        "hist_action": move_inps.hist_action[:num_rollouts],
        "hist_turn": move_inps.hist_turn[:num_rollouts],
        "hist_phase": move_inps.hist_phase[:num_rollouts],
        "hand": move_inps.hand[:num_rollouts],
        "player_idx": move_inps.player_idx[:num_rollouts],
        "trick": move_inps.trick[:num_rollouts],
        "turn": move_inps.turn[:num_rollouts],
        "phase": move_inps.phase[:num_rollouts],
        "task_idxs": move_inps.task_idxs[:num_rollouts],
        "valid_actions": move_inps.valid_actions[:num_rollouts],
        "h0": h,
        "c0": c,
    }
    return ort_inps


class AI:
    def __init__(
        self,
        path: Path,
        ts_settings: TreeSearchSettings | None = None,
        num_rollouts: int = 1,
    ):
        self.ort_model, self.settings, self.hp = load_ort_model(path)
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
            "lstm_state": init_lstm_state(1, self.hp),
        }

    def get_pv(self, engines: list[cpp_game.Engine], ai_states: list[dict], timer=None):
        assert len(engines) <= self.num_rollouts
        assert len(engines) == len(ai_states)
        inps = featurize(engines, self.featurizer, ai_states)
        valid_actions_li = [engine.valid_actions() for engine in engines]

        if timer:
            timer.start("forward")
        log_probs, values, h, c = self.ort_model.run(None, inps)
        if timer:
            timer.finish("forward")

        for i, ai_state in enumerate(ai_states):
            ai_state["lstm_state"] = (h[:, i : i + 1], c[:, i : i + 1])

        probs = np.exp(log_probs)

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
        and (4 <= cast(int, settings.min_difficulty) <= 14)
        and settings.max_num_tasks == 8
    ):
        for path in [
            Path("/Users/davidzheng/projects/crew-ai/crew-ai/models/v0"),
            Path("/app/models/v0"),
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
