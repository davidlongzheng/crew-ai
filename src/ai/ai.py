from functools import cache
from pathlib import Path
from typing import cast

import numpy as np
import torch

from ai.featurizer import featurize
from ai.models import load_model_for_eval
from game.engine import Engine
from game.settings import Settings
from game.types import Action
from game.utils import encode_action, encode_hand, encode_tasks


class AI:
    def __init__(self, path: Path):
        self.pv_model, _ = load_model_for_eval(path)

    def new_rollout(self):
        return {
            "public_history": {},
            "state": None,
        }

    def record_move(self, engine: Engine, action: Action, ai_state: dict):
        player_idx = engine.state.get_player_idx()
        turn = engine.state.get_turn()
        ai_state["public_history"] = {
            "trick": engine.state.trick,
            "action": encode_action(action, engine.settings),
            "player_idx": player_idx,
            "turn": turn,
            "phase": engine.settings.get_phase_idx(engine.state.phase),
            "task_idxs": encode_tasks(engine.state),
        }

    def get_pv(self, engine: Engine, ai_state: dict):
        player_idx = engine.state.get_player_idx()
        turn = engine.state.get_turn()
        private_inputs = {
            "hand": encode_hand(
                engine.state.hands[engine.state.cur_player], engine.settings
            ),
            "trick": engine.state.trick,
            "player_idx": player_idx,
            "turn": turn,
            "phase": engine.settings.get_phase_idx(engine.state.phase),
            "task_idxs": encode_tasks(engine.state),
        }
        valid_actions = engine.valid_actions()
        valid_actions_arr = [encode_action(x, engine.settings) for x in valid_actions]

        inps = featurize(
            ai_state["public_history"],
            private_inputs,
            valid_actions_arr,
            engine.settings,
            non_feature_dims=0,
        )
        # Add batch dim
        inps = inps.unsqueeze(0)
        self.pv_model.set_state(ai_state["state"])
        with torch.no_grad():
            log_probs, value, _ = self.pv_model(inps)
        ai_state["state"] = self.pv_model.get_state()

        probs = np.exp(log_probs.numpy()[0, : len(valid_actions)])

        return valid_actions, probs, value

    def get_move(self, engine: Engine, ai_state: dict):
        valid_actions, probs, _ = self.get_pv(engine, ai_state)
        action_idx = int(np.argmax(probs))
        action = valid_actions[action_idx]

        return action

    def get_move_uct(self, engine: Engine, ai_state: dict):
        return


@cache
def _get_ai_by_path(path) -> AI:
    return AI(path)


@cache
def get_ai(settings: Settings) -> AI:
    if (
        settings.num_players == 4
        and settings.use_drafting
        and not settings.use_signals
        and settings.bank == "med"
        and settings.min_difficulty == settings.max_difficulty
        and (4 <= cast(int, settings.min_difficulty) <= 7)
        and settings.max_num_tasks == 4
    ):
        return _get_ai_by_path(
            Path("/Users/davidzheng/projects/crew-ai/outdirs/0522/run_0")
        )
    else:
        raise ValueError(f"Unsupported settings for AI: {settings}")
