import functools
import random
from typing import override

import gym
import numpy as np
from gym.spaces import Box, Dict, Discrete
from gym.wrappers import OrderEnforcing
from torchrl.envs import GymWrapper

from ..game.engine import Engine
from ..game.settings import Settings
from ..game.types import Card
from ..lib.types import StrMap, req

ENV_NAME = "crew-v1"


def card_to_arr(x: Card):
    return [x.rank - 1, x.suit]


def get_torchrl_env(device=None, **kwargs):
    return GymWrapper(
        env=CrewEnv(**kwargs),
        device=device,
    )


def get_env(**kwargs):
    return OrderEnforcing(CrewEnv(**kwargs))


class CrewEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self, settings: Settings, render_mode=None, randomize_invalid_actions=True
    ):
        super().__init__()
        self.settings = settings

        self.render_mode = render_mode
        self.randomize_invalid_actions = randomize_invalid_actions

        self.engine = Engine(settings=settings)
        self.public_history: StrMap = {}

        self.observation_space = Dict(
            {
                "public_history": Dict(
                    {
                        "trick": Box(
                            low=np.array(-1),
                            high=np.array(self.settings.num_tricks - 1),
                            dtype=np.int8,
                        ),
                        "card": Box(
                            low=np.array(
                                [-1, -1],
                            ),
                            high=np.array(
                                [
                                    self.settings.num_ranks - 1,
                                    self.settings.num_suits - 1,
                                ]
                            ),
                            dtype=np.int8,
                        ),
                        "player_idx": Box(
                            low=np.array(-1),
                            high=np.array(self.settings.num_players - 1),
                            dtype=np.int8,
                        ),
                        "turn": Box(
                            low=np.array(-1),
                            high=np.array(self.settings.num_players - 1),
                            dtype=np.int8,
                        ),
                    }
                ),
                "private_inputs": Dict(
                    {
                        "hand": Box(
                            low=np.array(
                                [[-1, -1]] * self.settings.max_hand_size,
                            ),
                            high=np.array(
                                [
                                    [
                                        self.settings.num_ranks - 1,
                                        self.settings.num_suits - 1,
                                    ]
                                ]
                                * self.settings.max_hand_size
                            ),
                            dtype=np.int8,
                        ),
                        "trick": Box(
                            low=np.array(-1),
                            high=np.array(self.settings.num_tricks - 1),
                            dtype=np.int8,
                        ),
                        "player_idx": Box(
                            low=np.array(-1),
                            high=np.array(self.settings.num_players - 1),
                            dtype=np.int8,
                        ),
                        "turn": Box(
                            low=np.array(-1),
                            high=np.array(self.settings.num_players - 1),
                            dtype=np.int8,
                        ),
                    }
                ),
                "valid_actions": Box(
                    low=np.array(
                        [[-1, -1]] * self.settings.max_hand_size,
                    ),
                    high=np.array(
                        [
                            [
                                self.settings.num_ranks - 1,
                                self.settings.num_suits - 1,
                            ]
                        ]
                        * self.settings.max_hand_size
                    ),
                    dtype=np.int8,
                ),
                "action_mask": Box(
                    low=0, high=1, shape=(self.settings.max_hand_size,), dtype=np.int8
                ),
            }
        )
        self.action_space = Discrete(self.settings.max_hand_size)
        self.info: StrMap = {}

    @override
    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        return str(self.engine.state)

    @property
    def cur_player_idx(self):
        return (
            self.engine.state.player_turn - self.engine.state.captain
        ) % self.settings.num_players

    @property
    def cur_turn(self):
        return (
            self.engine.state.player_turn - self.engine.state.leader
        ) % self.settings.num_players

    @property
    def cur_hand(self):
        hand = self.engine.state.hands[self.engine.state.player_turn]
        return [card_to_arr(card) for card in hand] + [self.dummy_card] * (
            self.settings.max_hand_size - len(hand)
        )

    @property
    def cur_valid_actions(self):
        valid_actions = self.engine.valid_actions()
        assert all(x.type == "play" for x in valid_actions)
        assert all(x.card is not None for x in valid_actions)
        return [card_to_arr(req(x.card)) for x in valid_actions] + [self.dummy_card] * (
            self.settings.max_hand_size - len(valid_actions)
        )

    @functools.cached_property
    def dummy_card(self):
        return [-1, -1]

    @functools.cached_property
    def dummy_hand(self):
        return [self.dummy_card] * self.settings.max_hand_size

    @functools.cached_property
    def dummy_public_history(self):
        return {
            "trick": -1,
            "card": self.dummy_card,
            "player_idx": -1,
            "turn": -1,
        }

    @functools.cached_property
    def dummy_obs(self):
        private_inputs = {
            "hand": self.dummy_hand,
            "trick": -1,
            "player_idx": -1,
            "turn": -1,
        }
        action_mask = np.zeros(self.settings.max_hand_size, dtype=np.int8)
        return {
            "public_history": self.dummy_public_history,
            "private_inputs": private_inputs,
            "valid_actions": self.dummy_hand,
            "action_mask": action_mask,
        }

    def get_obs(self):
        if self.engine.state.phase == "end":
            return self.dummy_obs

        assert self.engine.state.phase == "play"

        private_inputs = {
            "hand": self.cur_hand,
            "trick": self.engine.state.trick,
            "player_idx": self.cur_player_idx,
            "turn": self.cur_turn,
        }
        valid_actions = self.cur_valid_actions
        action_mask = np.array(
            [x[0] != -1 for x in valid_actions],
            dtype=np.int8,
        )
        return {
            "public_history": self.public_history,
            "private_inputs": private_inputs,
            "valid_actions": valid_actions,
            "action_mask": action_mask,
        }

    @override
    def close(self):
        pass

    @override
    def reset(self, seed=None, options=None):
        self.engine.reset_state(seed=seed)
        self.public_history = self.dummy_public_history
        self.info = {"num_invalid_actions": 0}

        if self.render_mode == "human":
            self.render()

        return self.get_obs(), self.info

    def get_reward(self):
        assert self.engine.state.phase == "end"
        num_success_pp = [
            sum(x.status == "success" for x in tasks)
            for tasks in self.engine.state.assigned_tasks
        ]
        num_tasks_pp = [len(tasks) for tasks in self.engine.state.assigned_tasks]
        num_success = sum(num_success_pp)
        num_tasks = sum(num_tasks_pp)
        full_bonus = num_tasks if num_success_pp == num_tasks else 0

        self.info["num_success_tasks"] = []
        self.info["num_tasks"] = []
        for player_idx in range(self.settings.num_players):
            player = (
                self.engine.state.captain + player_idx
            ) % self.settings.num_players
            self.info["num_success_tasks"].append(num_success_pp[player])
            self.info["num_tasks"].append([player])

        return num_success + full_bonus

    @override
    def step(self, action):
        assert self.engine.state.phase == "play"
        valid_actions = self.engine.valid_actions()
        if self.randomize_invalid_actions and action >= len(valid_actions):
            self.info["num_invalid_actions"] += 1
            action = random.randint(0, len(valid_actions) - 1)

        action_obj = valid_actions[action]

        self.public_history = {
            "trick": self.engine.state.trick,
            "card": card_to_arr(req(action_obj.card)),
            "player_idx": self.cur_player_idx,
            "turn": self.cur_turn,
        }
        self.engine.move(action_obj)

        if self.engine.state.phase == "end":
            terminated = True
            reward = self.get_reward()
        else:
            terminated = False
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        return self.get_obs(), reward, terminated, False, self.info
