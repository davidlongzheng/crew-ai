import functools
from typing import override

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from torchrl.envs import PettingZooWrapper

from ..game.engine import Engine
from ..game.settings import Settings
from ..game.types import Card
from ..lib.types import StrMap, req

ENV_NAME = "crew-v1"


def card_to_arr(x: Card):
    return [x.rank - 1, x.suit]


def get_torchrl_env(settings, render_mode=None):
    return PettingZooWrapper(
        env=env(settings, render_mode=render_mode),
        use_mask=True,
        group_map={"players": [f"player_{i}" for i in range(settings.num_players)]},
    )


def env(settings, render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(settings, render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": ENV_NAME}

    def __init__(self, settings: Settings, render_mode=None):
        super().__init__()
        self.possible_agents = [f"player_{i}" for i in range(settings.num_players)]
        self.player_idx_map = {x: i for i, x in enumerate(self.possible_agents)}
        self.settings = settings

        self.engine = Engine(settings=settings)
        self.public_history: list[StrMap] = []
        self.render_mode = render_mode

    @override
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict(
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

    @override
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.settings.max_hand_size)

    @override
    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

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

    @override
    def observe(self, agent):
        if self.engine.state.phase == "end":
            return self.dummy_obs

        assert self.engine.state.phase == "play"
        if self.player_idx_map[agent] != self.cur_player_idx:
            return self.dummy_obs

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
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[self.cur_player_idx]

    def on_game_end(self):
        num_success_pp = [
            sum(x.status == "success" for x in tasks)
            for tasks in self.engine.state.assigned_tasks
        ]
        num_tasks_pp = [len(tasks) for tasks in self.engine.state.assigned_tasks]
        num_success = sum(num_success_pp)
        num_tasks = sum(num_tasks_pp)
        full_bonus = num_tasks if num_success_pp == num_tasks else 0

        for agent, info in self.infos.items():
            player_idx = self.player_idx_map[agent]
            player = (
                self.engine.state.captain + player_idx
            ) % self.settings.num_players
            self.rewards[agent] = (
                0.5 * num_success + 0.5 * num_success_pp[player] + full_bonus
            )
            info["num_success_tasks"] = num_success_pp[player]
            info["num_tasks"] = num_tasks_pp[player]

        self.terminations = {agent: True for agent in self.agents}

    @override
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        assert self.player_idx_map[agent] == self.cur_player_idx
        assert self.engine.state.phase == "play"

        action_obj = self.engine.valid_actions()[action]

        self.public_history = {
            "trick": self.engine.state.trick,
            "card": card_to_arr(req(action_obj.card)),
            "player_idx": self.cur_player_idx,
            "turn": self.cur_turn,
        }
        self.engine.move(action_obj)

        if self.engine.state.phase == "end":
            self.on_game_end()
            self.agent_selection = self.agents[0]
        else:
            self.agent_selection = self.agents[self.cur_player_idx]

        self._cumulative_rewards[self.agent_selection] = 0

        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()
