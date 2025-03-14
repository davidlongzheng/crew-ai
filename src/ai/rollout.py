import numpy as np
import torch

from ..game.engine import Engine
from ..game.settings import Settings
from ..game.types import Card
from ..lib.types import StrMap, req
from .featurizer import featurize
from .models import PolicyModel


def do_rollout(
    settings: Settings,
    seed: int | None = None,
    policy_model: PolicyModel | None = None,
    device: torch.device | None = None,
) -> StrMap:
    """Does a roll out of one game and returns all the necessary
    inputs/actions/rewards to do training.
    """
    engine = Engine(settings=settings, seed=seed)
    rng = np.random.default_rng(seed)

    public_history: list[StrMap] = [{}]
    private_inputs: list[StrMap] = []
    valid_actions_li: list[list[tuple]] = []
    log_probs_li: list[list[float]] = []
    actions: list[int] = []

    if policy_model:
        policy_model.eval()
        policy_model.start_single_step()

    def card_to_tuple(x: Card):
        return (x.rank - 1, x.suit)

    while engine.state.phase != "end":
        # Player index relative to the captain.
        player_idx = (
            engine.state.player_turn - engine.state.captain
        ) % settings.num_players
        turn = (engine.state.player_turn - engine.state.leader) % settings.num_players
        assert engine.state.phase == "play"
        private_inputs.append(
            {
                "hand": [
                    card_to_tuple(card)
                    for card in engine.state.hands[engine.state.player_turn]
                ],
                "trick": engine.state.trick,
                "player_idx": player_idx,
                "turn": turn,
            }
        )
        valid_actions = engine.valid_actions()
        assert all(x.type == "play" for x in valid_actions)
        assert all(x.card is not None for x in valid_actions)
        valid_actions_li.append([card_to_tuple(req(x.card)) for x in valid_actions])

        if policy_model:
            inp = featurize(
                public_history[-1],
                private_inputs[-1],
                valid_actions_li[-1],
                settings,
                non_feature_dims=0,
                device=device,
            )
            with torch.no_grad():
                log_probs = policy_model(inp).to("cpu").numpy()
            probs = np.exp(log_probs)
        else:
            probs = np.full(len(valid_actions), 1.0 / len(valid_actions))
            log_probs = np.log(probs)

        probs = probs / np.sum(probs)
        action_idx = int(rng.choice(np.arange(len(probs)), size=1, p=probs)[0])

        assert action_idx < len(valid_actions)
        action = valid_actions[action_idx]
        actions.append(action_idx)
        log_probs_li.append(list(log_probs))
        public_history.append(
            {
                "trick": engine.state.trick,
                "card": card_to_tuple(req(action.card)),
                "player_idx": player_idx,
                "turn": turn,
            }
        )
        engine.move(action)

    if policy_model:
        policy_model.stop_single_step()

    public_history.pop()
    assert (
        len(valid_actions_li)
        == len(public_history)
        == len(private_inputs)
        == len(actions)
    )

    rewards_pp = []
    num_success_pp = [
        sum(x.status == "success" for x in tasks)
        for tasks in engine.state.assigned_tasks
    ]
    num_tasks_pp = [len(tasks) for tasks in engine.state.assigned_tasks]
    num_success = sum(num_success_pp)
    num_tasks = sum(num_tasks_pp)
    full_bonus = 1 if num_success == num_tasks else 0
    num_success_tasks_pp: list[tuple] = []

    for player_idx in range(settings.num_players):
        player = (engine.state.captain + player_idx) % settings.num_players
        rewards_pp.append(
            -1
            + 0.5 * num_success / max(1, num_tasks)
            + 0.5 * num_success_pp[player] / max(1, num_tasks_pp[player])
            + full_bonus
        )
        num_success_tasks_pp.append((num_success_pp[player], num_tasks_pp[player]))

    return {
        "public_history": public_history,
        "private_inputs": private_inputs,
        "valid_actions": valid_actions_li,
        "log_probs": log_probs_li,
        "actions": actions,
        "rewards_pp": rewards_pp,
        "num_success_tasks_pp": num_success_tasks_pp,
    }


def do_batch_rollout(
    settings: Settings,
    num_rollouts: int,
    seed: int | None = None,
    policy_model: PolicyModel | None = None,
    device: torch.device | None = None,
) -> list[StrMap]:
    """Does a roll out of one game and returns all the necessary
    inputs/actions/rewards to do training.
    """
    rng = np.random.default_rng(seed)
    engines = [
        Engine(settings=settings, seed=int(rng.integers(0, 100000, ())))
        for _ in range(num_rollouts)
    ]

    public_history_pr_pt: list[list[StrMap]] = [[{}] for _ in range(num_rollouts)]
    private_inputs_pr_pt: list[list[StrMap]] = [[] for _ in range(num_rollouts)]
    valid_actions_pr_pt: list[list[list[tuple]]] = [[] for _ in range(num_rollouts)]
    log_probs_pr_pt: list[list[list[float]]] = [[] for _ in range(num_rollouts)]
    actions_pr_pt: list[list[int]] = [[] for _ in range(num_rollouts)]

    if policy_model:
        policy_model.eval()
        policy_model.start_single_step()

    def card_to_tuple(x: Card):
        return (x.rank - 1, x.suit)

    while any(engine.state.phase != "end" for engine in engines):
        for rollout_idx, engine in enumerate(engines):
            if engine.state.phase == "end":
                continue
            # Player index relative to the captain.
            player_idx = (
                engine.state.player_turn - engine.state.captain
            ) % settings.num_players
            turn = (
                engine.state.player_turn - engine.state.leader
            ) % settings.num_players
            assert engine.state.phase == "play"
            private_inputs_pr_pt[rollout_idx].append(
                {
                    "hand": [
                        card_to_tuple(card)
                        for card in engine.state.hands[engine.state.player_turn]
                    ],
                    "trick": engine.state.trick,
                    "player_idx": player_idx,
                    "turn": turn,
                }
            )
            valid_actions = engine.valid_actions()
            assert all(x.type == "play" for x in valid_actions)
            assert all(x.card is not None for x in valid_actions)
            valid_actions_pr_pt[rollout_idx].append(
                [card_to_tuple(req(x.card)) for x in valid_actions]
            )

        # Purposefully create dummy entries for finished
        # samples so that the hidden state of the policy model is
        # nice and aligned.
        # The entries are discarded later.
        if policy_model:
            inps = featurize(
                [x[-1] for x in public_history_pr_pt],
                [x[-1] for x in private_inputs_pr_pt],
                [x[-1] for x in valid_actions_pr_pt],
                settings,
                non_feature_dims=1,
                device=device,
            )
            with torch.no_grad():
                log_probs_pr = policy_model(inps).to("cpu").numpy()
            probs_pr = np.exp(log_probs_pr)
        else:
            probs_pr = [
                np.full(len(x[-1]), 1.0 / len(x[-1])) for x in valid_actions_pr_pt
            ]
            log_probs_pr = [np.log(x) for x in probs_pr]

        for rollout_idx, engine in enumerate(engines):
            if engine.state.phase == "end":
                continue
            # Player index relative to the captain.
            player_idx = (
                engine.state.player_turn - engine.state.captain
            ) % settings.num_players
            turn = (
                engine.state.player_turn - engine.state.leader
            ) % settings.num_players
            probs = probs_pr[rollout_idx]
            log_probs = log_probs_pr[rollout_idx]
            probs = probs / np.sum(probs)
            action_idx = int(rng.choice(np.arange(len(probs)), size=1, p=probs)[0])
            valid_actions = engine.valid_actions()
            assert action_idx < len(valid_actions)
            action = valid_actions[action_idx]
            actions_pr_pt[rollout_idx].append(action_idx)
            log_probs_pr_pt[rollout_idx].append(list(log_probs))
            public_history_pr_pt[rollout_idx].append(
                {
                    "trick": engine.state.trick,
                    "card": card_to_tuple(req(action.card)),
                    "player_idx": player_idx,
                    "turn": turn,
                }
            )
            engine.move(action)

    if policy_model:
        policy_model.stop_single_step()

    ret: list[StrMap] = []
    for rollout_idx, engine in enumerate(engines):
        public_history_pt = public_history_pr_pt[rollout_idx]
        valid_actions_pt = valid_actions_pr_pt[rollout_idx]
        private_inputs_pt = private_inputs_pr_pt[rollout_idx]
        actions_pt = actions_pr_pt[rollout_idx]
        log_probs_pt = log_probs_pr_pt[rollout_idx]

        public_history_pt.pop()
        assert (
            len(valid_actions_pt)
            == len(public_history_pt)
            == len(private_inputs_pt)
            == len(actions_pt)
            == len(log_probs_pt)
        )

        rewards_pp = []
        num_success_pp = [
            sum(x.status == "success" for x in tasks)
            for tasks in engine.state.assigned_tasks
        ]
        num_tasks_pp = [len(tasks) for tasks in engine.state.assigned_tasks]
        num_success = sum(num_success_pp)
        num_tasks = sum(num_tasks_pp)
        full_bonus = 1 if num_success == num_tasks else 0
        num_success_tasks_pp: list[tuple] = []

        for player_idx in range(settings.num_players):
            player = (engine.state.captain + player_idx) % settings.num_players
            rewards_pp.append(
                -1
                + 0.5 * num_success / max(1, num_tasks)
                + 0.5 * num_success_pp[player] / max(1, num_tasks_pp[player])
                + full_bonus
            )
            num_success_tasks_pp.append((num_success_pp[player], num_tasks_pp[player]))

        ret.append(
            {
                "public_history": public_history_pt,
                "private_inputs": private_inputs_pt,
                "valid_actions": valid_actions_pt,
                "log_probs": log_probs_pt,
                "actions": actions_pt,
                "rewards_pp": rewards_pp,
                "num_success_tasks_pp": num_success_tasks_pp,
            }
        )
    return ret
