import numpy as np
import torch

from ..game.engine import Engine
from ..game.settings import Settings
from ..game.types import Card
from ..lib.types import StrMap, req
from .featurizer import featurize
from .models import PolicyValueModel


def do_rollout(
    settings: Settings,
    seed: int | None = None,
    pv_model: PolicyValueModel | None = None,
    device: torch.device | None = None,
) -> StrMap:
    """Does a roll out of one game and returns all the necessary
    input/target/rewards to do training.
    """
    engine = Engine(settings=settings, seed=seed)
    rng = np.random.default_rng(seed)

    public_history: list[StrMap] = [{}]
    private_inputs: list[StrMap] = []
    valid_actions_li: list[list[tuple]] = []
    targets: list[int] = []

    if pv_model:
        pv_model.reset_state()

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

        if pv_model:
            inp = featurize(
                public_history[-1],
                private_inputs[-1],
                valid_actions_li[-1],
                settings,
                non_feature_dims=0,
                device=device,
            )
            with torch.no_grad():
                action_probs, _ = pv_model(inp)
            action_idx = int(
                rng.choice(np.arange(len(action_probs)), size=1, p=action_probs)[0]
            )
        else:
            action_idx = int(rng.choice(np.arange(len(valid_actions)), size=1)[0])

        assert action_idx < len(valid_actions)
        action = valid_actions[action_idx]
        targets.append(action_idx)
        public_history.append(
            {
                "trick": engine.state.trick,
                "card": card_to_tuple(req(action.card)),
                "player_idx": player_idx,
                "turn": turn,
            }
        )
        engine.move(action)

    if pv_model:
        pv_model.reset_state()

    public_history.pop()
    assert (
        len(valid_actions_li)
        == len(public_history)
        == len(private_inputs)
        == len(targets)
    )

    rewards_pp = []
    num_success_pp = [
        sum(x.status == "success" for x in tasks)
        for tasks in engine.state.assigned_tasks
    ]
    num_tasks_pp = [len(tasks) for tasks in engine.state.assigned_tasks]
    num_success = sum(num_success_pp)
    num_tasks = sum(num_tasks_pp)
    full_bonus = num_tasks if num_success_pp == num_tasks else 0
    num_success_tasks: list[tuple] = []

    for player_idx in range(settings.num_players):
        player = (engine.state.captain + player_idx) % settings.num_players
        rewards_pp.append(0.5 * num_success + 0.5 * num_success_pp[player] + full_bonus)
        num_success_tasks.append((num_success_pp[player], num_tasks_pp[player]))

    return {
        "public_history": public_history,
        "private_inputs": private_inputs,
        "valid_actions": valid_actions_li,
        "targets": targets,
        "rewards_pp": rewards_pp,
        "num_success_tasks": num_success_tasks,
    }
