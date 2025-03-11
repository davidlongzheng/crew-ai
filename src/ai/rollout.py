import numpy as np
from torch import nn

from ..game.engine import Engine
from ..game.settings import Settings
from ..game.types import Card
from ..lib.types import StrMap, req
from .featurizer import featurize
from .models import PolicyModel


def rollout(
    settings: Settings,
    seed: int | None = None,
    embed_models: dict[str, nn.Module] | None = None,
    policy_model: PolicyModel | None = None,
) -> StrMap:
    """Does a roll out of one game and returns all the necessary
    input/target/rewards to do training.
    """
    engine = Engine(settings=settings, seed=seed)
    rng = np.random.default_rng(seed)

    public_history: list[StrMap] = []
    private_inputs: list[StrMap] = []
    valid_actions_li: list[list[tuple]] = []
    targets: list[int] = []

    if policy_model:
        policy_model.reset_state()

    def card_to_tuple(x: Card):
        return (x.rank - 1, x.suit)

    while engine.state.phase != "end":
        # Player index relative to the captain.
        player_idx = (
            engine.state.player_turn - engine.state.captain
        ) % settings.num_players
        assert engine.state.phase == "play"
        private_inputs.append(
            {
                "hand": [
                    card_to_tuple(card)
                    for card in engine.state.hands[engine.state.player_turn]
                ],
                "trick": engine.state.trick,
                "player_idx": player_idx,
            }
        )
        valid_actions = engine.valid_actions()
        assert all(x.type == "play" for x in valid_actions)
        assert all(x.card is not None for x in valid_actions)
        valid_actions_li.append([card_to_tuple(req(x.card)) for x in valid_actions])

        if policy_model:
            assert embed_models is not None
            inp = featurize(
                embed_models,
                public_history=(public_history[-1] if public_history else None),
                private_inputs=private_inputs[-1],
                valid_actions=valid_actions_li[-1],
                non_feature_dims=0,
            )
            action_probs = policy_model(*inp)
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
                "last_in_trick": (engine.state.player_turn + 1) % settings.num_players
                == engine.state.leader,
                "card": card_to_tuple(req(action.card)),
                "player_idx": player_idx,
            }
        )
        engine.move(action)

    if policy_model:
        policy_model.reset_state()

    public_history.pop()
    assert (
        len(valid_actions_li)
        == len(public_history) + 1
        == len(private_inputs)
        == len(targets)
    )

    rewards = []
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
        rewards.append(0.5 * num_success + 0.5 * num_success_pp[player] + full_bonus)
        num_success_tasks.append((num_success_pp[player], num_tasks_pp[player]))

    return {
        "private_inputs": private_inputs,
        "public_history": public_history,
        "valid_actions": valid_actions_li,
        "targets": targets,
        "rewards": rewards,
        "num_success_tasks": num_success_tasks,
    }
