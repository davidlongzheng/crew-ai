import random

import numpy as np
import torch
from tensordict import TensorDict

import cpp_game

from ..game.engine import Engine
from ..game.settings import Settings
from ..game.types import Card
from ..lib.types import StrMap, TaskIdxs
from .aux_info import AuxInfoTracker
from .featurizer import featurize
from .models import PolicyValueModel


def do_batch_rollout(
    settings: Settings,
    num_rollouts: int,
    batch_seed: int | None = None,
    engine_seeds: int | list[int] | None = None,
    pv_model: PolicyValueModel | None = None,
    device: torch.device | None = None,
    argmax: bool = False,
) -> list[StrMap]:
    """Does a roll out of one game and returns all the necessary
    inputs/actions/rewards to do training.
    """
    if engine_seeds is None:
        engine_seed_rng = random.Random(batch_seed)
        engine_seeds = [
            engine_seed_rng.randint(0, 100_000_000) for _ in range(num_rollouts)
        ]
    elif isinstance(engine_seeds, list):
        engine_seed_rng = random.Random(batch_seed)
        engine_seeds = engine_seed_rng.sample(engine_seeds, num_rollouts)
    else:
        assert isinstance(engine_seeds, int)
        engine_seeds = [engine_seeds for _ in range(num_rollouts)]

    engines = [Engine(settings=settings, seed=seed) for seed in engine_seeds]
    policy_rng = np.random.default_rng(batch_seed)
    public_history_pr_pt: list[list[StrMap]] = [[{}] for _ in range(num_rollouts)]
    private_inputs_pr_pt: list[list[StrMap]] = [[] for _ in range(num_rollouts)]
    valid_actions_pr_pt: list[list[list[tuple]]] = [[] for _ in range(num_rollouts)]
    log_probs_pr_pt: list[list[list[float]]] = [[] for _ in range(num_rollouts)]
    actions_pr_pt: list[list[int]] = [[] for _ in range(num_rollouts)]
    rewards_pr_pt: list[list[float]] = [[] for _ in range(num_rollouts)]
    task_idxs_pr: list[TaskIdxs] = [[] for _ in range(num_rollouts)]
    # Literally just duplicates of task_idxs_pr
    task_idxs_pr_pt: list[list[TaskIdxs]] = [[] for _ in range(num_rollouts)]
    aux_tracker_pr: list[AuxInfoTracker] = [
        AuxInfoTracker(settings, engine) for engine in engines
    ]

    # Task assignment
    for rollout_idx, engine in enumerate(engines):
        for player, tasks in enumerate(engine.state.assigned_tasks):
            player_idx = engine.state.get_player_idx(player)
            task_idxs_pr[rollout_idx] += [(task.task_idx, player_idx) for task in tasks]

    if pv_model:
        pv_model.eval()
        pv_model.start_single_step()

    def card_to_tuple(x: Card | None):
        if x is None:
            return (settings.max_suit_length, settings.num_suits)
        return (x.rank - 1, settings.get_suit_idx(x.suit))

    def encode_hand(hand: list[Card]):
        return [card_to_tuple(x) for x in hand]

    while any(engine.state.phase != "end" for engine in engines):
        # Before running policy model
        for rollout_idx, engine in enumerate(engines):
            if engine.state.phase == "end":
                continue
            # Player index relative to the captain.
            player_idx = engine.state.get_player_idx()
            turn = engine.state.get_turn()
            assert engine.state.phase in ["play", "signal"]
            private_inputs_pr_pt[rollout_idx].append(
                {
                    "hand": encode_hand(engine.state.hands[engine.state.cur_player]),
                    "trick": engine.state.trick,
                    "player_idx": player_idx,
                    "turn": turn,
                    "phase": engine.state.phase_idx,
                }
            )
            valid_actions = engine.valid_actions()
            if engine.state.phase == "play":
                assert all(
                    x.type == "play" and x.card is not None for x in valid_actions
                )
            else:
                assert all(
                    (x.type == "signal" and x.card is not None)
                    or (x.type == "nosignal" and x.card is None)
                    for x in valid_actions
                )
            valid_actions_pr_pt[rollout_idx].append(
                [card_to_tuple(x.card) for x in valid_actions]
            )
            task_idxs_pr_pt[rollout_idx].append(task_idxs_pr[rollout_idx])
            aux_tracker_pr[rollout_idx].on_decision()

        # Purposefully create dummy entries for finished
        # samples so that the hidden state of the policy model is
        # nice and aligned.
        # The entries are discarded later.
        if pv_model:
            inps = featurize(
                [x[-1] for x in public_history_pr_pt],
                [x[-1] for x in private_inputs_pr_pt],
                [x[-1] for x in valid_actions_pr_pt],
                [x[-1] for x in task_idxs_pr_pt],
                settings,
                non_feature_dims=1,
                device=device,
            )
            with torch.no_grad():
                log_probs_pr, _, _ = pv_model(inps)
                log_probs_pr = log_probs_pr.to("cpu").numpy()
        else:
            log_probs_pr = [
                np.array(
                    [np.log(1.0 / len(x[-1]))] * len(x[-1])
                    + [-float("inf")] * (settings.max_hand_size + 1 - len(x[-1]))
                )
                for x in valid_actions_pr_pt
            ]

        # After running policy model
        for rollout_idx, engine in enumerate(engines):
            if engine.state.phase == "end":
                continue
            # Player index relative to the captain.
            player_idx = engine.state.get_player_idx()
            turn = engine.state.get_turn()
            log_probs = log_probs_pr[rollout_idx]
            probs = np.exp(log_probs)
            probs = probs / np.sum(probs)
            if argmax:
                action_idx = int(np.argmax(probs))
            else:
                action_idx = int(
                    policy_rng.choice(np.arange(len(probs)), size=1, p=probs)[0]
                )
            valid_actions = engine.valid_actions()
            assert action_idx < len(valid_actions)
            action = valid_actions[action_idx]
            actions_pr_pt[rollout_idx].append(action_idx)
            log_probs_pr_pt[rollout_idx].append(list(log_probs))
            public_history_pr_pt[rollout_idx].append(
                {
                    "trick": engine.state.trick,
                    "card": card_to_tuple(action.card),
                    "player_idx": player_idx,
                    "turn": turn,
                    "phase": engine.state.phase_idx,
                }
            )
            reward = engine.move(action)
            rewards_pr_pt[rollout_idx].append(reward)
            aux_tracker_pr[rollout_idx].on_move(action)

    if pv_model:
        pv_model.stop_single_step()

    ret: list[StrMap] = []
    for rollout_idx, engine in enumerate(engines):
        public_history_pt = public_history_pr_pt[rollout_idx]
        valid_actions_pt = valid_actions_pr_pt[rollout_idx]
        private_inputs_pt = private_inputs_pr_pt[rollout_idx]
        actions_pt = actions_pr_pt[rollout_idx]
        log_probs_pt = log_probs_pr_pt[rollout_idx]
        task_idxs_pt = task_idxs_pr_pt[rollout_idx]
        rewards_pt = rewards_pr_pt[rollout_idx]

        public_history_pt.pop()
        assert (
            len(valid_actions_pt)
            == len(public_history_pt)
            == len(private_inputs_pt)
            == len(actions_pt)
            == len(log_probs_pt)
            == len(task_idxs_pt)
            == len(rewards_pt)
        )

        win = engine.state.status == "success"
        num_success_pp = [
            sum(x.status == "success" for x in tasks)
            for tasks in engine.state.assigned_tasks
        ]
        num_tasks_pp = [len(tasks) for tasks in engine.state.assigned_tasks]
        num_success_tasks_pp: list[tuple] = []
        for player_idx in range(settings.num_players):
            player = engine.state.get_player(player_idx)
            num_success_tasks_pp.append((num_success_pp[player], num_tasks_pp[player]))

        aux_tracker_pr[rollout_idx].on_game_end()
        aux_info_pt = aux_tracker_pr[rollout_idx].get_aux_info()
        assert len(private_inputs_pt) == len(aux_info_pt)

        ret.append(
            {
                "public_history": public_history_pt,
                "private_inputs": private_inputs_pt,
                "valid_actions": valid_actions_pt,
                "log_probs": log_probs_pt,
                "actions": actions_pt,
                "rewards": rewards_pt,
                "num_success_tasks_pp": num_success_tasks_pp,
                "task_idxs": task_idxs_pt,
                "aux_infos": aux_info_pt,
                "win": win,
            }
        )

    return ret


def do_batch_rollout_cpp(
    batch_rollout: cpp_game.BatchRollout,
    batch_seed: int | None = None,
    engine_seeds: int | list[int] | None = None,
    pv_model: PolicyValueModel | None = None,
    device: torch.device | None = None,
    argmax: bool = False,
) -> TensorDict:
    """Does a roll out of one game and returns all the necessary
    inputs/actions/rewards to do training.
    """
    num_rollouts = batch_rollout.num_rollouts

    if engine_seeds is None:
        engine_seed_rng = random.Random(batch_seed)
        engine_seeds = [
            engine_seed_rng.randint(0, 100_000_000) for _ in range(num_rollouts)
        ]
    elif isinstance(engine_seeds, list):
        engine_seed_rng = random.Random(batch_seed)
        engine_seeds = engine_seed_rng.sample(engine_seeds, num_rollouts)
    else:
        assert isinstance(engine_seeds, int)
        engine_seeds = [engine_seeds for _ in range(num_rollouts)]

    batch_rollout.reset_state(engine_seeds)
    policy_rng = np.random.default_rng(batch_seed)

    if pv_model:
        pv_model.eval()
        pv_model.start_single_step()

    while not batch_rollout.is_done():
        move_inps = batch_rollout.get_move_inputs()

        # Purposefully create dummy entries for finished
        # samples so that the hidden state of the policy model is
        # nice and aligned.
        # The entries are discarded later.
        if pv_model:
            inps = TensorDict(
                hist=TensorDict(
                    player_idxs=move_inps.hist_player_idxs,
                    tricks=move_inps.hist_tricks,
                    cards=move_inps.hist_cards,
                    turns=move_inps.hist_turns,
                    phases=move_inps.hist_phases,
                ),
                private=TensorDict(
                    hand=move_inps.hand,
                    player_idx=move_inps.player_idx,
                    trick=move_inps.trick,
                    turn=move_inps.turn,
                    phase=move_inps.phase,
                ),
                valid_actions=move_inps.valid_actions,
                task_idxs=move_inps.task_idxs,
            )
            inps = inps.to(device)
            with torch.no_grad():
                log_probs_pr, _, _ = pv_model(inps)
                log_probs_pr = log_probs_pr.to("cpu").numpy()
        else:
            valid_moves = move_inps.valid_actions[:, :, 0] >= 0
            num_valid = np.sum(valid_moves, axis=1, keepdims=True)
            log_probs_pr = np.where(
                valid_moves,
                np.log(1.0 / num_valid),
                -float("inf"),
            )

        probs_pr = np.exp(log_probs_pr)
        probs_pr /= probs_pr.sum(axis=1, keepdims=True)
        if argmax:
            action_idxs = np.argmax(probs_pr, axis=1)
        else:
            action_idxs = np.array(
                [policy_rng.choice(len(row), p=row) for row in probs_pr]
            )

        batch_rollout.move(action_idxs, log_probs_pr)

    if pv_model:
        pv_model.stop_single_step()

    results = batch_rollout.get_results()
    ret = TensorDict(
        inps=TensorDict(
            hist=TensorDict(
                player_idxs=results.hist_player_idxs,
                tricks=results.hist_tricks,
                cards=results.hist_cards,
                turns=results.hist_turns,
                phases=results.hist_phases,
            ),
            private=TensorDict(
                hand=results.hand,
                player_idx=results.player_idx,
                trick=results.trick,
                turn=results.turn,
                phase=results.phase,
            ),
            valid_actions=results.valid_actions,
            task_idxs=results.task_idxs,
        ),
        orig_log_probs=results.log_probs,
        actions=results.actions,
        rewards=results.rewards,
        frac_success=results.frac_success,
        win=results.win,
    )
    ret = ret.to(device)
    ret.auto_batch_size_()

    return ret
