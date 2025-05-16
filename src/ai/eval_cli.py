from collections import defaultdict
from pathlib import Path

import click
import torch

import cpp_game
from src.ai.actor import BatchActor, GreedyBatchActor, ModelBatchActor
from src.ai.models import get_models
from src.ai.rollout import do_batch_rollout_cpp
from src.game.settings import DEFAULT_PRESET, SETTINGS_TYPE, Settings, get_preset

from ..game.tasks import Task, get_task_defs
from ..game.utils import calc_trick_winner, split_by_suit, to_card, to_hand


def hand_to_str(hand, settings):
    hand = to_hand(hand, settings)
    return " | ".join(
        " ".join(str(card) for card in sub_hand) for sub_hand in split_by_suit(hand)
    )


def print_rollout(rollout, settings):
    seq_length = settings.get_seq_length()

    task_defs = get_task_defs(settings.bank)

    task_idxs = rollout["inps", "task_idxs"]
    pidx_to_tasks = defaultdict(list)
    for task_idx, pidx in task_idxs:
        task_def = task_defs[task_idx]
        pidx_to_tasks[pidx].append(Task(*task_def, task_idx=task_idx))

    print("Tasks:")
    for pidx, tasks in sorted(pidx_to_tasks.items()):
        print(f"P{pidx}: {' '.join(map(str, tasks))}")

    num_tricks_won = {pidx: 0 for pidx in range(settings.num_players)}

    i = 0
    while i < seq_length:
        trick = rollout["inps", "private", "trick"][i].item()
        actions_in_trick = 1
        while (
            i + actions_in_trick < seq_length
            and rollout["inps", "private", "trick"][i + actions_in_trick].item()
            == trick
        ):
            actions_in_trick += 1
        leader = rollout["inps", "private", "player_idx"][i].item()

        print("-" * 50)
        print(f"Trick: {trick}")
        print()

        pidx_to_hand = {}
        for j in range(settings.num_players):
            hand = rollout["inps", "private", "hand"][i + j]
            pidx = rollout["inps", "private", "player_idx"][i + j].item()
            pidx_to_hand[pidx] = hand_to_str(hand, settings)

        for pidx, hand in sorted(pidx_to_hand.items()):
            print(f"{'* ' if pidx == leader else ''}P{pidx}: {hand}")
        print()

        active_cards = []
        action_strs = []
        prev_phase = -1
        for j in range(actions_in_trick):
            action_idx = rollout["actions"][i + j].item()
            probs = torch.exp(rollout["orig_log_probs"][i + j])
            valid_actions = rollout["inps", "valid_actions"][i + j]
            action = valid_actions[action_idx]
            probs = [(prob, a) for prob, a in zip(probs, valid_actions) if prob >= 0.1]
            probs = sorted(probs, reverse=True)[:3]
            prob_str = " ".join(
                f"{to_card(a, settings)}={prob:.2f}" for prob, a in probs
            )
            phase = rollout["inps", "private", "phase"][i + j].item()
            if phase != prev_phase:
                prev_phase = phase
                action_strs.append([])
            pidx = rollout["inps", "private", "player_idx"][i + j].item()
            verb = "plays" if phase == 0 else "signals"
            card = to_card(action, settings)
            active_cards.append((card, pidx))
            action_strs[-1].append(f"P{pidx} {verb} {card} ({prob_str}).")
        print("\n".join(" ".join(x) for x in action_strs))
        print()

        trick_winner = calc_trick_winner(active_cards)
        num_tricks_won[trick_winner] += 1

        print(
            f"Winner: P{trick_winner} Tricks won: {' '.join(f'P{pidx}={n}' for pidx, n in sorted(num_tricks_won.items()))}"
        )
        i += actions_in_trick

    win = rollout["win"].item()
    print("-" * 50)
    print(f"We {'won' if win else 'lost'}!")


@click.command()
@click.option(
    "--outdir",
    type=Path,
    help="Outdir. If unset, use greedy actor.",
)
@click.option(
    "--settings",
    type=SETTINGS_TYPE,
    help="Settings",
    default=get_preset(DEFAULT_PRESET),
)
@click.option(
    "--use-cache",
    is_flag=True,
    help="Use cache instead of recomputing rollouts.",
)
@click.option(
    "--num-rollouts",
    type=int,
    help="Num rollouts",
    default=500,
)
@click.option(
    "--num-failed-examples",
    type=int,
    help="Num failed examples",
    default=0,
)
@click.option(
    "--batch-seed",
    type=int,
    help="Batch seed",
    default=42,
)
def main(
    outdir: Path | None,
    settings: Settings,
    use_cache: bool,
    num_rollouts: int,
    num_failed_examples: int,
    batch_seed: int,
) -> None:
    if outdir:
        state_dict = torch.load(outdir / "checkpoint.pth", weights_only=False)
        settings_dict = torch.load(outdir / "settings.pth", weights_only=False)
        hp = settings_dict["hp"]
        # Override settings with the ones from the checkpoint.
        settings = settings_dict["settings"]
        pv_model = get_models(hp, settings)["pv"]
        pv_model.load_state_dict(state_dict["pv_model"])
        actor: BatchActor = ModelBatchActor(pv_model)
    else:
        actor = GreedyBatchActor(settings, num_rollouts=num_rollouts)

    if use_cache:
        assert outdir is not None
        td = state_dict["td"]
    else:
        cpp_settings = settings.to_cpp()
        batch_rollout = cpp_game.BatchRollout(cpp_settings, num_rollouts)
        td = do_batch_rollout_cpp(
            batch_rollout, batch_seed=batch_seed, actor=actor, argmax=True
        )

    N = len(td)
    win_rate = td["win"].float().mean()
    mean_reward = td["rewards"].sum(dim=-1).mean()
    frac_success = td["frac_success"].mean()
    print(
        f"num_rollouts: {N}, win_rate: {win_rate:.3f}, mean_reward: {mean_reward:.3f}, frac_success: {frac_success:.3f}"
    )

    failed_examples = td[td["win"] == 0][:num_failed_examples]
    for i, rollout in enumerate(failed_examples):
        print("=" * 50)
        print(f"Failed example {i}:")
        print_rollout(rollout, settings)


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
