from collections import defaultdict
from pathlib import Path

import click
import torch

import cpp_game
from ai.actor import BatchActor, ModelBatchActor
from ai.models import get_models
from ai.rollout import do_batch_rollout_cpp
from game.settings import DEFAULT_PRESET, SETTINGS_TYPE, Settings, get_preset
from game.tasks import Task, get_task_defs
from game.utils import calc_trick_winner, split_by_suit, to_action, to_hand


def hand_to_str(hand, settings):
    hand = to_hand(hand, settings)
    return " | ".join(
        " ".join(str(card) for card in sub_hand) for sub_hand in split_by_suit(hand)
    )


def print_rollout(rollout, settings, skip_hands):
    seq_length = settings.get_seq_length()

    task_defs = get_task_defs(settings.bank)

    task_idxs = rollout["inps", "private", "task_idxs"][-1]
    pidx_to_tasks = defaultdict(list)
    for task_idx, pidx in task_idxs:
        task_def = task_defs[task_idx.item()]
        pidx_to_tasks[pidx.item()].append(Task(*task_def, task_idx=task_idx))

    print("Tasks:")
    for pidx, tasks in sorted(pidx_to_tasks.items(), key=lambda x: x[0]):
        print(f"P{pidx}: {' '.join(f'{x.task_idx}({x})' for x in tasks)}")

    for i in range(0, seq_length, settings.num_players):
        phase_idx = rollout["inps", "private", "phase"][i].item()
        phase = settings.get_phase(phase_idx)
        trick = rollout["inps", "private", "trick"][i].item()
        leader = rollout["inps", "private", "player_idx"][i].item()

        print("-" * 50)
        print(f"Phase: {phase} Trick: {trick}")
        print()

        if not skip_hands:
            pidx_to_hand = {}
            for j in range(settings.num_players):
                hand = rollout["inps", "private", "hand"][i + j]
                pidx = rollout["inps", "private", "player_idx"][i + j].item()
                pidx_to_hand[pidx] = hand_to_str(hand, settings)

            for pidx, hand in sorted(pidx_to_hand.items(), key=lambda x: x[0]):
                print(f"{'* ' if pidx == leader else ''}P{pidx}: {hand}")
            print()

        active_cards = []
        action_strs = []
        for j in range(settings.num_players):
            action_idx = rollout["actions"][i + j].item()
            probs = [x.item() for x in torch.exp(rollout["orig_log_probs"][i + j])]
            pidx = rollout["inps", "private", "player_idx"][i + j].item()

            valid_actions = [x for x in rollout["inps", "valid_actions"][i + j]]
            action = to_action(valid_actions[action_idx], phase_idx, pidx, settings)
            probs_and_actions = [
                (prob, a) for prob, a in zip(probs, valid_actions) if prob >= 0.1
            ]
            probs_and_actions = sorted(
                probs_and_actions, reverse=True, key=lambda x: x[0]
            )[:3]
            prob_str = " ".join(
                f"{to_action(a, phase_idx, pidx, settings).short_str()}={prob:.2f}"
                for prob, a in probs_and_actions
            )
            if phase == "play":
                active_cards.append((action.card, pidx))
            action_strs.append(f"{action} ({prob_str})")
        print(" ".join(action_strs))
        print()

        if phase == "play":
            trick_winner = calc_trick_winner(active_cards)
            print(f"Winner: P{trick_winner}")

    win = rollout["win"].item()
    print("-" * 50)
    print(f"We {'won' if win else 'lost'}!")


@click.command()
@click.option(
    "--outdir",
    type=Path,
    help="Outdir. If unset, use random actions.",
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
    "--num-success-examples",
    type=int,
    help="Num success examples",
    default=0,
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
@click.option(
    "--skip-hands",
    is_flag=True,
    help="Skip printing hands",
)
def main(
    outdir: Path | None,
    settings: Settings,
    use_cache: bool,
    num_rollouts: int,
    num_success_examples: int,
    num_failed_examples: int,
    batch_seed: int,
    skip_hands: bool,
) -> None:
    if outdir:
        state_dict = torch.load(outdir / "checkpoint.pth", weights_only=False)
        settings_dict = torch.load(outdir / "settings.pth", weights_only=False)
        hp = settings_dict["hp"]
        # Override settings with the ones from the checkpoint.
        settings = settings_dict["settings"]
        pv_model = get_models(hp, settings)["pv"]
        pv_model.load_state_dict(state_dict["pv_model"])
        actor: BatchActor | None = ModelBatchActor(pv_model)
        argmax = True
    else:
        actor = None
        argmax = False

    if use_cache:
        assert outdir is not None
        td = state_dict["td"]
    else:
        cpp_settings = settings.to_cpp()
        batch_rollout = cpp_game.BatchRollout(cpp_settings, num_rollouts)
        td = do_batch_rollout_cpp(
            batch_rollout, batch_seed=batch_seed, actor=actor, argmax=argmax
        )

    N = len(td)
    win_rate = td["win"].float().mean()
    mean_reward = td["rewards"].sum(dim=-1).mean()
    frac_success = td["task_success"].float().mean(dim=1).mean()
    print(
        f"num_rollouts: {N}, win_rate: {win_rate:.3f}, mean_reward: {mean_reward:.3f}, frac_success: {frac_success:.3f}"
    )

    for status in ["success", "fail"]:
        num_examples = (
            num_success_examples if status == "success" else num_failed_examples
        )
        examples = td[td["win"] == (1 if status == "success" else 0)][:num_examples]
        for i, rollout in enumerate(examples):
            print("=" * 50)
            print(f"{'Success' if status == 'success' else 'Failed'} example {i}:")
            print_rollout(rollout, settings, skip_hands)


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
