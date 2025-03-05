from __future__ import annotations

import click

from app.engine import Engine
from app.settings import Settings
from app.tasks import Task
from app.types import Action
from app.utils import split_by_suit


def display_game_state(engine: Engine) -> None:
    """Display the current game state to the user."""
    state = engine.state
    click.echo()
    click.echo("=" * 80)
    click.echo(f"\nPhase: {state.phase} | Trick: {state.trick}")

    # Show tasks and their status
    click.echo("\nTasks:")
    for player, tasks in enumerate(state.assigned_tasks):
        for task in tasks:
            click.echo(f"P{player}: {task.formula} - {task.status}")

    # Show active cards
    if state.active_cards:
        click.echo("\nActive cards:")
        for card, player in state.active_cards:
            click.echo(f"P{player}: {card}")

    # Show current player's hand
    click.echo(f"\nYour hand (P{state.player_turn}):")
    hand = state.hands[state.player_turn]
    click.echo(
        " | ".join(
            " ".join(str(card) for card in sub_hand) for sub_hand in split_by_suit(hand)
        )
    )

    # Show signals if in signal phase
    if state.phase == "signal":
        click.echo("\nSignals:")
        for player, signal in enumerate(state.signals):
            if signal:
                click.echo(f"P{player}: {signal.card} ({signal.value})")


def get_user_action(engine: Engine) -> Action:
    """Get the next action from the user."""
    valid_actions = engine.valid_actions()

    click.echo("\nValid actions:")
    for i, action in enumerate(valid_actions):
        click.echo(f"{i}: {action}")

    while True:
        try:
            choice = click.prompt("Choose action number", type=int)
            if 0 <= choice < len(valid_actions):
                return valid_actions[choice]
            click.echo("Invalid choice. Please try again.")
        except ValueError:
            click.echo("Please enter a valid number.")


@click.command()
@click.option("--num-players", default=4, help="Number of players")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option(
    "--use-signals/--no-signals", default=False, help="Enable/disable signals"
)
@click.option("--num-side-suits", default=4, help="Number of side suits")
@click.option("--side-suit-length", default=9, help="Length of each side suit")
@click.option("--trump-suit-length", default=4, help="Length of the trump suit")
@click.argument("tasks", nargs=-1)
def play(
    num_players: int,
    seed: int | None,
    use_signals: bool,
    num_side_suits: int,
    side_suit_length: int,
    trump_suit_length: int,
    tasks: list[str],
) -> None:
    """Play The Crew card game through CLI."""
    # Hardcode a list of tasks for now.
    settings = Settings(
        num_players=num_players,
        use_signals=use_signals,
        num_side_suits=num_side_suits,
        side_suit_length=side_suit_length,
        trump_suit_length=trump_suit_length,
        tasks=[Task(f, "") for f in tasks],
    )

    engine = Engine(settings=settings, seed=seed)

    click.echo("Welcome to The Crew!")
    click.echo(f"Playing with {num_players} players")

    while engine.state.phase != "end":
        display_game_state(engine)
        action = get_user_action(engine)
        try:
            engine.move(action)
        except Exception as e:
            click.echo(f"Error: {e}")
            continue

    # Game ended, show final results
    click.echo()
    click.echo("=" * 80)
    click.echo("\nGame Over!")
    for player, player_tasks in enumerate(engine.state.assigned_tasks):
        click.echo(f"\nPlayer {player} tasks:")
        for task in player_tasks:
            click.echo(f"- {task.formula} - {task.status}")

    # Check if all tasks were successful
    click.echo(
        f"\n{'Mission Successful!' if engine.state.status == 'success' else 'Mission Failed!'}"
    )


if __name__ == "__main__":
    play()
