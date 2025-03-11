import pytest

from ...game.settings import Settings
from ..rollout import rollout


def test_rollout_deterministic():
    """Test that rollout with same seed produces same results."""
    settings = Settings(
        num_players=3,
        side_suit_length=9,
        num_side_suits=4,
        trump_suit_length=4,
        use_trump_suit=True,
        use_signals=False,
        tasks=[],
    )

    seed = 42
    result1 = rollout(settings=settings, seed=seed)
    result2 = rollout(settings=settings, seed=seed)

    # Check that all components match
    assert result1["private_inputs"] == result2["private_inputs"]
    assert result1["public_history"] == result2["public_history"]
    assert result1["valid_actions"] == result2["valid_actions"]
    assert result1["targets"] == result2["targets"]
    assert result1["rewards"] == result2["rewards"]


def test_rollout_structure():
    """Test that rollout returns dictionary with expected structure and types."""
    settings = Settings(
        num_players=3,
        side_suit_length=9,
        num_side_suits=4,
        trump_suit_length=4,
        use_trump_suit=True,
        use_signals=False,
        tasks=[],
    )

    result = rollout(settings=settings, seed=42)

    # Check dictionary structure
    assert isinstance(result, dict)

    # Check types of components
    assert isinstance(result["private_inputs"], list)
    assert isinstance(result["public_history"], list)
    assert isinstance(result["valid_actions"], list)
    assert isinstance(result["targets"], list)
    assert isinstance(result["rewards"], list)

    # Check rewards length matches number of players
    assert len(result["rewards"]) == settings.num_players


def test_rollout_valid_card_tuples():
    """Test that all card tuples in the output are valid."""
    settings = Settings(
        num_players=3,
        side_suit_length=9,
        num_side_suits=4,
        trump_suit_length=4,
        use_trump_suit=True,
        use_signals=False,
        tasks=[],
    )

    result = rollout(settings=settings, seed=42)

    def is_valid_card_tuple(card_tuple: tuple) -> bool:
        if not isinstance(card_tuple, tuple) or len(card_tuple) != 2:
            return False
        rank, suit = card_tuple
        if not isinstance(rank, int) or not isinstance(suit, int):
            return False
        if suit == settings.num_side_suits:  # Trump suit
            return 0 <= rank < settings.trump_suit_length
        return (
            0 <= suit < settings.num_side_suits
            and 0 <= rank < settings.side_suit_length
        )

    # Check private inputs
    for private_input in result["private_inputs"]:
        assert all(is_valid_card_tuple(card) for card in private_input["hand"])

    # Check public actions
    for action in result["public_history"][1:]:
        assert is_valid_card_tuple(action["card"])

    # Check valid actions
    for valid_action_list in result["valid_actions"]:
        assert all(is_valid_card_tuple(card) for card in valid_action_list)

    # Check targets
    for target in result["targets"]:
        assert isinstance(target, int)


def test_rollout_rewards_range():
    """Test that rewards are within expected range."""
    settings = Settings(
        num_players=3,
        side_suit_length=9,
        num_side_suits=4,
        trump_suit_length=4,
        use_trump_suit=True,
        use_signals=False,
        tasks=[],
    )

    result = rollout(settings=settings, seed=42)

    # Since there are no tasks in this test, rewards should be 0
    assert all(reward == 0 for reward in result["rewards"])


@pytest.mark.slow
def test_rollout_timing():
    """Test that rollout() executes within reasonable time."""
    import time

    settings = Settings(
        num_players=3,
        side_suit_length=9,
        num_side_suits=4,
        trump_suit_length=4,
        use_trump_suit=True,
        use_signals=False,
        tasks=[],
    )

    num_iterations = 100
    total_time = 0.0

    for _ in range(num_iterations):
        start_time = time.perf_counter()
        rollout(settings=settings, seed=42)
        total_time += time.perf_counter() - start_time

    avg_time = total_time / num_iterations
    assert avg_time < 0.001, f"Average rollout time {avg_time:.3f}s exceeds threshold"
