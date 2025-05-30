# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""A much faster MCTS implementation for AlphaZero.
Where we use Numpy arrays to store node statistics,
and create child node on demand.


This implementation is adapted from the Minigo project developed by Google.
https://github.com/tensorflow/minigo



The positions are evaluated from the current player (or to move) perspective.

        A           Black to move

    B       C       White to move

  D   E             Black to move

For example, in the above two-player, zero-sum games search tree. 'A' is the root node,
and when the game is in state corresponding to node 'A', it's black's turn to move.
However the children nodes of 'A' are evaluated from white player's perspective.
So if we select the best child for node 'A', without further consideration,
we'd be actually selecting the best child for white player, which is not what we want.

Let's look at an simplified example where we don't consider number of visits and total values,
just the raw evaluation scores, if the evaluated scores (from white's perspective)
for 'B' and 'C' are 0.8 and 0.3 respectively. Then according to these results,
the best child of 'A' max(0.8, 0.3) is 'B', however this is done from white player's perspective.
But node 'A' represents black's turn to move, so we need to select the best child from black player's perspective,
which should be 'C' - the worst move for white, thus a best move for black.

One way to resolve this issue is to always switching the signs of the child node's Q values when we select the best child.

For example:
    ucb_scores = -node.child_Q() + node.child_U()

In this case, a max(-0.8, -0.3) will give us the correct results for black player when we select the best child for node 'A'.

"""

import collections
import copy
import math
from typing import Any

import numpy as np

from ai.ai import AI
from ai.determinization import sample_determinization
from game.engine import Engine
from game.types import Action


class DummyNode(object):
    """A place holder to make computation possible for the root node."""

    def __init__(self):
        self.parent = None
        self.child_W = collections.defaultdict(float)
        self.child_N = collections.defaultdict(float)


class Node:
    """Node in the MCTS search tree."""

    def __init__(
        self,
        valid_actions: list[Action],
        move: int | None = None,
        parent: Any = None,
    ) -> None:
        """
        Args:
            num_actions: number of total actions, including illegal move.
            move: the action associated with getting to this node.
            None for the root node.
            parent: the parent node, could be a `DummyNode` if this is the root node.
        """

        self.move = move
        self.parent = parent
        self.is_expanded = False

        N = len(valid_actions)
        self.valid_actions = valid_actions
        self.child_W = np.zeros(N, dtype=np.float32)
        self.child_N = np.zeros(N, dtype=np.float32)
        self.child_P = np.zeros(N, dtype=np.float32)

        self.children: dict[tuple[Action, ...], Node] = {}

    def child_U(self, c_puct_base: float, c_puct_init: float) -> np.ndarray:
        """Returns a 1D numpy.array contains prior score for all child."""
        pb_c = math.log((1 + self.N + c_puct_base) / c_puct_base) + c_puct_init
        return pb_c * self.child_P * (math.sqrt(self.N) / (1 + self.child_N))

    def child_Q(self):
        """Returns a 1D numpy.array contains mean action value for all child."""
        # Avoid division by zero
        child_N = np.where(self.child_N > 0, self.child_N, 1)

        return self.child_W / child_N

    @property
    def N(self):
        """The number of visits for current node is stored at parent's level."""
        return self.parent.child_N[self.move]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.move] = value

    @property
    def W(self):
        """The total value for current node is stored at parent's level."""
        return self.parent.child_W[self.move]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.move] = value

    @property
    def Q(self):
        """Returns the mean action value Q(s, a)."""
        if self.parent.child_N[self.move] > 0:
            return self.parent.child_W[self.move] / self.parent.child_N[self.move]
        else:
            return 0.0

    @property
    def has_parent(self) -> bool:
        return isinstance(self.parent, Node)


def best_move(
    node: Node,
    c_puct_base: float,
    c_puct_init: float,
) -> int:
    """Returns best child node with maximum action value Q plus an upper confidence bound U.
    And creates the selected best child node if not already exists.

    Args:
        node: the current node in the search tree.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.

    Returns:
        The best child node corresponding to the UCT score.

    Raises:
        ValueError:
            if the node instance itself is a leaf node.
    """
    if not node.is_expanded:
        raise ValueError("Expand leaf node first.")

    # The child Q value is evaluated from the opponent perspective.
    # when we select the best child for node, we want to do so from node.to_play's perspective,
    # so we always switch the sign for node.child_Q values, this is required since we're talking about two-player, zero-sum games.
    ucb_scores = node.child_Q() + node.child_U(c_puct_base, c_puct_init)

    move = int(np.argmax(ucb_scores))
    return move


def expand(node: Node, prior_prob: np.ndarray) -> None:
    """Expand all actions, including illegal actions.

    Args:
        node: current leaf node in the search tree.
        prior_prob: 1D numpy.array contains prior probabilities of the state for all actions.

    Raises:
        ValueError:
            if node instance already expanded.
            if input argument `prior` is not a valid 1D float numpy.array.
    """
    if node.is_expanded:
        raise RuntimeError("Node already expanded.")

    assert len(prior_prob) == len(node.valid_actions)
    node.child_P = prior_prob
    node.is_expanded = True


def backup(node: Node, value: float) -> None:
    """Update statistics of the this node and all traversed parent nodes.

    Args:
        node: current leaf node in the search tree.
        value: the evaluation value evaluated from current player's perspective.

    Raises:
        ValueError:
            if input argument `value` is not float data type.
    """

    if not isinstance(value, float):
        raise ValueError(f"Expect `value` to be a float type, got {type(value)}")

    while isinstance(node, Node):
        node.N += 1
        node.W += value
        node = node.parent


def add_dirichlet_noise(node: Node, eps: float = 0.25, alpha: float = 0.03) -> None:
    """Add dirichlet noise to a given node.

    Args:
        node: the root node we want to add noise to.
        eps: epsilon constant to weight the priors vs. dirichlet noise.
        alpha: parameter of the dirichlet noise distribution.

    Raises:
        ValueError:
            if input argument `node` is not expanded.
            if input argument `eps` or `alpha` is not float type
                or not in the range of [0.0, 1.0].
    """

    if not isinstance(node, Node) or not node.is_expanded:
        raise ValueError("Expect `node` to be expanded")
    if not isinstance(eps, float) or not 0.0 <= eps <= 1.0:
        raise ValueError(
            f"Expect `eps` to be a float in the range [0.0, 1.0], got {eps}"
        )
    if not isinstance(alpha, float) or not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"Expect `alpha` to be a float in the range [0.0, 1.0], got {alpha}"
        )

    alphas = np.ones_like(node.child_P) * alpha
    noise = np.random.dirichlet(alphas)

    node.child_P = node.child_P * (1 - eps) + noise * eps


def generate_search_policy(
    child_N: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Returns a policy action probabilities after MCTS search,
    proportional to its exponentialted visit count.

    Args:
        child_N: the visit number of the children nodes from the root node of the search tree.
        temperature: a parameter controls the level of exploration.

    Returns:
        a 1D numpy.array contains the action probabilities after MCTS search.

    Raises:
        ValueError:
            if input argument `temperature` is not float type or not in range (0.0, 1.0].
    """
    if not isinstance(temperature, float) or not 0 < temperature <= 1.0:
        raise ValueError(
            f"Expect `temperature` to be float type in the range (0.0, 1.0], got {temperature}"
        )

    if temperature > 0.0:
        # Simple hack to avoid overflow when call np.power over large numbers
        exp = max(1.0, min(5.0, 1.0 / temperature))
        child_N = np.power(child_N, exp)

    assert np.all(child_N >= 0) and not np.any(np.isnan(child_N))
    pi_probs = child_N
    sums = np.sum(pi_probs)
    if sums > 0:
        pi_probs /= sums

    return pi_probs


def uct_search(
    engine: Engine,
    ai: AI,
    ai_state: dict,
    c_puct_base: float = 19652,
    c_puct_init: float = 1.25,
    num_simulations: int = 200,
    skip_thresh: float = 0.9,
    root_noise: bool = False,
) -> tuple[Action, Node]:
    """Single-threaded Upper Confidence Bound (UCB) for Trees (UCT) search without any rollout.

    This implementation uses tree parallel search and batched evaluation.

    It follows the following general UCT search algorithm, except here we don't do rollout.
    ```
    function UCTSEARCH(r,m)
      i←1
      for i ≤ m do
          n ← select(r)
          n ← expand(n)
          ∆ ← rollout(n)
          backup(n,∆)
      end for
      return end function
    ```

    Args:
        env: a gym like custom GoEnv environment.
        eval_func: a evaluation function when called returns the
            action probabilities and predicted value from
            current player's perspective.
        root_node: root node of the search tree, this comes from reuse sub-tree.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.
        num_simulations: number of simulations to run.
        num_parallel: Number of parallel leaves for MCTS search. This is also the batch size for neural network evaluation.
        root_noise: whether add dirichlet noise to root node to encourage exploration,
            default off.


    Returns:
        tuple contains:
            a integer indicate the sampled action to play in the environment.
            a 1D numpy.array search policy action probabilities from the MCTS search result.
            a float indicate the root node value
            a float indicate the best child value
            a Node instance represent subtree of this MCTS search, which can be used as next root node for MCTS search.

    Raises:
        ValueError:
            if input argument `env` is not valid GoEnv instance.
            if input argument `num_simulations` is not a positive integer.
        RuntimeError:
            if the game is over.
    """

    assert engine.state.phase == "play"

    # Create root node
    root_node = Node(
        engine.valid_actions(),
        parent=DummyNode(),
    )
    _, prior_prob, value = ai.get_pv(engine, ai_state)
    if np.max(prior_prob) >= skip_thresh:
        move = int(np.argmax(prior_prob))
        action = root_node.valid_actions[move]
        return action, root_node

    expand(root_node, prior_prob)
    backup(root_node, value)

    # Add dirichlet noise to the prior probabilities to root node.
    if root_noise:
        add_dirichlet_noise(root_node)

    orig_state = engine.state
    orig_player = engine.state.cur_player

    while root_node.N < num_simulations:
        node = root_node

        # Make sure do not touch the actual environment.
        sim_state = sample_determinization(orig_state)
        engine.state = sim_state
        sim_ai_state = copy.deepcopy(ai_state)
        cum_reward = 0.0

        # Phase 1 - Select
        # Select best child node until one of the following is true:
        # - reach a leaf node.
        # - game is over.
        while node.is_expanded and engine.state.phase != "end":
            actions: list[Action] = []

            # Select the best move and create the child node on demand
            assert engine.state.cur_player == orig_player
            if node is not root_node:
                ai.get_pv(engine, sim_ai_state)
            move = best_move(
                node,
                c_puct_base,
                c_puct_init,
            )
            action = node.valid_actions[move]
            actions.append(action)
            # Make move on the simulation environment.
            cum_reward += engine.move(action)
            if engine.state.phase != "end":
                ai.record_move(engine, action, sim_ai_state)

            while (
                engine.state.cur_player != orig_player and engine.state.phase != "end"
            ):
                valid_actions, probs, _ = ai.get_pv(engine, sim_ai_state)
                action = valid_actions[int(np.argmax(probs))]
                actions.append(action)
                cum_reward += engine.move(action)
                if engine.state.phase != "end":
                    ai.record_move(engine, action, sim_ai_state)

            actions_tuple = tuple(actions)
            if actions_tuple not in node.children:
                valid_actions = (
                    engine.valid_actions() if engine.state.phase != "end" else []
                )
                node.children[actions_tuple] = Node(
                    valid_actions=valid_actions, move=move, parent=node
                )
            node = node.children[actions_tuple]

        # Special case - If game is over, using the actual reward from the game to update statistics.
        if engine.state.phase == "end":
            # The reward is for the last player who made the move won/loss the game.
            backup(node, cum_reward)
            continue

        assert engine.state.cur_player == orig_player

        # Phase 2 - Expand and evaluation
        _, prior_prob, value = ai.get_pv(engine, sim_ai_state)
        expand(node, prior_prob)

        # Phase 3 - Backup statistics
        backup(node, cum_reward + value)

    # Choose the child with most visit count.
    move = int(np.argmax(root_node.child_N))
    action = root_node.valid_actions[move]

    engine.state = orig_state

    return action, root_node
