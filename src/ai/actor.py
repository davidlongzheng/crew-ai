import numpy as np
import torch

from game.tasks import get_task_defs
from game.utils import calc_trick_winner, to_hand


class BatchActor:
    def start(self):
        pass

    def get_log_probs(self, inps):
        raise NotImplementedError()

    def stop(self):
        pass


class ModelBatchActor(BatchActor):
    def __init__(self, pv_model):
        self.pv_model = pv_model

    def start(self):
        self.pv_model.eval()
        self.pv_model.start_single_step()

    def get_log_probs(self, inps):
        with torch.no_grad():
            log_probs_pr, _, _ = self.pv_model(inps)
            log_probs_pr = log_probs_pr.to("cpu").numpy()

        return log_probs_pr

    def stop(self):
        self.pv_model.stop_single_step()


class GreedySingleActor(BatchActor):
    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def reset(self):
        # Track num_tricks_needed & num_tricks_won per player to determine
        # if player should try to win or lose the trick.
        self.num_tricks_needed = None
        self.num_tricks_won = {pidx: 0 for pidx in range(self.settings.num_players)}
        # Claimed means that someone is trying to win the trick and the
        # rest of the players should duck.
        self.claimed = False
        # Which cards are unplayed per suit, which helps to determine
        # which cards are the most likely winners.
        self.cards_unseen = {
            suit: set(range(1, self.settings.get_suit_length(suit) + 1))
            for suit in self.settings.get_suits()
        }
        # Used to determine if trick is new trick.
        self.prev_trick = -1
        # Whether the leader should try to win this trick or not.
        self.leader_should_win = False
        # Active cards of the trick.
        self.active_cards = []
        # How many times each player has tried to pass control in a given suit.
        # Helps to pass in different suits when possible.
        self.led_passes = {
            pidx: {suit: 0 for suit in self.settings.get_suits()}
            for pidx in range(self.settings.num_players)
        }

    def save_tasks(self, task_idxs):
        self.num_tricks_needed = {pidx: 0 for pidx in range(self.settings.num_players)}
        task_defs = get_task_defs(self.settings.bank)
        for task_idx, pidx in task_idxs:
            task_idx = task_idx.item()
            pidx = pidx.item()
            formula, _, _ = task_defs[task_idx]
            assert formula.startswith("#T>=")
            num_tricks_needed = int(formula.removeprefix("#T>="))
            self.num_tricks_needed[pidx] = max(
                self.num_tricks_needed[pidx], num_tricks_needed
            )

    def compute_score(self, card, *, highest):
        """Score = number of cards better than the given card."""
        return sum(
            x > card.rank if highest else x < card.rank
            for x in self.cards_unseen[card.suit]
        )

    def choose_card(
        self, valid_cards, pidx, *, highest, winning=False, losing=False, is_pass=False
    ):
        assert not (winning and losing)
        if is_pass:
            assert not highest

        best_card = None
        best_score = None
        for card in valid_cards:
            is_winner = calc_trick_winner(self.active_cards + [(card, pidx)]) == pidx
            if winning and not is_winner:
                continue
            if losing and is_winner:
                continue
            score = self.compute_score(card, highest=highest)
            if is_pass:
                score = (self.led_passes[pidx][card.suit], score)
            if best_card is None or score < best_score:
                best_card = card
                best_score = score

        return best_card

    def get_log_probs(self, inps):
        """Strat is to play the highest rank as the leader if the leader isn't done."""
        if self.num_tricks_needed is None:
            self.save_tasks(inps["task_idxs"])

        phase = inps["private"]["phase"].item()
        trick = inps["private"]["trick"].item()
        assert phase == 0

        pidx = inps["private"]["player_idx"].item()
        is_leader = trick != self.prev_trick
        if is_leader and self.active_cards:
            trick_winner = calc_trick_winner(self.active_cards)
            self.num_tricks_won[trick_winner] += 1

        did_finish = self.num_tricks_won[pidx] >= self.num_tricks_needed[pidx]
        if is_leader:
            self.leader_should_win = not did_finish
            self.claimed = self.leader_should_win
            self.active_cards = []

        valid_cards = to_hand(inps["valid_actions"], self.settings)

        if self.leader_should_win:
            if is_leader:
                card = self.choose_card(valid_cards, pidx, highest=True)
            else:
                if not did_finish:
                    card = self.choose_card(valid_cards, pidx, highest=False)
                else:
                    card = self.choose_card(
                        valid_cards, pidx, highest=True, losing=True
                    ) or self.choose_card(valid_cards, pidx, highest=False)
        else:
            if is_leader:
                card = self.choose_card(valid_cards, pidx, highest=False, is_pass=True)
                self.led_passes[pidx][card.suit] += 1
            else:
                if not did_finish and not self.claimed:
                    card = self.choose_card(
                        valid_cards, pidx, highest=True, winning=True
                    )
                    if card:
                        self.claimed = True
                    else:
                        card = self.choose_card(valid_cards, pidx, highest=False)
                elif not did_finish and self.claimed:
                    card = self.choose_card(valid_cards, pidx, highest=False)
                else:
                    card = self.choose_card(
                        valid_cards, pidx, highest=True, losing=True
                    ) or self.choose_card(valid_cards, pidx, highest=False)

        assert card is not None

        self.active_cards.append((card, pidx))
        self.cards_unseen[card.suit].remove(card.rank)
        self.prev_trick = trick

        action_idx = valid_cards.index(card)
        log_probs = np.full(inps["valid_actions"].size(0), -np.inf)
        log_probs[action_idx] = 0
        return log_probs


class GreedyBatchActor(BatchActor):
    def __init__(self, settings, num_rollouts):
        self.actors = [GreedySingleActor(settings) for _ in range(num_rollouts)]
        self.settings = settings
        self.num_rollouts = num_rollouts

    def start(self):
        for actor in self.actors:
            actor.reset()

    def get_log_probs(self, inps):
        inps.auto_batch_size_()
        log_probs_pr = []
        for i, actor in enumerate(self.actors):
            log_probs = actor.get_log_probs(inps[i])
            log_probs_pr.append(log_probs)

        return np.stack(log_probs_pr)
