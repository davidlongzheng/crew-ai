#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>

#include "engine.h"
#include "settings.h"
#include "state.h"
#include "tasks.h"
#include "types.h"
#include "utils.h"
#include "rng.h"

Engine::Engine(const Settings &settings, std::optional<int> seed)
    : settings(settings)
{
    settings.validate();
    reset_state(seed);
}

std::vector<std::vector<Card>> Engine::gen_hands(Rng &rng) const
{
    // Create deck
    std::vector<Card> deck;
    for (int suit : settings.get_suits())
    {
        for (int rank = 1; rank <= settings.get_suit_length(suit); ++rank)
        {
            deck.emplace_back(rank, suit);
        }
    }

    // Shuffle deck
    deck = rng.shuffle(deck);

    // Deal cards
    std::vector<std::vector<Card>> hands(settings.num_players);
    int cur_player = rng.randint(0, settings.num_players - 1);

    for (const Card &card : deck)
    {
        hands[cur_player].push_back(card);
        cur_player = (cur_player + 1) % settings.num_players;
    }

    // Sort hands by suit and rank
    for (auto &hand : hands)
    {
        std::sort(hand.begin(), hand.end(),
                  [](const Card &a, const Card &b)
                  {
                      if (a.suit != b.suit)
                      {
                          return a.suit < b.suit;
                      }
                      return a.rank < b.rank;
                  });
    }

    return hands;
}

std::vector<std::vector<AssignedTask>> Engine::gen_tasks(
    int leader, Rng &rng) const
{
    auto task_defs = get_task_defs(settings.bank);

    std::vector<int> task_idxs;
    if (!settings.task_idxs.empty())
    {
        task_idxs = std::vector<int>(settings.task_idxs.begin(),
                                     settings.task_idxs.end());
    }
    else
    {
        assert(settings.min_difficulty.has_value() &&
               settings.max_difficulty.has_value() &&
               settings.max_num_tasks.has_value());

        int difficulty = rng.randint(
            settings.min_difficulty.value(),
            settings.max_difficulty.value());

        while (true)
        {
            task_idxs.clear();
            int cur_difficulty = 0;

            while ((int)task_idxs.size() < settings.max_num_tasks.value() &&
                   cur_difficulty < difficulty)
            {
                int task_idx = rng.randint(0, task_defs.size() - 1);

                if (std::find(task_idxs.begin(), task_idxs.end(), task_idx) !=
                    task_idxs.end())
                {
                    continue;
                }

                int task_diff = std::get<2>(task_defs[task_idx]);
                if (cur_difficulty + task_diff <= difficulty)
                {
                    task_idxs.push_back(task_idx);
                    cur_difficulty += task_diff;
                }
            }

            if (cur_difficulty == difficulty)
            {
                break;
            }
        }
    }

    std::vector<std::vector<AssignedTask>> assigned_tasks(settings.num_players);

    if (settings.task_distro == "fixed" || settings.task_distro == "shuffle")
    {
        int start_player;
        if (settings.task_distro == "shuffle")
        {
            task_idxs = rng.shuffle(task_idxs);
            start_player = rng.randint(0, settings.num_players - 1);
        }
        else
        {
            start_player = leader;
        }

        for (size_t i = 0; i < task_idxs.size(); ++i)
        {
            int task_idx = task_idxs[i];
            if (task_idx < 0)
            {
                continue;
            }

            int player = (start_player + i) % settings.num_players;
            const auto &[formula, desc, difficulty] = task_defs[task_idx];

            assigned_tasks[player].emplace_back(
                formula, desc, difficulty, task_idx, player, settings);
        }
    }
    else
    {
        assert(settings.task_distro == "random");

        for (size_t i = 0; i < task_idxs.size(); ++i)
        {
            int task_idx = task_idxs[i];
            if (task_idx < 0)
            {
                continue;
            }

            int player = rng.randint(0, settings.num_players - 1);

            const auto &[formula, desc, difficulty] = task_defs[task_idx];

            assigned_tasks[player].emplace_back(
                formula, desc, difficulty, task_idx, player, settings);
        }
    }

    return assigned_tasks;
}

void Engine::reset_state(std::optional<int> seed)
{
    Rng rng(seed);

    auto hands = gen_hands(rng);

    // Find leader (player with trump card)
    int leader = -1;
    for (size_t i = 0; i < hands.size(); ++i)
    {
        for (const Card &card : hands[i])
        {
            if (card.rank == settings.trump_suit_length &&
                card.suit == TRUMP_SUIT_NUM)
            {
                leader = static_cast<int>(i);
                break;
            }
        }
        if (leader >= 0)
        {
            break;
        }
    }
    assert(leader >= 0);

    Phase phase = settings.use_signals ? Phase::kSignal : Phase::kPlay;

    state.num_players = settings.num_players;
    state.phase = phase;
    state.hands = hands;
    state.actions = std::vector<Action>();
    state.trick = 0;
    state.leader = leader;
    state.captain = leader;
    state.cur_player = leader;
    state.active_cards = std::vector<std::pair<Card, int>>();
    state.past_tricks = std::vector<std::pair<std::vector<Card>, int>>();
    state.signals = std::vector<std::optional<Signal>>(settings.num_players);
    state.trick_winner = std::nullopt;
    state.assigned_tasks = gen_tasks(leader, rng);
    state.status = Status::kUnresolved;
    state.value = 0.0;
}

int Engine::calc_trick_winner(
    const std::vector<std::pair<Card, int>> &active_cards) const
{
    auto max_it = std::max_element(
        active_cards.begin(), active_cards.end(),
        [this](const auto &a, const auto &b)
        {
            // Compare by trump, then by matching lead suit, then by rank
            bool a_is_trump = a.first.is_trump();
            bool b_is_trump = b.first.is_trump();

            if (a_is_trump != b_is_trump)
            {
                return !a_is_trump;
            }

            bool a_matches_lead = a.first.suit == state.lead_suit();
            bool b_matches_lead = b.first.suit == state.lead_suit();

            if (a_matches_lead != b_matches_lead)
            {
                return !a_matches_lead;
            }

            return a.first.rank < b.first.rank;
        });

    return max_it->second;
}

double Engine::move(const Action &action)
{
    if (state.phase == Phase::kSignal)
    {
        assert(settings.use_signals);
        assert(action.player == state.cur_player);
        auto &player_hand = state.hands[state.cur_player];
        assert(action.type == ActionType::kSignal || (settings.use_nosignal() && action.type == ActionType::kNoSignal));

        if (action.type == ActionType::kSignal)
        {
            assert(std::find(player_hand.begin(), player_hand.end(), action.card.value()) !=
                   player_hand.end());
            if (!settings.cheating_signal)
            {
                assert(!action.card.value().is_trump());
                assert(!state.signals[state.cur_player].has_value());
            }

            // Find matching suit cards
            std::vector<Card> matching_suit_cards;
            for (const Card &card : player_hand)
            {
                if (card.suit == action.card.value().suit)
                {
                    matching_suit_cards.push_back(card);
                }
            }

            // Sort by rank
            std::sort(matching_suit_cards.begin(), matching_suit_cards.end(),
                      [](const Card &a, const Card &b)
                      {
                          return a.rank < b.rank;
                      });

            if (!settings.cheating_signal)
            {
                assert(action.card.value() == matching_suit_cards.front() ||
                       action.card.value() == matching_suit_cards.back());
            }

            SignalValue value;
            if (matching_suit_cards.size() == 1)
            {
                value = SignalValue::kSingleton;
            }
            else if (action.card.value() == matching_suit_cards.back())
            {
                value = SignalValue::kHighest;
            }
            else if (action.card.value() == matching_suit_cards.front())
            {
                value = SignalValue::kLowest;
            }
            else
            {
                value = SignalValue::kOther;
            }

            state.signals[state.cur_player] = Signal{action.card.value(), value, state.trick};
        }

        state.actions.push_back(action);
        state.cur_player = state.get_next_player();

        if (state.cur_player == state.leader)
        {
            state.phase = Phase::kPlay;
        }
    }
    else if (state.phase == Phase::kPlay)
    {
        assert(action.player == state.cur_player);
        auto &player_hand = state.hands[state.cur_player];
        assert(std::find(player_hand.begin(), player_hand.end(), action.card.value()) !=
               player_hand.end());
        assert(action.type == ActionType::kPlay);

        if (state.cur_player != state.leader)
        {
            bool has_lead_suit = false;
            for (const Card &card : player_hand)
            {
                if (card.suit == state.lead_suit())
                {
                    has_lead_suit = true;
                    break;
                }
            }

            assert(action.card.value().suit == state.lead_suit() || !has_lead_suit);
        }

        // Remove card from hand
        auto it = std::find(player_hand.begin(), player_hand.end(), action.card.value());
        player_hand.erase(it);

        // Add to active cards
        state.active_cards.emplace_back(action.card.value(), action.player);
        state.actions.push_back(action);

        if (state.get_next_player() == state.leader)
        {
            int trick_winner = calc_trick_winner(state.active_cards);
            state.trick_winner = trick_winner;

            // Update tasks
            for (auto &tasks : state.assigned_tasks)
            {
                for (auto &task : tasks)
                {
                    task.on_trick_end(state);
                }
            }

            state.trick_winner = std::nullopt;

            // Record past trick
            std::vector<Card> trick_cards;
            for (const auto &[card, _] : state.active_cards)
            {
                trick_cards.push_back(card);
            }
            state.past_tricks.emplace_back(trick_cards, trick_winner);

            state.trick++;
            state.leader = trick_winner;
            state.cur_player = trick_winner;
            state.active_cards.clear();

            if (settings.use_signals && !settings.single_signal)
            {
                state.phase = Phase::kSignal;
            }
        }
        else
        {
            state.cur_player = state.get_next_player();
        }

        if (state.trick == settings.num_tricks())
        {
            state.phase = Phase::kEnd;

            for (auto &tasks : state.assigned_tasks)
            {
                for (auto &task : tasks)
                {
                    task.on_game_end();
                    assert(task.status != Status::kUnresolved);
                }
            }
        }
    }
    else if (state.phase == Phase::kEnd)
    {
        throw std::runtime_error("Game has ended!");
    }
    else
    {
        throw std::runtime_error("Unhandled phase: " +
                                 std::to_string(static_cast<int>(state.phase)));
    }

    // Update game status
    bool all_success = true;
    bool any_fail = false;

    for (const auto &tasks : state.assigned_tasks)
    {
        for (const auto &task : tasks)
        {
            if (task.status != Status::kSuccess)
            {
                all_success = false;
            }
            if (task.status == Status::kFail)
            {
                any_fail = true;
            }
        }
    }

    if (all_success)
    {
        state.status = Status::kSuccess;
    }
    else if (any_fail || state.phase == Phase::kEnd)
    {
        state.status = Status::kFail;
    }
    else
    {
        state.status = Status::kUnresolved;
    }

    if (state.phase == Phase::kEnd)
    {
        assert(state.status != Status::kUnresolved);
    }

    // Calculate value
    double prev_value = state.value;

    double total_task_value = 0.0;
    double total_difficulty = 0.0;

    for (const auto &tasks : state.assigned_tasks)
    {
        for (const auto &task : tasks)
        {
            total_task_value += task.value * task.difficulty;
            total_difficulty += task.difficulty;
        }
    }

    double avg_tasks_value = total_task_value / total_difficulty;
    assert(-1 <= avg_tasks_value && avg_tasks_value <= 1);

    double win_bonus = 0.0;
    if (state.status == Status::kSuccess)
    {
        win_bonus = settings.win_bonus;
    }
    else if (state.status == Status::kFail)
    {
        win_bonus = -settings.win_bonus;
    }

    state.value = (avg_tasks_value + win_bonus) / (settings.win_bonus + 1);
    assert(-1 <= state.value && state.value <= 1);

    double reward = state.value - prev_value;

    return reward;
}

std::vector<Action> Engine::valid_actions() const
{
    if (state.phase == Phase::kSignal)
    {
        assert(settings.use_signals);

        std::vector<Action> ret;

        if (!settings.single_signal && !settings.cheating_signal)
        {
            ret.push_back({state.cur_player, ActionType::kNoSignal, std::nullopt});
        }

        if (state.signals[state.cur_player].has_value() && !settings.cheating_signal)
        {
            return ret;
        }

        // Filter out trump cards
        std::vector<Card> player_hand;
        for (const Card &card : state.hands[state.cur_player])
        {
            if (!card.is_trump() || settings.cheating_signal)
            {
                player_hand.push_back(card);
            }
        }

        if (settings.cheating_signal)
        {
            for (const auto &card : player_hand)
            {
                ret.push_back({state.cur_player, ActionType::kSignal, card});
            }
        }
        else
        {
            // Split by suit
            auto sub_hands = split_by_suit(player_hand);

            for (const auto &sub_hand : sub_hands)
            {
                if (sub_hand.size() == 1)
                {
                    ret.push_back({state.cur_player, ActionType::kSignal, sub_hand.front()});
                }
                else
                {
                    ret.push_back({state.cur_player, ActionType::kSignal, sub_hand.front()});
                    ret.push_back({state.cur_player, ActionType::kSignal, sub_hand.back()});
                }
            }
        }

        return ret;
    }
    else if (state.phase == Phase::kPlay)
    {
        const auto &player_hand = state.hands[state.cur_player];

        if (state.cur_player != state.leader)
        {
            // Find matching suit cards
            std::vector<Card> matching_suit_cards;
            for (const Card &card : player_hand)
            {
                if (card.suit == state.lead_suit())
                {
                    matching_suit_cards.push_back(card);
                }
            }

            if (!matching_suit_cards.empty())
            {
                std::vector<Action> actions;
                for (const Card &card : matching_suit_cards)
                {
                    actions.push_back({state.cur_player, ActionType::kPlay, card});
                }
                return actions;
            }
        }

        // If leader or no matching suit cards, can play any card
        std::vector<Action> actions;
        for (const Card &card : player_hand)
        {
            actions.push_back({state.cur_player, ActionType::kPlay, card});
        }
        return actions;
    }
    else if (state.phase == Phase::kEnd)
    {
        throw std::runtime_error("Game has ended!");
    }
    else
    {
        throw std::runtime_error("Unhandled phase: " +
                                 std::to_string(static_cast<int>(state.phase)));
    }
}