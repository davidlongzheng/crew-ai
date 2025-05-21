#pragma once

#include <optional>
#include <string>
#include <vector>

#include "tasks.h"
#include "types.h"

inline int safe_mod(int x, int mod)
{
    int ret = x % mod;
    return ret < 0 ? ret + mod : ret;
}

// Game state struct representing the current state of the game
struct State
{
    // Member fields
    int num_players;
    Phase phase;
    std::vector<std::vector<Card>> hands;
    std::vector<Action> actions;
    int trick;
    int leader;
    int captain;
    int cur_player;
    std::vector<std::pair<Card, int>> active_cards;
    std::vector<std::pair<std::vector<Card>, int>> past_tricks;
    std::vector<std::optional<Signal>> signals;
    std::optional<int> trick_winner;
    std::vector<int> task_idxs;
    std::vector<int> unassigned_task_idxs;
    std::vector<std::vector<AssignedTask>> assigned_tasks;
    Status status;
    double value;

    // Helper methods
    int get_next_player(int player = -1) const
    {
        player = (player == -1) ? cur_player : player;
        return safe_mod(player + 1, num_players);
    }

    int get_player_idx(int player = -1) const
    {
        player = (player == -1) ? cur_player : player;
        return safe_mod(player - captain, num_players);
    }

    int get_player(int player_idx) const
    {
        return safe_mod(captain + player_idx, num_players);
    }

    int get_turn(int player = -1) const
    {
        player = (player == -1) ? cur_player : player;
        return safe_mod(player - leader, num_players);
    }

    int lead_suit() const
    {
        assert(!active_cards.empty());
        return active_cards[0].first.suit;
    }

    std::string to_string() const
    {
        std::string result = "Phase: " + phase_to_string(phase) +
                             " Trick: " + std::to_string(trick) +
                             " Player: " + std::to_string(cur_player) + "\n\n";

        result += "Active Cards:\n";
        for (const auto &[card, player] : active_cards)
        {
            result += std::to_string(player) + " plays " + card.to_string() + "\n";
        }

        result += "\nHands:\n";
        for (size_t i = 0; i < hands.size(); ++i)
        {
            if (i == static_cast<size_t>(cur_player))
                result += "** ";
            else
                result += "";

            result += std::to_string(i) + ": ";
            for (const auto &card : hands[i])
            {
                result += card.to_string() + " ";
            }
            result += "\n";
        }

        return result;
    }
};