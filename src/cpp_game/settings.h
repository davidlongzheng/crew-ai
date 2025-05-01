#pragma once

#include <string>
#include <vector>
#include <optional>
#include <set>
#include <array>

#include "types.h"

// Game settings struct representing the configuration of a game
struct Settings
{
    // Default values match Python implementation
    int num_players = 4;
    int num_side_suits = 4;
    bool use_trump_suit = true;
    int side_suit_length = 9;
    int trump_suit_length = 4;
    bool use_signals = true;

    std::string bank = "easy";
    // In fixed, tasks are distributed according to the order of tasks,
    // starting from the leader.
    // In shuffle, tasks are shuffled and distributed clockwise starting
    // from a random player.
    // In random, each task is given to a random player.
    std::string task_distro = "shuffle";
    std::vector<int> task_idxs;
    std::optional<int> min_difficulty;
    std::optional<int> max_difficulty;
    std::optional<int> max_num_tasks;

    // Task bonus is how much of a bonus you get for a fully completed task
    // vs a partially completed task.
    // Partial points range from [-1, 1] and task bonus is {-task_bonus, 0, task_bonus}
    // depending on if the task was fail/unresolved/success.
    // Then everything is rescaled by 1 / (1 + task_bonus) so that the final value
    // ranges from [-1, 1].
    double task_bonus = 5.0;
    // Win bonus is like task bonus but for a game win vs a partially
    // completed tasks. The math is the same s.t. the final value is again
    // from [-1, 1]
    double win_bonus = 1.0;

    // Methods
    int num_tricks() const
    {
        return (num_side_suits * side_suit_length +
                (use_trump_suit ? trump_suit_length : 0)) /
               num_players;
    }

    int max_hand_size() const
    {
        return (num_side_suits * side_suit_length +
                (use_trump_suit ? trump_suit_length : 0) - 1) /
                   num_players +
               1;
    }

    int get_suit_idx(int suit) const
    {
        if (suit < num_side_suits)
        {
            return suit;
        }
        else if (suit == TRUMP_SUIT_NUM && use_trump_suit)
        {
            return num_side_suits;
        }
        else
        {
            throw std::runtime_error("Invalid suit");
        }
    }

    int get_suit(int suit_idx) const
    {
        if (suit_idx < num_side_suits)
        {
            return suit_idx;
        }
        else if (suit_idx == num_side_suits && use_trump_suit)
        {
            return TRUMP_SUIT_NUM;
        }
        else
        {
            throw std::runtime_error("Invalid suit index");
        }
    }

    std::vector<int> get_suits() const
    {
        std::vector<int> ret;
        for (int i = 0; i < num_side_suits; ++i)
        {
            ret.push_back(i);
        }
        if (use_trump_suit)
        {
            ret.push_back(TRUMP_SUIT_NUM);
        }
        return ret;
    }

    int get_suit_length(int suit) const
    {
        if (suit < num_side_suits)
        {
            return side_suit_length;
        }
        else if (suit == TRUMP_SUIT_NUM && use_trump_suit)
        {
            return trump_suit_length;
        }
        else
        {
            throw std::runtime_error("Invalid suit");
        }
    }

    int max_suit_length() const
    {
        return use_trump_suit ? std::max(side_suit_length, trump_suit_length) : side_suit_length;
    }

    int num_suits() const
    {
        return num_side_suits + (use_trump_suit ? 1 : 0);
    }

    int num_phases() const
    {
        return 1 + (use_signals ? 1 : 0);
    }

    int get_max_num_tasks() const
    {
        if (!task_idxs.empty())
        {
            return task_idxs.size();
        }
        else
        {
            return max_num_tasks.value_or(0);
        }
    }

    // Validation
    bool validate() const
    {
        if (num_players < 2 || num_players > 5)
            return false;
        if (num_side_suits > TRUMP_SUIT_NUM)
            return false;
        if (side_suit_length < 1)
            return false;
        if (trump_suit_length < 1)
            return false;
        if (task_bonus < 0)
            return false;
        if (win_bonus < 0)
            return false;

        // Check task difficulty constraints
        bool has_task_idxs = !task_idxs.empty();
        bool has_min_difficulty = min_difficulty.has_value();
        bool has_max_difficulty = max_difficulty.has_value();
        bool has_max_num_tasks = max_num_tasks.has_value();

        if (has_task_idxs != !has_min_difficulty ||
            has_task_idxs != !has_max_difficulty ||
            has_task_idxs != !has_max_num_tasks)
        {
            return false;
        }

        if (has_min_difficulty && has_max_difficulty &&
            min_difficulty.value() >= max_difficulty.value())
        {
            return false;
        }

        return true;
    }

    // String representation
    std::string to_string() const
    {
        return "Settings(num_players=" + std::to_string(num_players) +
               ", num_side_suits=" + std::to_string(num_side_suits) +
               ", use_trump_suit=" + std::to_string(use_trump_suit) +
               ", side_suit_length=" + std::to_string(side_suit_length) +
               ", trump_suit_length=" + std::to_string(trump_suit_length) +
               ", use_signals=" + std::to_string(use_signals) +
               ", bank=" + bank +
               ", task_distro=" + task_distro +
               ", task_bonus=" + std::to_string(task_bonus) +
               ", win_bonus=" + std::to_string(win_bonus) + ")";
    }
};

// Preset factory functions
inline Settings get_preset(const std::string &preset)
{
    if (preset == "easy_p3")
    {
        Settings settings;
        settings.num_players = 3;
        settings.side_suit_length = 4;
        settings.trump_suit_length = 2;
        settings.use_signals = false;
        settings.bank = "easy";
        settings.task_idxs = {0, 0, 1};
        return settings;
    }
    else if (preset == "easy_p4")
    {
        Settings settings;
        settings.use_signals = false;
        settings.bank = "easy";
        settings.task_idxs = {0, 0, 1, 2};
        return settings;
    }
    else if (preset == "med")
    {
        Settings settings;
        settings.use_signals = false;
        settings.bank = "med";
        settings.min_difficulty = 1;
        settings.max_difficulty = 3;
        settings.max_num_tasks = 4;
        return settings;
    }
    else
    {
        throw std::runtime_error("Unknown preset: " + preset);
    }
}