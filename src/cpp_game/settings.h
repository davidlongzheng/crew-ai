#pragma once

#include <string>
#include <vector>
#include <optional>
#include <set>
#include <array>
#include <stdexcept>

#include "types.h"

// Game settings struct representing the configuration of a game
struct Settings
{
    Settings(int num_players_, int num_side_suits_, bool use_trump_suit_, int side_suit_length_, int trump_suit_length_, bool use_signals_, bool cheating_signal_, bool single_signal_, std::string bank_, std::string task_distro_, std::vector<int> task_idxs_, std::optional<int> min_difficulty_, std::optional<int> max_difficulty_, std::optional<int> max_num_tasks_, bool use_drafting_, int num_draft_tricks_, double task_bonus_, double win_bonus_);

    // Default values match Python implementation
    int num_players = 4;
    int num_side_suits = 4;
    bool use_trump_suit = true;
    int side_suit_length = 9;
    int trump_suit_length = 4;
    bool use_signals = true;
    bool cheating_signal = false;
    bool single_signal = false;

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

    bool use_drafting = false;
    int num_draft_tricks = 3;

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

    // cache some common values.
    int num_tricks;
    int max_hand_size;
    int num_cards;
    std::vector<int> suits;
    const std::vector<std::tuple<std::string, std::string, int>> &task_defs;
    int num_task_defs;
    int max_num_actions;
    bool use_nosignal;
    int max_suit_length;
    int num_suits;
    int num_phases;
    int resolved_max_num_tasks;
    int seq_length;

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

    int get_phase_idx(Phase phase) const
    {
        switch (phase)
        {
        case Phase::kPlay:
            return 0;
        case Phase::kSignal:
            assert(use_signals);
            return 1;
        case Phase::kDraft:
            assert(use_drafting);
            return use_signals ? 2 : 1;
        default:
            throw std::runtime_error("Invalid phase");
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

        if (single_signal || cheating_signal)
        {
            if (single_signal && cheating_signal)
            {
                return false;
            }
            if (!use_signals)
            {
                return false;
            }
        }

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

        if (use_drafting)
        {
            if (num_draft_tricks == 0)
                return false;
            if (num_draft_tricks * num_players < resolved_max_num_tasks)
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
               ", single_signal=" + std::to_string(single_signal) +
               ", cheating_signal=" + std::to_string(cheating_signal) +
               ", use_drafting=" + std::to_string(use_drafting) +
               ", num_draft_tricks=" + std::to_string(num_draft_tricks) +
               ", bank=" + bank +
               ", task_distro=" + task_distro +
               ", task_bonus=" + std::to_string(task_bonus) +
               ", win_bonus=" + std::to_string(win_bonus) + ")";
    }
};
