#pragma once

#include <vector>
#include <memory>
#include <optional>
#include "engine.h"
#include "settings.h"
#include "types.h"

struct PublicHistory
{
    int trick = -1;
    std::tuple<int, int> card;
    int player_idx = -1;
    int turn = -1;
    int phase = -1;
};

struct PrivateInput
{
    std::vector<std::tuple<int, int>> hand;
    std::vector<std::vector<std::tuple<int, int>>> hands;
    int trick;
    int player_idx;
    int turn;
    int phase;
};

struct MoveInput
{
    PublicHistory public_history;
    PrivateInput private_inputs;
    std::vector<std::tuple<int, int>> valid_actions;
    std::vector<std::tuple<int, int>> task_idxs;
};

struct RolloutResult
{
    std::vector<PublicHistory> public_history;
    std::vector<PrivateInput> private_inputs;
    std::vector<std::vector<std::tuple<int, int>>> valid_actions;
    std::vector<std::vector<double>> probs;
    std::vector<std::vector<double>> log_probs;
    std::vector<int> actions;
    std::vector<double> rewards;
    std::vector<std::tuple<int, int>> num_success_tasks_pp;
    std::vector<std::vector<std::tuple<int, int>>> task_idxs;
    bool win;
};

struct Rollout
{
    Rollout(const Settings &settings, int engine_seed);

    MoveInput get_move_input();
    void move(int action_idx, const std::vector<double> &probs, const std::vector<double> &log_probs);

    // state
    const Settings &settings;
    std::unique_ptr<Engine> engine;
    std::vector<Action> valid_actions;

    // history
    std::vector<PublicHistory> public_history_pt;
    std::vector<PrivateInput> private_inputs_pt;
    std::vector<std::vector<std::tuple<int, int>>> valid_actions_pt;
    std::vector<int> actions_pt;
    std::vector<std::vector<double>> probs_pt;
    std::vector<std::vector<double>> log_probs_pt;
    std::vector<double> rewards_pt;
    std::vector<std::vector<std::tuple<int, int>>> task_idxs_pt;
    std::vector<std::tuple<int, int>> task_idxs;

    // Helper functions
    std::tuple<int, int> card_to_tuple(const std::optional<Card> &card) const;
    std::vector<std::tuple<int, int>> encode_hand(const std::vector<Card> &hand) const;
};

struct BatchRollout
{
    BatchRollout(const Settings &settings, int num_rollouts, const std::vector<int> engine_seeds);

    std::vector<MoveInput> get_move_inputs();

    // Move the game forward with the given action indices
    void move(const std::vector<int> &action_indices, const std::vector<std::vector<double>> &probs, const std::vector<std::vector<double>> &log_probs);

    // Check if all rollouts are finished
    bool is_done() const;

    // Get results for all rollouts
    std::vector<RolloutResult> get_results() const;

    const Settings &settings;
    int num_rollouts;
    std::vector<Rollout> rollouts;
};