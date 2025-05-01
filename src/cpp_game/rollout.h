#pragma once

#include <vector>
#include <memory>
#include <optional>
#include <torch/torch.h>
#include "engine.h"
#include "settings.h"
#include "types.h"

struct MoveInputs
{
    // History tensors
    torch::Tensor hist_player_idxs;
    torch::Tensor hist_tricks;
    torch::Tensor hist_cards;
    torch::Tensor hist_turns;
    torch::Tensor hist_phases;

    // Private input tensors
    torch::Tensor hand;
    torch::Tensor hands;
    torch::Tensor player_idx;
    torch::Tensor trick;
    torch::Tensor turn;
    torch::Tensor phase;

    // Action and task tensors
    torch::Tensor valid_actions;
    torch::Tensor task_idxs;
};

struct RolloutResults
{
    // History tensors
    torch::Tensor hist_player_idxs;
    torch::Tensor hist_tricks;
    torch::Tensor hist_cards;
    torch::Tensor hist_turns;
    torch::Tensor hist_phases;

    // Private input tensors
    torch::Tensor hand;
    torch::Tensor hands;
    torch::Tensor player_idx;
    torch::Tensor trick;
    torch::Tensor turn;
    torch::Tensor phase;

    // Action and task tensors
    torch::Tensor valid_actions;
    torch::Tensor task_idxs;

    // Probability and reward tensors
    torch::Tensor probs;
    torch::Tensor log_probs;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor num_success_tasks_pp;
    torch::Tensor win;
};

struct Rollout
{
    Rollout(const Settings &settings, int engine_seed);

    void move(int action_idx, const torch::Tensor &probs, const torch::Tensor &log_probs);

    // state
    const Settings &settings;
    std::unique_ptr<Engine> engine;
    std::vector<Action> valid_actions;

    // history tensors
    std::vector<torch::Tensor> hist_player_idxs_pt;
    std::vector<torch::Tensor> hist_tricks_pt;
    std::vector<torch::Tensor> hist_cards_pt;
    std::vector<torch::Tensor> hist_turns_pt;
    std::vector<torch::Tensor> hist_phases_pt;

    // private input tensors
    std::vector<torch::Tensor> hand_pt;
    std::vector<torch::Tensor> hands_pt;
    std::vector<torch::Tensor> player_idx_pt;
    std::vector<torch::Tensor> trick_pt;
    std::vector<torch::Tensor> turn_pt;
    std::vector<torch::Tensor> phase_pt;

    // action and task tensors
    std::vector<torch::Tensor> valid_actions_pt;
    std::vector<torch::Tensor> task_idxs_pt;

    // probability and reward tensors
    std::vector<torch::Tensor> probs_pt;
    std::vector<torch::Tensor> log_probs_pt;
    std::vector<torch::Tensor> actions_pt;
    std::vector<torch::Tensor> rewards_pt;

    // Helper functions
    torch::Tensor card_to_tensor(const std::optional<Card> &card) const;
    torch::Tensor encode_hand(const std::vector<Card> &hand) const;
};

struct BatchRollout
{
    BatchRollout(const Settings &settings, int num_rollouts, const std::vector<int> engine_seeds);

    MoveInputs get_move_inputs();
    void move(const torch::Tensor &action_indices, const torch::Tensor &probs, const torch::Tensor &log_probs);
    bool is_done() const;
    RolloutResults get_results() const;

    const Settings &settings;
    int num_rollouts;
    std::vector<Rollout> rollouts;
};