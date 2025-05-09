#pragma once

#include <vector>
#include <array>
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "engine.h"
#include "settings.h"
#include "types.h"

namespace py = pybind11;

struct MoveInputs
{
    MoveInputs(int num_rollouts, int hand_pad_size, int max_num_tasks);

    // History tensors
    py::array_t<int8_t> hist_player_idxs;
    py::array_t<int8_t> hist_tricks;
    py::array_t<int8_t> hist_cards;
    py::array_t<int8_t> hist_turns;
    py::array_t<int8_t> hist_phases;

    // Private input tensors
    py::array_t<int8_t> hand;
    py::array_t<int8_t> player_idx;
    py::array_t<int8_t> trick;
    py::array_t<int8_t> turn;
    py::array_t<int8_t> phase;

    // Action and task tensors
    py::array_t<int8_t> valid_actions;
    py::array_t<int8_t> task_idxs;
};

struct RolloutResults
{
    RolloutResults(int num_rollouts, int seq_length, int hand_pad_size, int max_num_tasks);

    // History tensors
    py::array_t<int8_t> hist_player_idxs;
    py::array_t<int8_t> hist_tricks;
    py::array_t<int8_t> hist_cards;
    py::array_t<int8_t> hist_turns;
    py::array_t<int8_t> hist_phases;

    // Private input tensors
    py::array_t<int8_t> hand;
    py::array_t<int8_t> player_idx;
    py::array_t<int8_t> trick;
    py::array_t<int8_t> turn;
    py::array_t<int8_t> phase;

    // Action and task tensors
    py::array_t<int8_t> valid_actions;
    py::array_t<int8_t> task_idxs;

    // Probability and reward tensors
    py::array_t<float> probs;
    py::array_t<float> log_probs;
    py::array_t<int8_t> actions;
    py::array_t<float> rewards;
    py::array_t<int8_t> num_success_tasks_pp;
    py::array_t<bool> win;
};

struct Rollout
{
    Rollout(const Settings &settings, int engine_seed);

    void record_move_inputs();
    void move(int action_idx, const py::array_t<float> &probs, const py::array_t<float> &log_probs);
    void pop_last_history();
    // state
    const Settings &settings;
    std::unique_ptr<Engine> engine;
    size_t max_num_tasks;
    size_t max_suit_length;
    size_t num_suits;
    size_t hand_pad_size;
    std::vector<Action> valid_actions;

    // history tensors
    std::vector<int8_t> hist_player_idxs_pt;
    std::vector<int8_t> hist_tricks_pt;
    std::vector<std::array<int8_t, 2>> hist_cards_pt;
    std::vector<int8_t> hist_turns_pt;
    std::vector<int8_t> hist_phases_pt;

    // private input tensors
    std::vector<std::vector<int8_t>> hand_pt;
    std::vector<int8_t> player_idx_pt;
    std::vector<int8_t> trick_pt;
    std::vector<int8_t> turn_pt;
    std::vector<int8_t> phase_pt;

    // action and task tensors
    std::vector<std::vector<int8_t>> valid_actions_pt;
    std::vector<int8_t> task_idxs;

    // probability and reward tensors
    std::vector<std::vector<float>> probs_pt;
    std::vector<std::vector<float>> log_probs_pt;
    std::vector<int8_t> actions_pt;
    std::vector<float> rewards_pt;

    // Helper functions
    std::vector<int8_t> encode_tasks() const;
    std::array<int8_t, 2> card_to_arr(const std::optional<Card> &card) const;
    std::vector<int8_t> encode_hand(const std::vector<Card> &hand) const;
    std::vector<int8_t> encode_valid_actions(const std::vector<Action> &valid_actions) const;
};

struct BatchRollout
{
    BatchRollout(const Settings &settings, int num_rollouts, const std::vector<int> engine_seeds);

    MoveInputs get_move_inputs();
    void move(const py::array_t<int8_t> &action_indices, const py::array_t<float> &probs, const py::array_t<float> &log_probs);
    bool is_done() const;
    RolloutResults get_results();

    const Settings &settings;
    int num_rollouts;
    std::vector<Rollout> rollouts;
};