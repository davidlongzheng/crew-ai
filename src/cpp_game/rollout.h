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
#include "thread_pool.h"
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
    RolloutResults(int num_rollouts, int seq_length, int hand_pad_size, int max_num_tasks, int num_cards);

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
    py::array_t<float> log_probs;
    py::array_t<long> actions;
    py::array_t<float> rewards;
    py::array_t<float> frac_success;
    py::array_t<bool> win;

    // Aux info
    py::array_t<int8_t> aux_info;
};

struct Rollout
{
    Rollout(const Settings &settings);

    void init_state();
    void reset_state(int engine_seed);
    void record_move_inputs();
    void move(int action_idx);
    void pop_last_history();
    // state
    const Settings &settings;
    std::unique_ptr<Engine> engine;
    size_t seq_length;
    size_t max_suit_length;
    size_t num_suits;
    size_t hand_pad_size;
    size_t max_num_tasks;
    std::vector<Action> valid_actions;

    // history tensors
    std::vector<int8_t> hist_player_idxs_pt;
    std::vector<int8_t> hist_tricks_pt;
    std::vector<int8_t> hist_cards_pt;
    std::vector<int8_t> hist_turns_pt;
    std::vector<int8_t> hist_phases_pt;

    // private input tensors
    std::vector<int8_t> hand_pt;
    std::vector<int8_t> player_idx_pt;
    std::vector<int8_t> trick_pt;
    std::vector<int8_t> turn_pt;
    std::vector<int8_t> phase_pt;

    // action and task tensors
    std::vector<int8_t> valid_actions_pt;
    std::vector<int8_t> task_idxs;

    // probability and reward tensors
    std::vector<float> log_probs_pt;
    std::vector<long> actions_pt;
    std::vector<float> rewards_pt;

    // aux info
    std::vector<int8_t> aux_info;

    // Helper functions
    void encode_tasks();
    void encode_aux_info();
    void add_card(std::vector<int8_t> &vec, const std::optional<Card> &card);
    void add_hand(std::vector<int8_t> &vec, const std::vector<Card> &hand);
    void add_valid_actions(std::vector<int8_t> &vec, const std::vector<Action> &valid_actions);
    void add_log_probs(const py::array_t<float> &log_probs);
};

struct BatchRollout
{
    BatchRollout(const Settings &settings, int num_rollouts, bool multithread = false);
    void reset_state(const std::vector<int> &engine_seeds);

    const MoveInputs &get_move_inputs();
    void move(const py::array_t<int8_t> &action_indices, const py::array_t<float> &log_probs);
    bool is_done() const;
    const RolloutResults &get_results();

    const Settings &settings;
    int num_rollouts;
    int seq_length;
    int hand_pad_size;
    int max_num_tasks;

    std::vector<Rollout> rollouts;

    MoveInputs move_inputs;
    RolloutResults results;

    std::unique_ptr<ThreadPool> pool;
};