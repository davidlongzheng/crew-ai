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
#include "rollout.h"
namespace py = pybind11;

struct MoveInputPtrs
{
    int8_t *hist_player_idx;
    int8_t *hist_trick;
    int8_t *hist_action;
    int8_t *hist_turn;
    int8_t *hist_phase;

    int8_t *hand;
    int8_t *player_idx;
    int8_t *trick;
    int8_t *turn;
    int8_t *phase;
    int8_t *task_idxs;

    int8_t *valid_actions;
};

struct Featurizer
{
    Featurizer(const Settings &settings_, int num_rollouts_) : settings(settings_), num_rollouts(num_rollouts_), move_inputs(num_rollouts, settings.max_hand_size, settings.max_num_actions, settings.resolved_max_num_tasks) {}

    void reset()
    {
        // Reset pointers
        ptrs.hist_player_idx = static_cast<int8_t *>(move_inputs.hist_player_idx.mutable_data());
        ptrs.hist_trick = static_cast<int8_t *>(move_inputs.hist_trick.mutable_data());
        ptrs.hist_action = static_cast<int8_t *>(move_inputs.hist_action.mutable_data());
        ptrs.hist_turn = static_cast<int8_t *>(move_inputs.hist_turn.mutable_data());
        ptrs.hist_phase = static_cast<int8_t *>(move_inputs.hist_phase.mutable_data());
        ptrs.hand = static_cast<int8_t *>(move_inputs.hand.mutable_data());
        ptrs.player_idx = static_cast<int8_t *>(move_inputs.player_idx.mutable_data());
        ptrs.trick = static_cast<int8_t *>(move_inputs.trick.mutable_data());
        ptrs.turn = static_cast<int8_t *>(move_inputs.turn.mutable_data());
        ptrs.phase = static_cast<int8_t *>(move_inputs.phase.mutable_data());
        ptrs.task_idxs = static_cast<int8_t *>(move_inputs.task_idxs.mutable_data());
        ptrs.valid_actions = static_cast<int8_t *>(move_inputs.valid_actions.mutable_data());
        rollout_idx = 0;
    }

    void record_move_inputs(const Engine &engine)
    {
        const State &state = engine.state;

        assert(rollout_idx < num_rollouts);
        if (state.last_action)
        {
            const auto [hist_player_idx, hist_trick, hist_action, hist_turn, hist_phase] = *state.last_action;
            *ptrs.hist_player_idx = hist_player_idx;
            *ptrs.hist_trick = hist_trick;
            std::vector<int8_t> hist_action_vec(2, -1);
            Rollout::add_action(hist_action_vec, 0, hist_action, settings);
            std::copy(hist_action_vec.begin(), hist_action_vec.end(), ptrs.hist_action);
            *ptrs.hist_turn = hist_turn;
            *ptrs.hist_phase = hist_phase;
        }
        else
        {
            *ptrs.hist_player_idx = -1;
            *ptrs.hist_trick = -1;
            ptrs.hist_action[0] = -1;
            ptrs.hist_action[1] = -1;
            *ptrs.hist_turn = -1;
            *ptrs.hist_phase = -1;
        }

        ++ptrs.hist_player_idx;
        ++ptrs.hist_trick;
        ptrs.hist_action += 2;
        ++ptrs.hist_turn;
        ++ptrs.hist_phase;

        int player_idx = state.get_player_idx();
        int turn = state.get_turn();

        std::vector<int8_t> hand_vec;
        Rollout::add_hand(hand_vec, state.hands[state.private_player >= 0 ? state.private_player : state.get_player(player_idx)], settings);
        std::copy(hand_vec.begin(), hand_vec.end(), ptrs.hand);
        ptrs.hand += hand_vec.size();

        *ptrs.player_idx = player_idx;
        ++ptrs.player_idx;
        *ptrs.trick = state.trick;
        ++ptrs.trick;
        *ptrs.turn = turn;
        ++ptrs.turn;
        *ptrs.phase = settings.get_phase_idx(state.phase);
        ++ptrs.phase;

        std::vector<int8_t> valid_actions_vec;
        Rollout::add_valid_actions(valid_actions_vec, engine.valid_actions(), settings);
        std::copy(valid_actions_vec.begin(), valid_actions_vec.end(), ptrs.valid_actions);
        ptrs.valid_actions += valid_actions_vec.size();

        std::vector<int8_t> task_idxs_vec;
        Rollout::add_tasks(task_idxs_vec, settings, state);
        std::copy(task_idxs_vec.begin(), task_idxs_vec.end(), ptrs.task_idxs);
        ptrs.task_idxs += task_idxs_vec.size();

        rollout_idx++;
    }

    const MoveInputs &get_move_inputs() { return move_inputs; }

    const Settings settings;
    int num_rollouts;
    MoveInputs move_inputs;
    MoveInputPtrs ptrs;
    int rollout_idx = 0;
};