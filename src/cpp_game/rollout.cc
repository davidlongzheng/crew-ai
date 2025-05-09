#include "rollout.h"
#include <iostream>
#include <random>
#include <algorithm>

MoveInputs::MoveInputs(int num_rollouts, int hand_pad_size, int max_num_tasks)
    : hist_player_idxs(num_rollouts),
      hist_tricks(num_rollouts),
      hist_cards({num_rollouts, 2}),
      hist_turns(num_rollouts),
      hist_phases(num_rollouts),
      hand({num_rollouts, hand_pad_size, 2}),
      player_idx(num_rollouts),
      trick(num_rollouts),
      turn(num_rollouts),
      phase(num_rollouts),
      valid_actions({num_rollouts, hand_pad_size, 2}),
      task_idxs({num_rollouts, max_num_tasks, 2})
{
}

RolloutResults::RolloutResults(int num_rollouts, int seq_length, int hand_pad_size, int max_num_tasks)
    : hist_player_idxs({num_rollouts, seq_length}),
      hist_tricks({num_rollouts, seq_length}),
      hist_cards({num_rollouts, seq_length, 2}),
      hist_turns({num_rollouts, seq_length}),
      hist_phases({num_rollouts, seq_length}),
      hand({num_rollouts, seq_length, hand_pad_size, 2}),
      player_idx({num_rollouts, seq_length}),
      trick({num_rollouts, seq_length}),
      turn({num_rollouts, seq_length}),
      phase({num_rollouts, seq_length}),
      valid_actions({num_rollouts, seq_length, hand_pad_size, 2}),
      task_idxs({num_rollouts, seq_length, max_num_tasks, 2}),
      probs({num_rollouts, seq_length, hand_pad_size}),
      log_probs({num_rollouts, seq_length, hand_pad_size}),
      actions({num_rollouts, seq_length}),
      rewards({num_rollouts, seq_length}),
      num_success_tasks_pp({num_rollouts, seq_length, 2}),
      win({num_rollouts, seq_length})
{
}

Rollout::Rollout(const Settings &settings, int engine_seed)
    : settings(settings), engine(std::make_unique<Engine>(settings, engine_seed)),
      max_suit_length(settings.max_suit_length()), num_suits(settings.num_suits()),
      hand_pad_size(settings.max_hand_size() + 1),
      max_num_tasks(settings.get_max_num_tasks())
{
    hist_player_idxs_pt.push_back(-1);
    hist_tricks_pt.push_back(-1);
    hist_cards_pt.push_back({-1, -1});
    hist_turns_pt.push_back(-1);
    hist_phases_pt.push_back(-1);

    task_idxs = encode_tasks();
}

std::vector<int8_t> Rollout::encode_tasks() const
{
    std::vector<int8_t> task_idxs_(max_num_tasks * 2, -1);
    size_t i = 0;
    for (int player = 0; player < settings.num_players; ++player)
    {
        int player_idx = engine->state.get_player_idx(player);
        for (const auto &task : engine->state.assigned_tasks[player])
        {
            task_idxs_[i] = task.task_idx;
            task_idxs_[i + 1] = player_idx;
            i += 2;
        }
    }
    return task_idxs_;
}

std::array<int8_t, 2> Rollout::card_to_arr(const std::optional<Card> &card) const
{
    if (!card.has_value())
    {
        return {static_cast<int8_t>(max_suit_length), static_cast<int8_t>(num_suits)};
    }
    else
    {
        return {static_cast<int8_t>(card->rank - 1), static_cast<int8_t>(settings.get_suit_idx(card->suit))};
    }
}

std::vector<int8_t> Rollout::encode_hand(const std::vector<Card> &hand) const
{
    size_t num_cards = hand.size();

    // Allocate buffer for all cards (including padding)
    std::vector<int8_t> result(hand_pad_size * 2, -1);

    // Fill with actual cards
    for (size_t i = 0; i < num_cards; ++i)
    {
        auto card_data = card_to_arr(hand[i]);
        result[i * 2] = card_data[0];
        result[i * 2 + 1] = card_data[1];
    }

    return result;
}

std::vector<int8_t> Rollout::encode_valid_actions(const std::vector<Action> &valid_actions) const
{
    size_t num_actions = valid_actions.size();

    // Allocate buffer for all cards (including padding)
    std::vector<int8_t> result(hand_pad_size * 2, -1);

    // Fill with actual cards
    for (size_t i = 0; i < num_actions; ++i)
    {
        auto card_data = card_to_arr(valid_actions[i].card);
        result[i * 2] = card_data[0];
        result[i * 2 + 1] = card_data[1];
    }

    return result;
}

std::vector<float> py_arr_to_vec(const py::array_t<float> &arr)
{
    auto buffer = arr.request();
    return std::vector<float>(static_cast<float *>(buffer.ptr), static_cast<float *>(buffer.ptr) + buffer.size);
}

void Rollout::record_move_inputs()
{
    int player_idx = engine->state.get_player_idx();
    int turn = engine->state.get_turn();

    // Record private inputs
    hand_pt.push_back(encode_hand(engine->state.hands[engine->state.get_player(player_idx)]));
    player_idx_pt.push_back(player_idx);
    trick_pt.push_back(engine->state.trick);
    turn_pt.push_back(turn);
    phase_pt.push_back(engine->state.phase_idx());

    // Record valid actions
    valid_actions = engine->valid_actions();
    valid_actions_pt.push_back(encode_valid_actions(valid_actions));
}

void Rollout::move(int action_idx, const py::array_t<float> &probs, const py::array_t<float> &log_probs)
{
    if (engine->state.phase == Phase::kEnd)
    {
        return;
    }

    assert(action_idx < (int)valid_actions.size());
    auto action = valid_actions[action_idx];

    // Record the action and its probability
    actions_pt.push_back(action_idx);
    probs_pt.push_back(py_arr_to_vec(probs));
    log_probs_pt.push_back(py_arr_to_vec(log_probs));

    // Record public history
    hist_player_idxs_pt.push_back(engine->state.get_player_idx());
    hist_tricks_pt.push_back(engine->state.trick);
    hist_cards_pt.push_back(card_to_arr(action.card));
    hist_turns_pt.push_back(engine->state.get_turn());
    hist_phases_pt.push_back(engine->state.phase_idx());

    // Execute the action and record reward
    float reward = engine->move(action);
    rewards_pt.push_back(reward);
}

void Rollout::pop_last_history()
{
    hist_player_idxs_pt.pop_back();
    hist_tricks_pt.pop_back();
    hist_cards_pt.pop_back();
    hist_turns_pt.pop_back();
    hist_phases_pt.pop_back();
}

BatchRollout::BatchRollout(const Settings &settings_, int num_rollouts_, const std::vector<int> engine_seeds)
    : settings(settings_), num_rollouts(num_rollouts_)
{
    assert((int)engine_seeds.size() == num_rollouts_);
    for (int seed : engine_seeds)
    {
        rollouts.emplace_back(settings, seed);
    }
}

MoveInputs BatchRollout::get_move_inputs()
{

    MoveInputs move_inputs(num_rollouts, rollouts[0].hand_pad_size, rollouts[0].max_num_tasks);
    auto *hist_player_idxs_ptr = static_cast<int8_t *>(move_inputs.hist_player_idxs.mutable_data());
    auto *hist_tricks_ptr = static_cast<int8_t *>(move_inputs.hist_tricks.mutable_data());
    auto *hist_cards_ptr = static_cast<int8_t *>(move_inputs.hist_cards.mutable_data());
    auto *hist_turns_ptr = static_cast<int8_t *>(move_inputs.hist_turns.mutable_data());
    auto *hist_phases_ptr = static_cast<int8_t *>(move_inputs.hist_phases.mutable_data());

    auto *hand_ptr = static_cast<int8_t *>(move_inputs.hand.mutable_data());
    auto *player_idx_ptr = static_cast<int8_t *>(move_inputs.player_idx.mutable_data());
    auto *trick_ptr = static_cast<int8_t *>(move_inputs.trick.mutable_data());
    auto *turn_ptr = static_cast<int8_t *>(move_inputs.turn.mutable_data());
    auto *phase_ptr = static_cast<int8_t *>(move_inputs.phase.mutable_data());

    auto *valid_actions_ptr = static_cast<int8_t *>(move_inputs.valid_actions.mutable_data());
    auto *task_idxs_ptr = static_cast<int8_t *>(move_inputs.task_idxs.mutable_data());

    // Collect arrays from each rollout
    for (auto &rollout : rollouts)
    {
        // Update the rollout's arrays if game is not over
        if (rollout.engine->state.phase != Phase::kEnd)
        {
            rollout.record_move_inputs();
        }

        *hist_player_idxs_ptr = rollout.hist_player_idxs_pt.back();
        hist_player_idxs_ptr++;
        *hist_tricks_ptr = rollout.hist_tricks_pt.back();
        hist_tricks_ptr++;
        std::copy(rollout.hist_cards_pt.back().begin(), rollout.hist_cards_pt.back().end(), hist_cards_ptr);
        hist_cards_ptr += 2;
        *hist_turns_ptr = rollout.hist_turns_pt.back();
        hist_turns_ptr++;
        *hist_phases_ptr = rollout.hist_phases_pt.back();
        hist_phases_ptr++;

        std::copy(rollout.hand_pt.back().begin(), rollout.hand_pt.back().end(), hand_ptr);
        hand_ptr += rollout.hand_pad_size * 2;
        *player_idx_ptr = rollout.player_idx_pt.back();
        player_idx_ptr++;
        *trick_ptr = rollout.trick_pt.back();
        trick_ptr++;
        *turn_ptr = rollout.turn_pt.back();
        turn_ptr++;
        *phase_ptr = rollout.phase_pt.back();
        phase_ptr++;

        std::copy(rollout.valid_actions_pt.back().begin(), rollout.valid_actions_pt.back().end(), valid_actions_ptr);
        valid_actions_ptr += rollout.hand_pad_size * 2;

        std::copy(rollout.task_idxs.begin(), rollout.task_idxs.end(), task_idxs_ptr);
        task_idxs_ptr += rollout.max_num_tasks * 2;
    }

    return move_inputs;
}

void BatchRollout::move(const py::array_t<int8_t> &action_indices, const py::array_t<float> &probs, const py::array_t<float> &log_probs)
{
    for (int rollout_idx = 0; rollout_idx < num_rollouts; ++rollout_idx)
    {
        // Extract single action, probs, and log_probs for this rollout
        int action_idx = action_indices.at(rollout_idx);

        // Create slice views of probs and log_probs
        py::slice rollout_slice(rollout_idx, rollout_idx + 1, 1);
        py::array_t<float> probs_slice = probs[rollout_slice].cast<py::array_t<float>>();
        py::array_t<float> log_probs_slice = log_probs[rollout_slice].cast<py::array_t<float>>();

        rollouts[rollout_idx].move(action_idx, probs_slice, log_probs_slice);
    }
}

bool BatchRollout::is_done() const
{
    return std::all_of(rollouts.begin(), rollouts.end(),
                       [](const auto &rollout)
                       { return rollout.engine->state.phase == Phase::kEnd; });
}

RolloutResults BatchRollout::get_results()
{
    for (auto &rollout : rollouts)
    {
        rollout.pop_last_history();
    }

    size_t seq_length = rollouts[0].hist_player_idxs_pt.size();
    RolloutResults results(num_rollouts, seq_length, rollouts[0].hand_pad_size, rollouts[0].max_num_tasks);
    auto *hist_player_idxs_ptr = static_cast<int8_t *>(results.hist_player_idxs.mutable_data());
    auto *hist_tricks_ptr = static_cast<int8_t *>(results.hist_tricks.mutable_data());
    auto *hist_cards_ptr = static_cast<int8_t *>(results.hist_cards.mutable_data());
    auto *hist_turns_ptr = static_cast<int8_t *>(results.hist_turns.mutable_data());
    auto *hist_phases_ptr = static_cast<int8_t *>(results.hist_phases.mutable_data());

    auto *hand_ptr = static_cast<int8_t *>(results.hand.mutable_data());
    auto *player_idx_ptr = static_cast<int8_t *>(results.player_idx.mutable_data());
    auto *trick_ptr = static_cast<int8_t *>(results.trick.mutable_data());
    auto *turn_ptr = static_cast<int8_t *>(results.turn.mutable_data());
    auto *phase_ptr = static_cast<int8_t *>(results.phase.mutable_data());

    auto *valid_actions_ptr = static_cast<int8_t *>(results.valid_actions.mutable_data());
    auto *task_idxs_ptr = static_cast<int8_t *>(results.task_idxs.mutable_data());

    auto *probs_ptr = static_cast<float *>(results.probs.mutable_data());
    auto *log_probs_ptr = static_cast<float *>(results.log_probs.mutable_data());
    auto *actions_ptr = static_cast<int8_t *>(results.actions.mutable_data());
    auto *rewards_ptr = static_cast<float *>(results.rewards.mutable_data());
    auto *num_success_tasks_pp_ptr = static_cast<int8_t *>(results.num_success_tasks_pp.mutable_data());
    auto *win_ptr = static_cast<bool *>(results.win.mutable_data());

    for (int rollout_idx = 0; rollout_idx < num_rollouts; ++rollout_idx)
    {
        const auto &rollout = rollouts[rollout_idx];
        assert(rollout.hist_player_idxs_pt.size() == seq_length);
        assert(rollout.hist_tricks_pt.size() == seq_length);
        assert(rollout.hist_cards_pt.size() == seq_length);
        assert(rollout.hist_turns_pt.size() == seq_length);
        assert(rollout.hist_phases_pt.size() == seq_length);
        assert(rollout.hand_pt.size() == seq_length);
        assert(rollout.player_idx_pt.size() == seq_length);
        assert(rollout.trick_pt.size() == seq_length);
        assert(rollout.turn_pt.size() == seq_length);
        assert(rollout.phase_pt.size() == seq_length);
        assert(rollout.valid_actions_pt.size() == seq_length);
        assert(rollout.probs_pt.size() == seq_length);
        assert(rollout.log_probs_pt.size() == seq_length);
        assert(rollout.actions_pt.size() == seq_length);
        assert(rollout.rewards_pt.size() == seq_length);

        std::copy(rollout.hist_player_idxs_pt.begin(), rollout.hist_player_idxs_pt.end(), hist_player_idxs_ptr);
        hist_player_idxs_ptr += rollout.hist_player_idxs_pt.size();

        std::copy(rollout.hist_tricks_pt.begin(), rollout.hist_tricks_pt.end(), hist_tricks_ptr);
        hist_tricks_ptr += rollout.hist_tricks_pt.size();

        for (int i = 0; i < seq_length; ++i)
        {
            std::copy(rollout.hist_cards_pt[i].begin(), rollout.hist_cards_pt[i].end(), hist_cards_ptr);
            hist_cards_ptr += 2;
        }

        std::copy(rollout.hist_turns_pt.begin(), rollout.hist_turns_pt.end(), hist_turns_ptr);
        hist_turns_ptr += rollout.hist_turns_pt.size();

        std::copy(rollout.hist_phases_pt.begin(), rollout.hist_phases_pt.end(), hist_phases_ptr);
        hist_phases_ptr += rollout.hist_phases_pt.size();

        for (int i = 0; i < seq_length; ++i)
        {
            std::copy(rollout.hand_pt[i].begin(), rollout.hand_pt[i].end(), hand_ptr);
            hand_ptr += rollout.hand_pad_size * 2;
        }

        *player_idx_ptr = rollout.player_idx_pt.back();
        player_idx_ptr++;

        *trick_ptr = rollout.trick_pt.back();
        trick_ptr++;

        *turn_ptr = rollout.turn_pt.back();
        turn_ptr++;

        *phase_ptr = rollout.phase_pt.back();
        phase_ptr++;

        for (int i = 0; i < seq_length; ++i)
        {
            std::copy(rollout.valid_actions_pt[i].begin(), rollout.valid_actions_pt[i].end(), valid_actions_ptr);
            valid_actions_ptr += rollout.hand_pad_size * 2;
        }

        for (int i = 0; i < seq_length; ++i)
        {
            std::copy(rollout.task_idxs.begin(), rollout.task_idxs.end(), task_idxs_ptr);
            task_idxs_ptr += rollout.max_num_tasks * 2;
        }

        for (int i = 0; i < seq_length; ++i)
        {
            std::copy(rollout.probs_pt[i].begin(), rollout.probs_pt[i].end(), probs_ptr);
            probs_ptr += rollout.probs_pt[i].size();
        }

        for (int i = 0; i < seq_length; ++i)
        {
            std::copy(rollout.log_probs_pt[i].begin(), rollout.log_probs_pt[i].end(), log_probs_ptr);
            log_probs_ptr += rollout.log_probs_pt[i].size();
        }

        std::copy(rollout.actions_pt.begin(), rollout.actions_pt.end(), actions_ptr);
        actions_ptr += rollout.actions_pt.size();

        std::copy(rollout.rewards_pt.begin(), rollout.rewards_pt.end(), rewards_ptr);
        rewards_ptr += rollout.rewards_pt.size();

        // Calculate success metrics
        bool win = rollout.engine->state.status == Status::kSuccess;
        *win_ptr = win;
        win_ptr++;

        std::vector<int8_t> num_success_tasks;
        for (int player_idx = 0; player_idx < settings.num_players; ++player_idx)
        {
            int player = rollout.engine->state.get_player(player_idx);
            int num_success = std::count_if(
                rollout.engine->state.assigned_tasks[player].begin(),
                rollout.engine->state.assigned_tasks[player].end(),
                [](const auto &task)
                { return task.status == Status::kSuccess; });
            int num_tasks = rollout.engine->state.assigned_tasks[player].size();

            num_success_tasks.push_back(num_success);
            num_success_tasks.push_back(num_tasks);
        }

        std::copy(num_success_tasks.begin(), num_success_tasks.end(), num_success_tasks_pp_ptr);
        num_success_tasks_pp_ptr += num_success_tasks.size();
    }

    return results;
}