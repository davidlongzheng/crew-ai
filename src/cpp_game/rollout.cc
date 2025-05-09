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
      frac_success(num_rollouts),
      win(num_rollouts)
{
}

Rollout::Rollout(const Settings &settings)
    : settings(settings), engine(std::make_unique<Engine>(settings)),
      seq_length(settings.get_seq_length()),
      max_suit_length(settings.max_suit_length()), num_suits(settings.num_suits()),
      hand_pad_size(settings.max_hand_size() + 1), max_num_tasks(settings.get_max_num_tasks())
{
    hist_player_idxs_pt.reserve(seq_length);
    hist_tricks_pt.reserve(seq_length);
    hist_cards_pt.reserve(seq_length * 2);
    hist_turns_pt.reserve(seq_length);
    hist_phases_pt.reserve(seq_length);
    hand_pt.reserve(seq_length * hand_pad_size * 2);
    player_idx_pt.reserve(seq_length);
    trick_pt.reserve(seq_length);
    turn_pt.reserve(seq_length);
    phase_pt.reserve(seq_length);
    valid_actions_pt.reserve(seq_length * hand_pad_size * 2);
    task_idxs.reserve(max_num_tasks * 2);
    actions_pt.reserve(seq_length);
    probs_pt.reserve(seq_length * hand_pad_size);
    log_probs_pt.reserve(seq_length * hand_pad_size);
    rewards_pt.reserve(seq_length);

    init_state();
}

void Rollout::init_state()
{
    hist_player_idxs_pt.push_back(-1);
    hist_tricks_pt.push_back(-1);
    hist_cards_pt.push_back(-1);
    hist_cards_pt.push_back(-1);
    hist_turns_pt.push_back(-1);
    hist_phases_pt.push_back(-1);

    encode_tasks();
}

void Rollout::reset_state(int engine_seed)
{
    engine->reset_state(engine_seed);

    hist_player_idxs_pt.clear();
    hist_tricks_pt.clear();
    hist_cards_pt.clear();
    hist_turns_pt.clear();
    hist_phases_pt.clear();
    hand_pt.clear();
    player_idx_pt.clear();
    trick_pt.clear();
    turn_pt.clear();
    phase_pt.clear();
    valid_actions_pt.clear();
    task_idxs.clear();
    actions_pt.clear();
    probs_pt.clear();
    log_probs_pt.clear();
    actions_pt.clear();
    rewards_pt.clear();

    init_state();
}

void Rollout::encode_tasks()
{
    task_idxs.resize(max_num_tasks * 2, -1);
    size_t i = 0;
    for (int player = 0; player < settings.num_players; ++player)
    {
        int player_idx = engine->state.get_player_idx(player);
        for (const auto &task : engine->state.assigned_tasks[player])
        {
            task_idxs[i] = task.task_idx;
            task_idxs[i + 1] = player_idx;
            i += 2;
        }
    }
}

void Rollout::add_card(std::vector<int8_t> &vec, const std::optional<Card> &card)
{
    if (!card.has_value())
    {
        vec.push_back(static_cast<int8_t>(max_suit_length));
        vec.push_back(static_cast<int8_t>(num_suits));
    }
    else
    {
        vec.push_back(static_cast<int8_t>(card->rank - 1));
        vec.push_back(static_cast<int8_t>(settings.get_suit_idx(card->suit)));
    }
}

void Rollout::add_hand(std::vector<int8_t> &vec, const std::vector<Card> &hand)
{
    for (const auto &card : hand)
    {
        add_card(vec, card);
    }

    for (int i = 0; i < hand_pad_size - hand.size(); ++i)
    {
        vec.push_back(-1);
        vec.push_back(-1);
    }
}

void Rollout::add_valid_actions(std::vector<int8_t> &vec, const std::vector<Action> &valid_actions)
{
    for (const auto &action : valid_actions)
    {
        add_card(vec, action.card);
    }

    for (int i = 0; i < hand_pad_size - valid_actions.size(); ++i)
    {
        vec.push_back(-1);
        vec.push_back(-1);
    }
}

void Rollout::add_probs(std::vector<float> &vec, const py::array_t<float> &probs)
{
    auto buffer = probs.data();
    vec.insert(vec.end(), buffer, buffer + probs.size());
}

void Rollout::record_move_inputs()
{
    int player_idx = engine->state.get_player_idx();
    int turn = engine->state.get_turn();

    // Record private inputs
    add_hand(hand_pt, engine->state.hands[engine->state.get_player(player_idx)]);
    player_idx_pt.push_back(player_idx);
    trick_pt.push_back(engine->state.trick);
    turn_pt.push_back(turn);
    phase_pt.push_back(engine->state.phase_idx());

    // Record valid actions
    valid_actions = engine->valid_actions();
    add_valid_actions(valid_actions_pt, valid_actions);
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
    add_probs(probs_pt, probs);
    add_probs(log_probs_pt, log_probs);

    // Record public history
    hist_player_idxs_pt.push_back(engine->state.get_player_idx());
    hist_tricks_pt.push_back(engine->state.trick);
    add_card(hist_cards_pt, action.card);
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
    hist_cards_pt.resize(hist_cards_pt.size() - 2);
    hist_turns_pt.pop_back();
    hist_phases_pt.pop_back();
}

BatchRollout::BatchRollout(const Settings &settings_, int num_rollouts_)
    : settings(settings_), num_rollouts(num_rollouts_), seq_length(settings.get_seq_length()), hand_pad_size(settings.max_hand_size() + 1), max_num_tasks(settings.get_max_num_tasks()), move_inputs(num_rollouts, hand_pad_size, max_num_tasks), results(num_rollouts, seq_length, hand_pad_size, max_num_tasks)
{
    for (int i = 0; i < num_rollouts; ++i)
    {
        rollouts.emplace_back(settings);
    }
}

void BatchRollout::reset_state(const std::vector<int> &engine_seeds)
{
    assert((int)engine_seeds.size() == num_rollouts);
    for (int i = 0; i < num_rollouts; ++i)
    {
        rollouts[i].reset_state(engine_seeds[i]);
    }
}

const MoveInputs &BatchRollout::get_move_inputs()
{
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
        std::copy(rollout.hist_cards_pt.end() - 2, rollout.hist_cards_pt.end(), hist_cards_ptr);
        hist_cards_ptr += 2;
        *hist_turns_ptr = rollout.hist_turns_pt.back();
        hist_turns_ptr++;
        *hist_phases_ptr = rollout.hist_phases_pt.back();
        hist_phases_ptr++;

        std::copy(rollout.hand_pt.end() - hand_pad_size * 2, rollout.hand_pt.end(), hand_ptr);
        hand_ptr += hand_pad_size * 2;
        *player_idx_ptr = rollout.player_idx_pt.back();
        player_idx_ptr++;
        *trick_ptr = rollout.trick_pt.back();
        trick_ptr++;
        *turn_ptr = rollout.turn_pt.back();
        turn_ptr++;
        *phase_ptr = rollout.phase_pt.back();
        phase_ptr++;

        std::copy(rollout.valid_actions_pt.end() - hand_pad_size * 2, rollout.valid_actions_pt.end(), valid_actions_ptr);
        valid_actions_ptr += hand_pad_size * 2;

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

const RolloutResults &BatchRollout::get_results()
{
    for (auto &rollout : rollouts)
    {
        rollout.pop_last_history();
    }

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
    auto *actions_ptr = static_cast<long *>(results.actions.mutable_data());
    auto *rewards_ptr = static_cast<float *>(results.rewards.mutable_data());
    auto *frac_success_ptr = static_cast<float *>(results.frac_success.mutable_data());
    auto *win_ptr = static_cast<bool *>(results.win.mutable_data());

    for (int rollout_idx = 0; rollout_idx < num_rollouts; ++rollout_idx)
    {
        const auto &rollout = rollouts[rollout_idx];
        assert(rollout.hist_player_idxs_pt.size() == seq_length);
        assert(rollout.hist_tricks_pt.size() == seq_length);
        assert(rollout.hist_cards_pt.size() == seq_length * 2);
        assert(rollout.hist_turns_pt.size() == seq_length);
        assert(rollout.hist_phases_pt.size() == seq_length);
        assert(rollout.hand_pt.size() == seq_length * hand_pad_size * 2);
        assert(rollout.player_idx_pt.size() == seq_length);
        assert(rollout.trick_pt.size() == seq_length);
        assert(rollout.turn_pt.size() == seq_length);
        assert(rollout.phase_pt.size() == seq_length);
        assert(rollout.valid_actions_pt.size() == seq_length * hand_pad_size * 2);
        assert(rollout.probs_pt.size() == seq_length * hand_pad_size);
        assert(rollout.log_probs_pt.size() == seq_length * hand_pad_size);
        assert(rollout.actions_pt.size() == seq_length);
        assert(rollout.rewards_pt.size() == seq_length);

        std::copy(rollout.hist_player_idxs_pt.begin(), rollout.hist_player_idxs_pt.end(), hist_player_idxs_ptr);
        hist_player_idxs_ptr += rollout.hist_player_idxs_pt.size();

        std::copy(rollout.hist_tricks_pt.begin(), rollout.hist_tricks_pt.end(), hist_tricks_ptr);
        hist_tricks_ptr += rollout.hist_tricks_pt.size();

        std::copy(rollout.hist_cards_pt.begin(), rollout.hist_cards_pt.end(), hist_cards_ptr);
        hist_cards_ptr += seq_length * 2;

        std::copy(rollout.hist_turns_pt.begin(), rollout.hist_turns_pt.end(), hist_turns_ptr);
        hist_turns_ptr += rollout.hist_turns_pt.size();

        std::copy(rollout.hist_phases_pt.begin(), rollout.hist_phases_pt.end(), hist_phases_ptr);
        hist_phases_ptr += rollout.hist_phases_pt.size();

        std::copy(rollout.hand_pt.begin(), rollout.hand_pt.end(), hand_ptr);
        hand_ptr += seq_length * hand_pad_size * 2;

        std::copy(rollout.player_idx_pt.begin(), rollout.player_idx_pt.end(), player_idx_ptr);
        player_idx_ptr += rollout.player_idx_pt.size();

        std::copy(rollout.trick_pt.begin(), rollout.trick_pt.end(), trick_ptr);
        trick_ptr += rollout.trick_pt.size();

        std::copy(rollout.turn_pt.begin(), rollout.turn_pt.end(), turn_ptr);
        turn_ptr += rollout.turn_pt.size();

        std::copy(rollout.phase_pt.begin(), rollout.phase_pt.end(), phase_ptr);
        phase_ptr += rollout.phase_pt.size();

        std::copy(rollout.valid_actions_pt.begin(), rollout.valid_actions_pt.end(), valid_actions_ptr);
        valid_actions_ptr += seq_length * hand_pad_size * 2;

        for (int i = 0; i < seq_length; ++i)
        {
            std::copy(rollout.task_idxs.begin(), rollout.task_idxs.end(), task_idxs_ptr);
            task_idxs_ptr += rollout.max_num_tasks * 2;
        }

        std::copy(rollout.probs_pt.begin(), rollout.probs_pt.end(), probs_ptr);
        probs_ptr += seq_length * hand_pad_size;

        std::copy(rollout.log_probs_pt.begin(), rollout.log_probs_pt.end(), log_probs_ptr);
        log_probs_ptr += seq_length * hand_pad_size;

        std::copy(rollout.actions_pt.begin(), rollout.actions_pt.end(), actions_ptr);
        actions_ptr += rollout.actions_pt.size();

        std::copy(rollout.rewards_pt.begin(), rollout.rewards_pt.end(), rewards_ptr);
        rewards_ptr += rollout.rewards_pt.size();

        // Calculate success metrics
        bool win = rollout.engine->state.status == Status::kSuccess;
        *win_ptr = win;
        win_ptr++;

        int8_t num_success = 0, num_tasks = 0;
        for (int player_idx = 0; player_idx < settings.num_players; ++player_idx)
        {
            int player = rollout.engine->state.get_player(player_idx);
            num_success += std::count_if(
                rollout.engine->state.assigned_tasks[player].begin(),
                rollout.engine->state.assigned_tasks[player].end(),
                [](const auto &task)
                { return task.status == Status::kSuccess; });
            num_tasks += rollout.engine->state.assigned_tasks[player].size();
        }

        *frac_success_ptr = num_success / (float)num_tasks;
        frac_success_ptr++;
    }

    return results;
}