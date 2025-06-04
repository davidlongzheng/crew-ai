#include "rollout.h"
#include <iostream>
#include <random>
#include <algorithm>

MoveInputs::MoveInputs(int num_rollouts, int max_hand_size, int max_num_actions, int max_num_tasks)
    : hist_player_idx(num_rollouts),
      hist_trick(num_rollouts),
      hist_action({num_rollouts, 2}),
      hist_turn(num_rollouts),
      hist_phase(num_rollouts),
      hand({num_rollouts, max_hand_size, 2}),
      player_idx(num_rollouts),
      trick(num_rollouts),
      turn(num_rollouts),
      phase(num_rollouts),
      task_idxs({num_rollouts, max_num_tasks, 2}),
      valid_actions({num_rollouts, max_num_actions, 2})
{
}

RolloutResults::RolloutResults(int num_rollouts, int seq_length, int max_hand_size, int max_num_actions, int max_num_tasks, int num_cards)
    : hist_player_idx({num_rollouts, seq_length}),
      hist_trick({num_rollouts, seq_length}),
      hist_action({num_rollouts, seq_length, 2}),
      hist_turn({num_rollouts, seq_length}),
      hist_phase({num_rollouts, seq_length}),
      hand({num_rollouts, seq_length, max_hand_size, 2}),
      player_idx({num_rollouts, seq_length}),
      trick({num_rollouts, seq_length}),
      turn({num_rollouts, seq_length}),
      phase({num_rollouts, seq_length}),
      task_idxs({num_rollouts, seq_length, max_num_tasks, 2}),
      valid_actions({num_rollouts, seq_length, max_num_actions, 2}),
      log_probs({num_rollouts, seq_length, max_num_actions}),
      actions({num_rollouts, seq_length}),
      rewards({num_rollouts, seq_length}),
      task_idxs_no_pt({num_rollouts, max_num_tasks}),
      task_success({num_rollouts, max_num_tasks}),
      difficulty(num_rollouts),
      win(num_rollouts),
      aux_info({num_rollouts, num_cards})
{
}

Rollout::Rollout(const Settings &settings)
    : settings(settings), engine(std::make_unique<Engine>(settings)),
      seq_length(settings.seq_length),
      max_suit_length(settings.max_suit_length),
      num_suits(settings.num_suits),
      max_hand_size(settings.max_hand_size),
      max_num_actions(settings.max_num_actions),
      max_num_tasks(settings.resolved_max_num_tasks)
{
    hist_player_idx_pt.reserve(seq_length);
    hist_trick_pt.reserve(seq_length);
    hist_action_pt.reserve(seq_length * 2);
    hist_turn_pt.reserve(seq_length);
    hist_phase_pt.reserve(seq_length);
    hand_pt.reserve(seq_length * max_hand_size * 2);
    player_idx_pt.reserve(seq_length);
    trick_pt.reserve(seq_length);
    turn_pt.reserve(seq_length);
    phase_pt.reserve(seq_length);
    valid_actions_pt.reserve(seq_length * max_num_actions * 2);
    task_idxs_pt.reserve(seq_length * max_num_tasks * 2);
    actions_pt.reserve(seq_length);
    log_probs_pt.reserve(seq_length * max_num_actions);
    rewards_pt.reserve(seq_length);

    init_state();
}

void Rollout::init_state()
{
    encode_aux_info();
}

void Rollout::reset_state(int engine_seed)
{
    engine->reset_state(engine_seed);

    hist_player_idx_pt.clear();
    hist_trick_pt.clear();
    hist_action_pt.clear();
    hist_turn_pt.clear();
    hist_phase_pt.clear();
    hand_pt.clear();
    player_idx_pt.clear();
    trick_pt.clear();
    turn_pt.clear();
    phase_pt.clear();
    valid_actions_pt.clear();
    task_idxs_pt.clear();
    actions_pt.clear();
    log_probs_pt.clear();
    actions_pt.clear();
    rewards_pt.clear();
    aux_info.clear();

    init_state();
}

void Rollout::add_tasks(std::vector<int8_t> &vec, const Settings &settings, const State &state)
{
    int N = vec.size();
    vec.resize(N + settings.resolved_max_num_tasks * 2, -1);
    size_t i = N;
    for (int task_idx : state.unassigned_task_idxs)
    {
        vec[i] = task_idx;
        vec[i + 1] = settings.num_players;
        i += 2;
    }

    for (int player = 0; player < settings.num_players; ++player)
    {
        int player_idx = state.get_player_idx(player);
        for (const auto &task : state.assigned_tasks[player])
        {
            vec[i] = task.task_idx;
            vec[i + 1] = player_idx;
            i += 2;
        }
    }
}

void Rollout::encode_aux_info()
{
    aux_info.resize(settings.num_cards);
    for (int player = 0; player < settings.num_players; ++player)
    {
        int player_idx = engine->state.get_player_idx(player);
        for (auto &card : engine->state.hands[player])
        {
            int card_idx = settings.get_suit_idx(card.suit) * settings.side_suit_length + (card.rank - 1);
            aux_info[card_idx] = player_idx;
        }
    }
}

int Rollout::add_card(std::vector<int8_t> &vec, int i, const Card &card, const Settings &settings)
{
    vec[i] = static_cast<int8_t>(card.rank - 1);
    vec[i + 1] = static_cast<int8_t>(settings.get_suit_idx(card.suit));
    return i + 2;
}

int Rollout::add_action(std::vector<int8_t> &vec, int i, const Action &action, const Settings &settings)
{
    switch (action.type)
    {
    case ActionType::kSignal:
    case ActionType::kPlay:
        add_card(vec, i, action.card.value(), settings);
        break;
    case ActionType::kNoSignal:
        vec[i] = settings.max_suit_length;
        vec[i + 1] = settings.num_suits;
        break;
    case ActionType::kDraft:
        vec[i] = action.task_idx.value();
        break;
    case ActionType::kNoDraft:
        vec[i] = settings.num_task_defs;
        break;
    default:
        assert(false);
    }

    return i + 2;
}

void Rollout::add_hand(std::vector<int8_t> &vec, const std::vector<Card> &hand, const Settings &settings)
{
    int N = vec.size();
    vec.resize(N + settings.max_hand_size * 2, -1);
    int i = N;
    for (const auto &card : hand)
    {
        i = add_card(vec, i, card, settings);
    }
}

void Rollout::add_valid_actions(std::vector<int8_t> &vec, const std::vector<Action> &valid_actions, const Settings &settings)
{
    int N = vec.size();
    vec.resize(N + settings.max_num_actions * 2, -1);
    int i = N;
    for (const auto &action : valid_actions)
    {
        i = add_action(vec, i, action, settings);
    }
}

void Rollout::add_log_probs(const py::array_t<float> &log_probs)
{
    auto buffer = log_probs.data();
    log_probs_pt.insert(log_probs_pt.end(), buffer, buffer + log_probs.size());
}

void Rollout::record_move_inputs()
{
    int player_idx = engine->state.get_player_idx();
    int turn = engine->state.get_turn();

    // Record private inputs
    add_hand(hand_pt, engine->state.hands[engine->state.get_player(player_idx)], settings);
    player_idx_pt.push_back(player_idx);
    trick_pt.push_back(engine->state.trick);
    turn_pt.push_back(turn);
    phase_pt.push_back(settings.get_phase_idx(engine->state.phase));
    add_tasks(task_idxs_pt, settings, engine->state);

    // Record valid actions
    valid_actions = engine->valid_actions();
    add_valid_actions(valid_actions_pt, valid_actions, settings);

    // Record public history
    if (engine->state.last_action)
    {
        const auto [hist_player_idx, hist_trick, hist_action, hist_turn, hist_phase] = *engine->state.last_action;
        hist_player_idx_pt.push_back(hist_player_idx);
        hist_trick_pt.push_back(hist_trick);
        int N = hist_action_pt.size();
        hist_action_pt.resize(N + 2, -1);
        add_action(hist_action_pt, N, hist_action, settings);
        hist_turn_pt.push_back(hist_turn);
        hist_phase_pt.push_back(hist_phase);
    }
    else
    {
        hist_player_idx_pt.push_back(-1);
        hist_trick_pt.push_back(-1);
        int N = hist_action_pt.size();
        hist_action_pt.resize(N + 2, -1);
        hist_turn_pt.push_back(-1);
        hist_phase_pt.push_back(-1);
    }
}

void Rollout::move(int action_idx)
{
    if (engine->state.phase == Phase::kEnd)
    {
        return;
    }

    assert(action_idx < (int)valid_actions.size());
    auto action = valid_actions[action_idx];

    // Record the action and its probability
    actions_pt.push_back(action_idx);

    // Execute the action and record reward
    float reward = engine->move(action);
    rewards_pt.push_back(reward);
}

BatchRollout::BatchRollout(const Settings &settings_, int num_rollouts_, int num_threads)
    : settings(settings_), num_rollouts(num_rollouts_), seq_length(settings.seq_length), max_hand_size(settings.max_hand_size), max_num_actions(settings.max_num_actions), max_num_tasks(settings.resolved_max_num_tasks), move_inputs(num_rollouts, max_hand_size, max_num_actions, max_num_tasks), results(num_rollouts, seq_length, max_hand_size, max_num_actions, max_num_tasks, settings.num_cards)
{
    for (int i = 0; i < num_rollouts; ++i)
    {
        rollouts.emplace_back(settings);
    }

    if (num_threads > 1)
    {
        std::vector<std::vector<std::function<void()>>> tasks(num_threads, std::vector<std::function<void()>>(2));
        for (int i = 0; i < num_threads; ++i)
        {
            const int start = i * num_rollouts / num_threads;
            const int end = i < num_threads - 1 ? (i + 1) * num_rollouts / num_threads : num_rollouts;
            tasks[i][0] = [this, start, end]()
            {
                for (int j = start; j < end; ++j)
                {
                    rollouts[j].record_move_inputs();
                }
            };
            tasks[i][1] = [this, start, end]()
            {
                for (int j = start; j < end; ++j)
                {
                    rollouts[j].move(action_idxs[j]);
                }
            };
        }
        pool = std::make_unique<FixedThreadPool>(tasks);
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
    if (pool)
    {
        pool->run(0);
    }
    else
    {
        for (auto &rollout : rollouts)
        {
            // Update the rollout's arrays if game is not over
            assert(rollout.engine->state.phase != Phase::kEnd);
            rollout.record_move_inputs();
        }
    }

    auto *hist_player_idx_ptr = static_cast<int8_t *>(move_inputs.hist_player_idx.mutable_data());
    auto *hist_trick_ptr = static_cast<int8_t *>(move_inputs.hist_trick.mutable_data());
    auto *hist_action_ptr = static_cast<int8_t *>(move_inputs.hist_action.mutable_data());
    auto *hist_turn_ptr = static_cast<int8_t *>(move_inputs.hist_turn.mutable_data());
    auto *hist_phase_ptr = static_cast<int8_t *>(move_inputs.hist_phase.mutable_data());

    auto *hand_ptr = static_cast<int8_t *>(move_inputs.hand.mutable_data());
    auto *player_idx_ptr = static_cast<int8_t *>(move_inputs.player_idx.mutable_data());
    auto *trick_ptr = static_cast<int8_t *>(move_inputs.trick.mutable_data());
    auto *turn_ptr = static_cast<int8_t *>(move_inputs.turn.mutable_data());
    auto *phase_ptr = static_cast<int8_t *>(move_inputs.phase.mutable_data());
    auto *task_idxs_ptr = static_cast<int8_t *>(move_inputs.task_idxs.mutable_data());

    auto *valid_actions_ptr = static_cast<int8_t *>(move_inputs.valid_actions.mutable_data());

    // Collect arrays from each rollout
    for (auto &rollout : rollouts)
    {
        *hist_player_idx_ptr = rollout.hist_player_idx_pt.back();
        hist_player_idx_ptr++;
        *hist_trick_ptr = rollout.hist_trick_pt.back();
        hist_trick_ptr++;
        std::copy(rollout.hist_action_pt.end() - 2, rollout.hist_action_pt.end(), hist_action_ptr);
        hist_action_ptr += 2;
        *hist_turn_ptr = rollout.hist_turn_pt.back();
        hist_turn_ptr++;
        *hist_phase_ptr = rollout.hist_phase_pt.back();
        hist_phase_ptr++;

        std::copy(rollout.hand_pt.end() - max_hand_size * 2, rollout.hand_pt.end(), hand_ptr);
        hand_ptr += max_hand_size * 2;
        *player_idx_ptr = rollout.player_idx_pt.back();
        player_idx_ptr++;
        *trick_ptr = rollout.trick_pt.back();
        trick_ptr++;
        *turn_ptr = rollout.turn_pt.back();
        turn_ptr++;
        *phase_ptr = rollout.phase_pt.back();
        phase_ptr++;

        std::copy(rollout.valid_actions_pt.end() - max_num_actions * 2, rollout.valid_actions_pt.end(), valid_actions_ptr);
        valid_actions_ptr += max_num_actions * 2;

        std::copy(rollout.task_idxs_pt.end() - max_num_tasks * 2, rollout.task_idxs_pt.end(), task_idxs_ptr);
        task_idxs_ptr += max_num_tasks * 2;
    }

    return move_inputs;
}

void BatchRollout::move(const py::array_t<int8_t> &action_indices, const py::array_t<float> &log_probs)
{
    action_idxs.resize(action_indices.size());
    std::copy(action_indices.data(), action_indices.data() + action_indices.size(), action_idxs.begin());
    if (pool)
    {
        pool->run(1);
    }
    else
    {
        for (int rollout_idx = 0; rollout_idx < num_rollouts; ++rollout_idx)
        {
            int action_idx = action_idxs[rollout_idx];
            rollouts[rollout_idx].move(action_idx);
        }
    }

    for (int rollout_idx = 0; rollout_idx < num_rollouts; ++rollout_idx)
    {
        auto &rollout = rollouts[rollout_idx];
        py::slice rollout_slice(rollout_idx, rollout_idx + 1, 1);
        py::array_t<float> log_probs_slice = log_probs[rollout_slice].cast<py::array_t<float>>();
        rollout.add_log_probs(log_probs_slice);
    }
}

bool BatchRollout::is_done() const
{
    bool done = rollouts[0].engine->state.phase == Phase::kEnd;

    assert(std::all_of(rollouts.begin(), rollouts.end(),
                       [done](const auto &rollout)
                       { return (rollout.engine->state.phase == Phase::kEnd) == done; }));

    return done;
}

const RolloutResults &BatchRollout::get_results()
{
    auto *hist_player_idx_ptr = static_cast<int8_t *>(results.hist_player_idx.mutable_data());
    auto *hist_trick_ptr = static_cast<int8_t *>(results.hist_trick.mutable_data());
    auto *hist_action_ptr = static_cast<int8_t *>(results.hist_action.mutable_data());
    auto *hist_turn_ptr = static_cast<int8_t *>(results.hist_turn.mutable_data());
    auto *hist_phase_ptr = static_cast<int8_t *>(results.hist_phase.mutable_data());

    auto *hand_ptr = static_cast<int8_t *>(results.hand.mutable_data());
    auto *player_idx_ptr = static_cast<int8_t *>(results.player_idx.mutable_data());
    auto *trick_ptr = static_cast<int8_t *>(results.trick.mutable_data());
    auto *turn_ptr = static_cast<int8_t *>(results.turn.mutable_data());
    auto *phase_ptr = static_cast<int8_t *>(results.phase.mutable_data());
    auto *task_idxs_ptr = static_cast<int8_t *>(results.task_idxs.mutable_data());

    auto *valid_actions_ptr = static_cast<int8_t *>(results.valid_actions.mutable_data());

    auto *log_probs_ptr = static_cast<float *>(results.log_probs.mutable_data());
    auto *actions_ptr = static_cast<long *>(results.actions.mutable_data());
    auto *rewards_ptr = static_cast<float *>(results.rewards.mutable_data());
    auto *task_idxs_no_pt_ptr = static_cast<int8_t *>(results.task_idxs_no_pt.mutable_data());
    auto *task_success_ptr = static_cast<bool *>(results.task_success.mutable_data());
    auto *difficulty_ptr = static_cast<int8_t *>(results.difficulty.mutable_data());
    auto *win_ptr = static_cast<bool *>(results.win.mutable_data());
    auto *aux_info_ptr = static_cast<int8_t *>(results.aux_info.mutable_data());

    for (int rollout_idx = 0; rollout_idx < num_rollouts; ++rollout_idx)
    {
        const auto &rollout = rollouts[rollout_idx];
        assert(rollout.hist_player_idx_pt.size() == seq_length);
        assert(rollout.hist_trick_pt.size() == seq_length);
        assert(rollout.hist_action_pt.size() == seq_length * 2);
        assert(rollout.hist_turn_pt.size() == seq_length);
        assert(rollout.hist_phase_pt.size() == seq_length);
        assert(rollout.hand_pt.size() == seq_length * max_hand_size * 2);
        assert(rollout.player_idx_pt.size() == seq_length);
        assert(rollout.trick_pt.size() == seq_length);
        assert(rollout.turn_pt.size() == seq_length);
        assert(rollout.phase_pt.size() == seq_length);
        assert(rollout.task_idxs_pt.size() == seq_length * max_num_tasks * 2);
        assert(rollout.valid_actions_pt.size() == seq_length * max_num_actions * 2);
        assert(rollout.log_probs_pt.size() == seq_length * max_num_actions);
        assert(rollout.actions_pt.size() == seq_length);
        assert(rollout.rewards_pt.size() == seq_length);

        std::copy(rollout.hist_player_idx_pt.begin(), rollout.hist_player_idx_pt.end(), hist_player_idx_ptr);
        hist_player_idx_ptr += rollout.hist_player_idx_pt.size();

        std::copy(rollout.hist_trick_pt.begin(), rollout.hist_trick_pt.end(), hist_trick_ptr);
        hist_trick_ptr += rollout.hist_trick_pt.size();

        std::copy(rollout.hist_action_pt.begin(), rollout.hist_action_pt.end(), hist_action_ptr);
        hist_action_ptr += rollout.hist_action_pt.size();

        std::copy(rollout.hist_turn_pt.begin(), rollout.hist_turn_pt.end(), hist_turn_ptr);
        hist_turn_ptr += rollout.hist_turn_pt.size();

        std::copy(rollout.hist_phase_pt.begin(), rollout.hist_phase_pt.end(), hist_phase_ptr);
        hist_phase_ptr += rollout.hist_phase_pt.size();

        std::copy(rollout.hand_pt.begin(), rollout.hand_pt.end(), hand_ptr);
        hand_ptr += rollout.hand_pt.size();

        std::copy(rollout.player_idx_pt.begin(), rollout.player_idx_pt.end(), player_idx_ptr);
        player_idx_ptr += rollout.player_idx_pt.size();

        std::copy(rollout.trick_pt.begin(), rollout.trick_pt.end(), trick_ptr);
        trick_ptr += rollout.trick_pt.size();

        std::copy(rollout.turn_pt.begin(), rollout.turn_pt.end(), turn_ptr);
        turn_ptr += rollout.turn_pt.size();

        std::copy(rollout.phase_pt.begin(), rollout.phase_pt.end(), phase_ptr);
        phase_ptr += rollout.phase_pt.size();

        std::copy(rollout.task_idxs_pt.begin(), rollout.task_idxs_pt.end(), task_idxs_ptr);
        task_idxs_ptr += rollout.task_idxs_pt.size();

        std::copy(rollout.valid_actions_pt.begin(), rollout.valid_actions_pt.end(), valid_actions_ptr);
        valid_actions_ptr += rollout.valid_actions_pt.size();

        std::copy(rollout.log_probs_pt.begin(), rollout.log_probs_pt.end(), log_probs_ptr);
        log_probs_ptr += rollout.log_probs_pt.size();

        std::copy(rollout.actions_pt.begin(), rollout.actions_pt.end(), actions_ptr);
        actions_ptr += rollout.actions_pt.size();

        std::copy(rollout.rewards_pt.begin(), rollout.rewards_pt.end(), rewards_ptr);
        rewards_ptr += rollout.rewards_pt.size();

        // Calculate success metrics
        bool win = rollout.engine->state.status == Status::kSuccess;
        *win_ptr = win;
        win_ptr++;

        std::vector<int8_t> task_idxs_no_pt(max_num_tasks, -1);
        std::vector<bool> task_success(max_num_tasks, false);
        for (auto &player_tasks : rollout.engine->state.assigned_tasks)
        {
            for (auto &task : player_tasks)
            {
                auto it = std::find(rollout.engine->state.task_idxs.begin(),
                                    rollout.engine->state.task_idxs.end(),
                                    task.task_idx);
                assert(it != rollout.engine->state.task_idxs.end());
                int task_idx_pos = it - rollout.engine->state.task_idxs.begin();
                task_idxs_no_pt[task_idx_pos] = task.task_idx;
                task_success[task_idx_pos] = task.status == Status::kSuccess;
            }
        }

        std::copy(task_idxs_no_pt.begin(), task_idxs_no_pt.end(), task_idxs_no_pt_ptr);
        task_idxs_no_pt_ptr += task_idxs_no_pt.size();

        std::copy(task_success.begin(), task_success.end(), task_success_ptr);
        task_success_ptr += task_success.size();

        *difficulty_ptr = rollout.engine->state.difficulty;
        difficulty_ptr++;

        std::copy(rollout.aux_info.begin(), rollout.aux_info.end(), aux_info_ptr);
        aux_info_ptr += rollout.aux_info.size();
    }
    return results;
}