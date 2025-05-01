#include "rollout.h"
#include <random>
#include <algorithm>

Rollout::Rollout(const Settings &settings, int engine_seed)
    : settings(settings), engine(std::make_unique<Engine>(settings, engine_seed))
{
    public_history_pt.push_back({});

    for (int player = 0; player < settings.num_players; ++player)
    {
        int player_idx = engine->state.get_player_idx(player);
        for (const auto &task : engine->state.assigned_tasks[player])
        {
            task_idxs.push_back({task.task_idx, player_idx});
        }
    }
}

std::tuple<int, int> Rollout::card_to_tuple(const std::optional<Card> &card) const
{
    if (!card.has_value())
    {
        return {settings.max_suit_length(), settings.num_suits()};
    }
    return {card->rank - 1, settings.get_suit_idx(card->suit)};
}

std::vector<std::tuple<int, int>> Rollout::encode_hand(const std::vector<Card> &hand) const
{
    std::vector<std::tuple<int, int>> encoded;
    for (const auto &card : hand)
    {
        encoded.push_back(card_to_tuple(card));
    }
    return encoded;
}

MoveInput Rollout::get_move_input()
{
    // First, collect all valid actions and state info
    if (engine->state.phase != Phase::kEnd)
    {
        int player_idx = engine->state.get_player_idx();
        int turn = engine->state.get_turn();

        std::vector<std::vector<std::tuple<int, int>>> encoded_hands;
        for (int pidx = 0; pidx < settings.num_players; ++pidx)
        {
            encoded_hands.push_back(encode_hand(
                engine->state.hands[engine->state.get_player(pidx)]));
        }

        // Record private inputs
        PrivateInput private_input{
            .hand = encoded_hands[player_idx],
            .hands = encoded_hands,
            .trick = engine->state.trick,
            .player_idx = player_idx,
            .turn = turn,
            .phase = engine->state.phase_idx(),
        };
        private_inputs_pt.push_back(private_input);

        // Record valid actions
        valid_actions = engine->valid_actions();
        std::vector<std::tuple<int, int>> encoded_actions;
        for (const auto &action : valid_actions)
        {
            encoded_actions.push_back(card_to_tuple(action.card));
        }
        valid_actions_pt.push_back(encoded_actions);

        // Record task indices
        task_idxs_pt.push_back(task_idxs);
    }

    MoveInput move_input{
        .public_history = public_history_pt.back(),
        .private_inputs = private_inputs_pt.back(),
        .valid_actions = valid_actions_pt.back(),
        .task_idxs = task_idxs_pt.back(),
    };

    return move_input;
}

void Rollout::move(int action_idx, const std::vector<double> &probs, const std::vector<double> &log_probs)
{
    if (engine->state.phase == Phase::kEnd)
    {
        return;
    }

    assert(action_idx < (int)valid_actions.size());
    auto action = valid_actions[action_idx];

    // Record the action and its probability
    actions_pt.push_back(action_idx);
    probs_pt.push_back(probs);
    log_probs_pt.push_back(log_probs);

    // Record public history
    PublicHistory public_history{
        .trick = engine->state.trick,
        .card = card_to_tuple(action.card),
        .player_idx = engine->state.get_player_idx(),
        .turn = engine->state.get_turn(),
        .phase = engine->state.phase_idx(),
    };
    public_history_pt.push_back(public_history);

    // Execute the action and record reward
    double reward = engine->move(action);
    rewards_pt.push_back(reward);
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

std::vector<MoveInput> BatchRollout::get_move_inputs()
{
    std::vector<MoveInput> ret;
    for (auto &rollout : rollouts)
    {
        ret.push_back(rollout.get_move_input());
    }
    return ret;
}

void BatchRollout::move(const std::vector<int> &action_indices, const std::vector<std::vector<double>> &probs, const std::vector<std::vector<double>> &log_probs)
{
    assert((int)action_indices.size() == num_rollouts);
    for (int rollout_idx = 0; rollout_idx < num_rollouts; ++rollout_idx)
    {
        rollouts[rollout_idx].move(action_indices[rollout_idx], probs[rollout_idx], log_probs[rollout_idx]);
    }
}

bool BatchRollout::is_done() const
{
    return std::all_of(rollouts.begin(), rollouts.end(),
                       [](const auto &rollout)
                       { return rollout.engine->state.phase == Phase::kEnd; });
}

std::vector<RolloutResult> BatchRollout::get_results() const
{
    std::vector<RolloutResult> ret;
    for (int rollout_idx = 0; rollout_idx < num_rollouts; ++rollout_idx)
    {
        const auto &rollout = rollouts[rollout_idx];
        auto public_history_pt = rollout.public_history_pt;
        auto &valid_actions_pt = rollout.valid_actions_pt;
        auto &private_inputs_pt = rollout.private_inputs_pt;
        auto &actions_pt = rollout.actions_pt;
        auto &task_idxs_pt = rollout.task_idxs_pt;
        auto &rewards_pt = rollout.rewards_pt;
        auto &probs_pt = rollout.probs_pt;
        auto &log_probs_pt = rollout.log_probs_pt;

        // Remove the last dummy entry
        public_history_pt.pop_back();

        // Verify all lengths match
        assert(valid_actions_pt.size() == public_history_pt.size() &&
               private_inputs_pt.size() == public_history_pt.size() &&
               actions_pt.size() == public_history_pt.size() &&
               probs_pt.size() == public_history_pt.size() &&
               log_probs_pt.size() == public_history_pt.size() &&
               task_idxs_pt.size() == public_history_pt.size() &&
               rewards_pt.size() == public_history_pt.size());

        // Calculate success metrics
        bool win = rollout.engine->state.status == Status::kSuccess;
        std::vector<std::tuple<int, int>> num_success_tasks_pp;
        for (int player_idx = 0; player_idx < settings.num_players; ++player_idx)
        {
            int player = rollout.engine->state.get_player(player_idx);
            int num_success = std::count_if(
                rollout.engine->state.assigned_tasks[player].begin(),
                rollout.engine->state.assigned_tasks[player].end(),
                [](const auto &task)
                { return task.status == Status::kSuccess; });
            int num_tasks = rollout.engine->state.assigned_tasks[player].size();
            num_success_tasks_pp.push_back({num_success, num_tasks});
        }

        RolloutResult rollout_result{
            .public_history = public_history_pt,
            .private_inputs = private_inputs_pt,
            .valid_actions = valid_actions_pt,
            .probs = probs_pt,
            .log_probs = log_probs_pt,
            .actions = actions_pt,
            .rewards = rewards_pt,
            .num_success_tasks_pp = num_success_tasks_pp,
            .task_idxs = task_idxs_pt,
            .win = win,
        };
        ret.push_back(rollout_result);
    }
    return ret;
}