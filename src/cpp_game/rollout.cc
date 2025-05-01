#include "rollout.h"
#include <random>
#include <algorithm>

Rollout::Rollout(const Settings &settings, int engine_seed)
    : settings(settings), engine(std::make_unique<Engine>(settings, engine_seed))
{
    // Initialize empty history tensors
    hist_player_idxs_pt.push_back(torch::full({1}, -1, torch::kInt8));
    hist_tricks_pt.push_back(torch::full({1}, -1, torch::kInt8));
    hist_cards_pt.push_back(torch::full({1, 2}, -1, torch::kInt8));
    hist_turns_pt.push_back(torch::full({1}, -1, torch::kInt8));
    hist_phases_pt.push_back(torch::full({1}, -1, torch::kInt8));

    // Initialize task indices tensor
    std::vector<int> task_data_flat;
    for (int player = 0; player < settings.num_players; ++player)
    {
        int player_idx = engine->state.get_player_idx(player);
        for (const auto &task : engine->state.assigned_tasks[player])
        {
            task_data_flat.push_back(task.task_idx);
            task_data_flat.push_back(player_idx);
        }
    }
    auto task_tensor = torch::tensor(task_data_flat, torch::kInt8);
    task_idxs_pt.push_back(task_tensor.reshape({-1, 2}));
}

torch::Tensor Rollout::card_to_tensor(const std::optional<Card> &card) const
{
    if (!card.has_value())
    {
        return torch::tensor({settings.max_suit_length(), settings.num_suits()}, torch::kInt8);
    }
    return torch::tensor({card->rank - 1, settings.get_suit_idx(card->suit)}, torch::kInt8);
}

torch::Tensor Rollout::encode_hand(const std::vector<Card> &hand) const
{
    std::vector<torch::Tensor> encoded_cards;
    for (const auto &card : hand)
    {
        encoded_cards.push_back(card_to_tensor(card));
    }
    // Pad with (-1, -1) to max_hand_size + 1
    while (encoded_cards.size() < settings.max_hand_size() + 1)
    {
        encoded_cards.push_back(torch::tensor({-1, -1}, torch::kInt8));
    }
    return torch::stack(encoded_cards);
}

void Rollout::move(int action_idx, const torch::Tensor &probs, const torch::Tensor &log_probs)
{
    if (engine->state.phase == Phase::kEnd)
    {
        return;
    }

    assert(action_idx < (int)valid_actions.size());
    auto action = valid_actions[action_idx];

    // Record the action and its probability
    actions_pt.push_back(torch::tensor({action_idx}, torch::kInt8));
    probs_pt.push_back(probs);
    log_probs_pt.push_back(log_probs);

    // Record public history
    hist_player_idxs_pt.push_back(torch::tensor({engine->state.get_player_idx()}, torch::kInt8));
    hist_tricks_pt.push_back(torch::tensor({engine->state.trick}, torch::kInt8));
    hist_cards_pt.push_back(card_to_tensor(action.card));
    hist_turns_pt.push_back(torch::tensor({engine->state.get_turn()}, torch::kInt8));
    hist_phases_pt.push_back(torch::tensor({engine->state.phase_idx()}, torch::kInt8));

    // Execute the action and record reward
    double reward = engine->move(action);
    rewards_pt.push_back(torch::tensor({reward}, torch::kFloat32));
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
    // Prepare vectors to collect tensors from all rollouts
    std::vector<torch::Tensor> hist_player_idxs_batch;
    std::vector<torch::Tensor> hist_tricks_batch;
    std::vector<torch::Tensor> hist_cards_batch;
    std::vector<torch::Tensor> hist_turns_batch;
    std::vector<torch::Tensor> hist_phases_batch;

    std::vector<torch::Tensor> hand_batch;
    std::vector<torch::Tensor> hands_batch;
    std::vector<torch::Tensor> player_idx_batch;
    std::vector<torch::Tensor> trick_batch;
    std::vector<torch::Tensor> turn_batch;
    std::vector<torch::Tensor> phase_batch;

    std::vector<torch::Tensor> valid_actions_batch;
    std::vector<torch::Tensor> task_idxs_batch;

    // Collect tensors from each rollout
    for (auto &rollout : rollouts)
    {
        // Update the rollout's tensors if game is not over
        if (rollout.engine->state.phase != Phase::kEnd)
        {
            int player_idx = rollout.engine->state.get_player_idx();
            int turn = rollout.engine->state.get_turn();

            // Encode hands for all players
            std::vector<torch::Tensor> encoded_hands;
            for (int pidx = 0; pidx < settings.num_players; ++pidx)
            {
                encoded_hands.push_back(rollout.encode_hand(
                    rollout.engine->state.hands[rollout.engine->state.get_player(pidx)]));
            }
            torch::Tensor hands_tensor = torch::stack(encoded_hands);

            // Record private inputs
            rollout.hand_pt.push_back(encoded_hands[player_idx]);
            rollout.hands_pt.push_back(hands_tensor);
            rollout.player_idx_pt.push_back(torch::tensor({player_idx}, torch::kInt8));
            rollout.trick_pt.push_back(torch::tensor({rollout.engine->state.trick}, torch::kInt8));
            rollout.turn_pt.push_back(torch::tensor({turn}, torch::kInt8));
            rollout.phase_pt.push_back(torch::tensor({rollout.engine->state.phase_idx()}, torch::kInt8));

            // Record valid actions
            rollout.valid_actions = rollout.engine->valid_actions();
            std::vector<torch::Tensor> encoded_actions;
            for (const auto &action : rollout.valid_actions)
            {
                encoded_actions.push_back(rollout.card_to_tensor(action.card));
            }
            rollout.valid_actions_pt.push_back(torch::stack(encoded_actions));
        }

        // Add the latest tensors to our batch
        hist_player_idxs_batch.push_back(rollout.hist_player_idxs_pt.back());
        hist_tricks_batch.push_back(rollout.hist_tricks_pt.back());
        hist_cards_batch.push_back(rollout.hist_cards_pt.back());
        hist_turns_batch.push_back(rollout.hist_turns_pt.back());
        hist_phases_batch.push_back(rollout.hist_phases_pt.back());

        hand_batch.push_back(rollout.hand_pt.back());
        hands_batch.push_back(rollout.hands_pt.back());
        player_idx_batch.push_back(rollout.player_idx_pt.back());
        trick_batch.push_back(rollout.trick_pt.back());
        turn_batch.push_back(rollout.turn_pt.back());
        phase_batch.push_back(rollout.phase_pt.back());

        valid_actions_batch.push_back(rollout.valid_actions_pt.back());
        task_idxs_batch.push_back(rollout.task_idxs_pt.back());
    }

    // Stack all tensors into batch tensors
    return MoveInputs{
        .hist_player_idxs = torch::stack(hist_player_idxs_batch),
        .hist_tricks = torch::stack(hist_tricks_batch),
        .hist_cards = torch::stack(hist_cards_batch),
        .hist_turns = torch::stack(hist_turns_batch),
        .hist_phases = torch::stack(hist_phases_batch),

        .hand = torch::stack(hand_batch),
        .hands = torch::stack(hands_batch),
        .player_idx = torch::stack(player_idx_batch),
        .trick = torch::stack(trick_batch),
        .turn = torch::stack(turn_batch),
        .phase = torch::stack(phase_batch),

        .valid_actions = torch::stack(valid_actions_batch),
        .task_idxs = torch::stack(task_idxs_batch),
    };
}

void BatchRollout::move(const torch::Tensor &action_indices, const torch::Tensor &probs, const torch::Tensor &log_probs)
{
    assert(action_indices.size(0) == num_rollouts);
    for (int rollout_idx = 0; rollout_idx < num_rollouts; ++rollout_idx)
    {
        rollouts[rollout_idx].move(action_indices[rollout_idx].item<int>(), probs[rollout_idx], log_probs[rollout_idx]);
    }
}

bool BatchRollout::is_done() const
{
    return std::all_of(rollouts.begin(), rollouts.end(),
                       [](const auto &rollout)
                       { return rollout.engine->state.phase == Phase::kEnd; });
}

RolloutResults BatchRollout::get_results() const
{
    // Prepare batch tensors for each property
    std::vector<torch::Tensor> hist_player_idxs_batch;
    std::vector<torch::Tensor> hist_tricks_batch;
    std::vector<torch::Tensor> hist_cards_batch;
    std::vector<torch::Tensor> hist_turns_batch;
    std::vector<torch::Tensor> hist_phases_batch;

    std::vector<torch::Tensor> hand_batch;
    std::vector<torch::Tensor> hands_batch;
    std::vector<torch::Tensor> player_idx_batch;
    std::vector<torch::Tensor> trick_batch;
    std::vector<torch::Tensor> turn_batch;
    std::vector<torch::Tensor> phase_batch;

    std::vector<torch::Tensor> valid_actions_batch;
    std::vector<torch::Tensor> task_idxs_batch;

    std::vector<torch::Tensor> probs_batch;
    std::vector<torch::Tensor> log_probs_batch;
    std::vector<torch::Tensor> actions_batch;
    std::vector<torch::Tensor> rewards_batch;

    std::vector<torch::Tensor> num_success_tasks_pp_batch;
    std::vector<int> win_batch;

    for (int rollout_idx = 0; rollout_idx < num_rollouts; ++rollout_idx)
    {
        const auto &rollout = rollouts[rollout_idx];

        // Pad sequences as necessary for history tensors
        hist_player_idxs_batch.push_back(torch::stack(rollout.hist_player_idxs_pt));
        hist_tricks_batch.push_back(torch::stack(rollout.hist_tricks_pt));
        hist_cards_batch.push_back(torch::stack(rollout.hist_cards_pt));
        hist_turns_batch.push_back(torch::stack(rollout.hist_turns_pt));
        hist_phases_batch.push_back(torch::stack(rollout.hist_phases_pt));

        hand_batch.push_back(torch::stack(rollout.hand_pt));
        hands_batch.push_back(torch::stack(rollout.hands_pt));
        player_idx_batch.push_back(torch::stack(rollout.player_idx_pt));
        trick_batch.push_back(torch::stack(rollout.trick_pt));
        turn_batch.push_back(torch::stack(rollout.turn_pt));
        phase_batch.push_back(torch::stack(rollout.phase_pt));

        valid_actions_batch.push_back(torch::stack(rollout.valid_actions_pt));
        task_idxs_batch.push_back(torch::stack(rollout.task_idxs_pt));

        probs_batch.push_back(torch::stack(rollout.probs_pt));
        log_probs_batch.push_back(torch::stack(rollout.log_probs_pt));
        actions_batch.push_back(torch::stack(rollout.actions_pt));
        rewards_batch.push_back(torch::stack(rollout.rewards_pt));

        // Calculate success metrics
        bool win = rollout.engine->state.status == Status::kSuccess;
        win_batch.push_back(win);

        std::vector<torch::Tensor> num_success_tasks;
        for (int player_idx = 0; player_idx < settings.num_players; ++player_idx)
        {
            int player = rollout.engine->state.get_player(player_idx);
            int num_success = std::count_if(
                rollout.engine->state.assigned_tasks[player].begin(),
                rollout.engine->state.assigned_tasks[player].end(),
                [](const auto &task)
                { return task.status == Status::kSuccess; });
            int num_tasks = rollout.engine->state.assigned_tasks[player].size();
            num_success_tasks.push_back(torch::tensor({num_success, num_tasks}, torch::kInt8));
        }
        num_success_tasks_pp_batch.push_back(torch::stack(num_success_tasks));
    }

    // Stack all tensors into batch tensors
    return RolloutResults{
        .hist_player_idxs = torch::stack(hist_player_idxs_batch),
        .hist_tricks = torch::stack(hist_tricks_batch),
        .hist_cards = torch::stack(hist_cards_batch),
        .hist_turns = torch::stack(hist_turns_batch),
        .hist_phases = torch::stack(hist_phases_batch),

        .hand = torch::stack(hand_batch),
        .hands = torch::stack(hands_batch),
        .player_idx = torch::stack(player_idx_batch),
        .trick = torch::stack(trick_batch),
        .turn = torch::stack(turn_batch),
        .phase = torch::stack(phase_batch),

        .valid_actions = torch::stack(valid_actions_batch),
        .task_idxs = torch::stack(task_idxs_batch),

        .probs = torch::stack(probs_batch),
        .log_probs = torch::stack(log_probs_batch),
        .actions = torch::stack(actions_batch),
        .rewards = torch::stack(rewards_batch),

        .num_success_tasks_pp = torch::stack(num_success_tasks_pp_batch),
        .win = torch::tensor(win_batch, torch::kInt8)};
}