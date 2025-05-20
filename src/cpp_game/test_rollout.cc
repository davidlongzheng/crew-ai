#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <chrono>
#include "rollout.h"
#include "settings.h"
#include <pybind11/embed.h>

// Helper function to print tensor shape and contents
template <typename T>
void print_tensor_info(const std::string &name, const py::array_t<T> &tensor)
{
    std::cout << name << " shape: (";
    for (size_t i = 0; i < tensor.ndim(); ++i)
    {
        std::cout << tensor.shape(i) << (i < tensor.ndim() - 1 ? ", " : "");
    }
    std::cout << ")" << std::endl;
}

// Helper function to create random probabilities for actions
py::array_t<float> create_random_probs(size_t size)
{
    py::array_t<float> probs = py::array_t<float>(size);
    auto r = probs.mutable_unchecked<1>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    float sum = 0.0;
    for (size_t i = 0; i < size; ++i)
    {
        r(i) = dist(gen);
        sum += r(i);
    }

    // Normalize
    for (size_t i = 0; i < size; ++i)
    {
        r(i) /= sum;
    }

    return probs;
}

// Helper function to convert probs to log_probs
py::array_t<float> probs_to_log_probs(const py::array_t<float> &probs)
{
    py::array_t<float> log_probs(probs.size());
    auto probs_r = probs.unchecked<1>();
    auto log_probs_r = log_probs.mutable_unchecked<1>();

    for (size_t i = 0; i < probs.size(); ++i)
    {
        log_probs_r(i) = std::log(probs_r(i));
    }

    return log_probs;
}

void test_single_rollout()
{
    std::cout << "=== Testing Single Rollout ===" << std::endl;

    // Create settings
    Settings settings(
        4,            /* num_players */
        4,            /* num_side_suits */
        true,         /* use_trump_suit */
        9,            /* side_suit_length */
        4,            /* trump_suit_length */
        true,         /* use_signals */
        true,         /* single_signal */
        false,        /* cheating_signal */
        "easy",       /* bank */
        "shuffle",    /* task_distro */
        {0, 0, 1, 2}, /* task_idxs */
        std::nullopt, /* min_difficulty */
        std::nullopt, /* max_difficulty */
        std::nullopt, /* max_num_tasks */
        false,        /* use_drafting */
        2,            /* num_draft_tricks */
        5.0,          /* task_bonus */
        1.0           /* win_bonus */
    );

    // Create a rollout
    int engine_seed = 42;
    std::cout << "Creating rollout with engine_seed=" << engine_seed << std::endl;
    Rollout rollout(settings);
    rollout.reset_state(engine_seed);

    // Run the game until completion
    std::cout << "Running game until completion..." << std::endl;
    int move_count = 0;
    while (rollout.engine->state.phase != Phase::kEnd)
    {
        // Record inputs
        rollout.record_move_inputs();
        assert(!rollout.valid_actions.empty());

        // Choose first valid action (for simplicity)
        int action_idx = 0;

        // Make the move
        rollout.move(action_idx);

        move_count++;
    }

    std::cout << "Game completed in " << move_count << " moves." << std::endl;
}

void test_batch_rollout()
{
    std::cout << "\n=== Testing Batch Rollout ===" << std::endl;

    // Create settings
    Settings settings(
        4,            /* num_players */
        4,            /* num_side_suits */
        true,         /* use_trump_suit */
        9,            /* side_suit_length */
        4,            /* trump_suit_length */
        true,         /* use_signals */
        true,         /* single_signal */
        false,        /* cheating_signal */
        "easy",       /* bank */
        "shuffle",    /* task_distro */
        {0, 0, 1, 2}, /* task_idxs */
        std::nullopt, /* min_difficulty */
        std::nullopt, /* max_difficulty */
        std::nullopt, /* max_num_tasks */
        false,        /* use_drafting */
        2,            /* num_draft_tricks */
        5.0,          /* task_bonus */
        1.0           /* win_bonus */
    );

    // Create a batch rollout
    int num_rollouts = 1000;
    std::vector<int> engine_seeds;
    for (int i = 0; i < num_rollouts; ++i)
    {
        engine_seeds.push_back(i);
    }
    std::cout << "Creating batch rollout with " << num_rollouts << " rollouts" << std::endl;
    BatchRollout batch_rollout(settings, num_rollouts);
    batch_rollout.reset_state(engine_seeds);

    // Run until all games are done
    int batch_move_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    bool is_first = true;

    while (!batch_rollout.is_done())
    {
        // Get move inputs
        MoveInputs move_inputs = batch_rollout.get_move_inputs();
        if (is_first)
        {
            print_tensor_info("  move_inputs.hand", move_inputs.hand);
            print_tensor_info("  move_inputs.trick", move_inputs.trick);
            is_first = false;
        }

        // Create action indices (just choose first valid action for each rollout)
        py::array_t<int8_t> action_indices(num_rollouts);
        auto r_actions = action_indices.mutable_unchecked<1>();
        for (int i = 0; i < num_rollouts; ++i)
        {
            r_actions(i) = 0; // Choose first valid action for simplicity
        }

        int max_num_actions = batch_rollout.max_num_actions;

        // Create random probabilities and log probs
        py::array_t<float> probs(py::array::ShapeContainer({num_rollouts, (long)max_num_actions}));
        py::array_t<float> log_probs(py::array::ShapeContainer({num_rollouts, (long)max_num_actions}));

        auto r_probs = probs.mutable_unchecked<2>();
        auto r_log_probs = log_probs.mutable_unchecked<2>();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0, 1.0);

        for (int i = 0; i < num_rollouts; ++i)
        {
            float sum = 0.0;
            for (int j = 0; j < max_num_actions; ++j)
            {
                float val = dist(gen);
                r_probs(i, j) = val;
                sum += val;
            }

            // Normalize and compute log probs
            for (int j = 0; j < max_num_actions; ++j)
            {
                r_probs(i, j) /= sum;
                r_log_probs(i, j) = std::log(r_probs(i, j));
            }
        }

        // Make the moves
        batch_rollout.move(action_indices, log_probs);
        batch_move_count++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Batch completed in " << batch_move_count << " moves." << std::endl;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    // Get and print results
    RolloutResults results = batch_rollout.get_results();
    std::cout << "Results:" << std::endl;
    print_tensor_info("  hist_player_idx", results.hist_player_idx);
    print_tensor_info("  hist_trick", results.hist_trick);
    print_tensor_info("  hist_action", results.hist_action);
    print_tensor_info("  hist_turn", results.hist_turn);
    print_tensor_info("  hist_phase", results.hist_phase);
    print_tensor_info("  rewards", results.rewards);
    print_tensor_info("  actions", results.actions);
    print_tensor_info("  win", results.win);
}

int main()
{
    try
    {
        py::scoped_interpreter guard{}; // start the interpreter

        std::cout << "Starting Rollout/BatchRollout tests..." << std::endl;

        test_single_rollout();
        test_batch_rollout();

        std::cout << "\nAll tests completed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
