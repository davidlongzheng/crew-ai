#pragma once

#include <optional>
#include <random>
#include <string>
#include <vector>

#include "settings.h"
#include "state.h"
#include "tasks.h"
#include "types.h"
#include "rng.h"

// Crew game engine for one playthrough of the game.
//
// Example:
// Settings settings;
// Engine engine(settings);
// auto valid_actions = engine.valid_actions();  // Get valid actions
// engine.move(valid_actions[0]);
struct Engine
{
    // Constructs an engine with the given settings and optional seed.
    explicit Engine(const Settings &settings, std::optional<int> seed = std::nullopt);

    // Generates hands for all players.
    std::vector<std::vector<Card>> gen_hands(Rng &rng) const;

    // Generates tasks for all players.
    std::vector<std::vector<AssignedTask>> gen_tasks(
        int leader, Rng &rng) const;

    // Resets the game state with an optional seed.
    void reset_state(std::optional<int> seed = std::nullopt);

    // Calculates the winner of a trick based on the active cards.
    int calc_trick_winner(
        const std::vector<std::pair<Card, int>> &active_cards) const;

    // Skips to the next player who hasn't signaled yet.
    void skip_to_next_unsignaled();

    // Makes a move with the given action and returns the reward.
    double move(const Action &action);

    // Returns the valid actions for the current state.
    std::vector<Action> valid_actions() const;

    Settings settings;
    State state;
};