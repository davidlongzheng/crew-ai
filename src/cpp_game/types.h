#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <cassert>

// Constants
constexpr int TRUMP_SUIT_NUM = 4;

// Suit mapping
const std::map<int, char> TO_SUIT_LETTER = {
    {0, 'b'}, {1, 'g'}, {2, 'p'}, {3, 'y'}, {4, 't'}};

const std::map<char, int> TO_SUIT_NUM = {
    {'b', 0}, {'g', 1}, {'p', 2}, {'y', 3}, {'t', 4}};

// Card class representing a playing card
struct Card
{
    int rank;
    int suit;

    Card(int rank_, int suit_) : rank(rank_), suit(suit_)
    {
        assert(0 <= suit && suit <= 4);
        assert(1 <= rank && rank <= 9);
    }

    bool is_trump() const { return suit == TRUMP_SUIT_NUM; }

    std::string to_string() const
    {
        return std::to_string(rank) + TO_SUIT_LETTER.at(suit);
    }

    // Equality operators
    bool operator==(const Card &other) const
    {
        return rank == other.rank && suit == other.suit;
    }

    bool operator!=(const Card &other) const
    {
        return !(*this == other);
    }
};

// Action types
enum class ActionType
{
    kPlay,
    kSignal,
    kNoSignal
};

// Convert string to ActionType
inline ActionType string_to_action_type(std::string_view type)
{
    if (type == "play")
        return ActionType::kPlay;
    if (type == "signal")
        return ActionType::kSignal;
    if (type == "nosignal")
        return ActionType::kNoSignal;
    throw std::runtime_error("Invalid action type: " + std::string(type));
}

// Convert ActionType to string
inline std::string action_type_to_string(ActionType type)
{
    switch (type)
    {
    case ActionType::kPlay:
        return "play";
    case ActionType::kSignal:
        return "signal";
    case ActionType::kNoSignal:
        return "nosignal";
    default:
        throw std::runtime_error("Invalid action type");
    }
}

// Game phases
enum class Phase
{
    kPlay,
    kSignal,
    kEnd
};

// Convert ActionType to string
inline std::string phase_to_string(Phase type)
{
    switch (type)
    {
    case Phase::kPlay:
        return "play";
    case Phase::kSignal:
        return "signal";
    case Phase::kEnd:
        return "end";
    default:
        throw std::runtime_error("Invalid phase type");
    }
}

// Action class representing a player action
struct Action
{
    int player;
    ActionType type;
    std::optional<Card> card;

    std::string to_string() const
    {
        if (type == ActionType::kNoSignal)
        {
            return "P" + std::to_string(player) + " does not signal.";
        }
        return "P" + std::to_string(player) + " " +
               action_type_to_string(type) + "s " +
               (card ? card->to_string() : "None") + ".";
    }

    // Equality operators
    bool operator==(const Action &other) const
    {
        return player == other.player && type == other.type && card == other.card;
    }

    bool operator!=(const Action &other) const
    {
        return !(*this == other);
    }
};

// Signal values
enum class SignalValue
{
    kHighest,
    kLowest,
    kSingleton
};

// Convert string to SignalValue
inline SignalValue string_to_signal_value(std::string_view value)
{
    if (value == "highest")
        return SignalValue::kHighest;
    if (value == "lowest")
        return SignalValue::kLowest;
    if (value == "singleton")
        return SignalValue::kSingleton;
    throw std::runtime_error("Invalid signal value: " + std::string(value));
}

// Convert SignalValue to string
inline std::string signal_value_to_string(SignalValue value)
{
    switch (value)
    {
    case SignalValue::kHighest:
        return "highest";
    case SignalValue::kLowest:
        return "lowest";
    case SignalValue::kSingleton:
        return "singleton";
    default:
        throw std::runtime_error("Invalid signal value");
    }
}

// Signal class representing a player signal
struct Signal
{
    Card card;
    SignalValue value;
    int trick;

    std::string to_string() const
    {
        return card.to_string() + " " + signal_value_to_string(value) + " " +
               std::to_string(trick);
    }

    // Equality operators
    bool operator==(const Signal &other) const
    {
        return card == other.card && value == other.value && trick == other.trick;
    }

    bool operator!=(const Signal &other) const
    {
        return !(*this == other);
    }
};
