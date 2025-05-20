#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <cassert>
#include <stdexcept>

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
    kNoSignal,
    kDraft,
    kNoDraft
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
    if (type == "draft")
        return ActionType::kDraft;
    if (type == "nodraft")
        return ActionType::kNoDraft;
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
    case ActionType::kDraft:
        return "draft";
    case ActionType::kNoDraft:
        return "nodraft";
    default:
        throw std::runtime_error("Invalid action type");
    }
}

// Game phases
enum class Phase
{
    kPlay,
    kSignal,
    kDraft,
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
    case Phase::kDraft:
        return "draft";
    case Phase::kEnd:
        return "end";
    default:
        throw std::runtime_error("Invalid phase type");
    }
}

// Action class representing a player action
struct Action
{
    Action(int player_, ActionType type_, std::optional<Card> card_ = std::nullopt, std::optional<int> task_idx_ = std::nullopt)
        : player(player_), type(type_), card(card_), task_idx(task_idx_) {}

    int player;
    ActionType type;
    std::optional<Card> card;
    std::optional<int> task_idx;
    std::string to_string() const
    {
        if (type == ActionType::kDraft)
        {
            return "P" + std::to_string(player) + " drafts " +
                   std::to_string(task_idx.value()) + ".";
        }
        else if (type == ActionType::kNoDraft)
        {
            return "P" + std::to_string(player) + " does not draft.";
        }
        else if (type == ActionType::kNoSignal)
        {
            return "P" + std::to_string(player) + " does not signal.";
        }
        return "P" + std::to_string(player) + " " +
               action_type_to_string(type) + "s " +
               card->to_string() + ".";
    }

    // Equality operators
    bool operator==(const Action &other) const
    {
        return player == other.player && type == other.type && card == other.card && task_idx == other.task_idx;
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
    kSingleton,
    kOther
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
    if (value == "other")
        return SignalValue::kOther;
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
    case SignalValue::kOther:
        return "other";
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
