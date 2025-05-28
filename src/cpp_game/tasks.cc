#include "tasks.h"
#include "state.h"

// Implementation of TrickCond methods
void TrickCond::on_trick_end(const State &state)
{
    if (status != Status::kUnresolved)
        return;

    if (state.trick == trick)
    {
        bool won_trick = state.trick_winner == player;
        status = (won_trick == !no) ? Status::kSuccess : Status::kFail;
    }
}

// Implementation of CardCond methods
void CardCond::on_trick_end(const State &state)
{
    if (status != Status::kUnresolved)
        return;

    if (std::any_of(state.active_cards.begin(), state.active_cards.end(),
                    [this](const auto &pair)
                    { return card == pair.first; }))
    {
        status = state.trick_winner == player ? Status::kSuccess : Status::kFail;
    }
}

// Implementation of CumTrickCond methods
void CumTrickCond::on_trick_end(const State &state)
{
    assert(state.trick_winner.has_value());

    int trick_winner = state.trick_winner.value();

    if (trick_winner == player)
    {
        num_tricks_won++;
    }
    else
    {
        if (std::holds_alternative<TrickCountType>(num_tricks))
        {
            auto trick_type = std::get<TrickCountType>(num_tricks);

            if (trick_type == TrickCountType::kCapt)
            {
                assert(state.captain != player);
                if (state.trick_winner == state.captain)
                {
                    num_other_tricks_won++;
                }
            }
            else if (trick_type == TrickCountType::kSumOthers)
            {
                num_other_tricks_won++;
            }
            else if (trick_type == TrickCountType::kAnyOther)
            {
                num_other_tricks_won_per_player[trick_winner]++;
            }
        }
    }

    int num_other_tricks = get_num_other_tricks();
    bool comp_ok = compare_dir(num_tricks_won, num_other_tricks, direction);

    if (std::holds_alternative<int>(num_tricks))
    {
        if ((direction == Direction::kGreaterThanOrEqual || direction == Direction::kGreaterThan) && comp_ok)
        {
            status = Status::kSuccess;
        }
        else if ((direction == Direction::kLessThan || direction == Direction::kLessThanOrEqual) && !comp_ok)
        {
            status = Status::kFail;
        }
        else if (direction == Direction::kEqual && num_tricks_won > num_other_tricks)
        {
            status = Status::kFail;
        }
    }

    if (direction == Direction::kGreaterThanOrEqual || direction == Direction::kGreaterThan)
    {
        partial_value = clamp(1.0 - (num_other_tricks - num_tricks_won) / 3.0);
    }
    else if (direction == Direction::kLessThanOrEqual || direction == Direction::kLessThan)
    {
        partial_value = clamp(1.0 - (num_tricks_won - num_other_tricks) / 3.0);
    }
    else
    {
        partial_value = clamp(1.0 - std::abs(num_tricks_won - num_other_tricks) / 3.0);
    }
}

void CumTrickCond::on_end()
{
    if (status == Status::kUnresolved)
    {
        status = compare_dir(num_tricks_won, get_num_other_tricks(), direction)
                     ? Status::kSuccess
                     : Status::kFail;
    }
}

// Implementation of CumCardCond methods
void CumCardCond::on_trick_end(const State &state)
{
    if (state.trick_winner == player)
    {
        // Count cards in this trick that match our filters
        for (const auto &pair : state.active_cards)
        {
            if (card_filter(pair.first))
            {
                num_cards_won++;
            }

            if (other_card_filter.has_value() && other_card_filter.value()(pair.first))
            {
                num_other_cards_won++;
            }
        }
    }

    int num_other_cards = get_num_other_cards();
    bool comp_ok = compare_dir(num_cards_won, num_other_cards, direction);

    if (num_cards.has_value())
    {
        if ((direction == Direction::kGreaterThanOrEqual || direction == Direction::kGreaterThan) && comp_ok)
        {
            status = Status::kSuccess;
        }
        else if ((direction == Direction::kLessThan || direction == Direction::kLessThanOrEqual) && !comp_ok)
        {
            status = Status::kFail;
        }
        else if (direction == Direction::kEqual && num_cards_won > num_other_cards)
        {
            status = Status::kFail;
        }
    }

    if (direction == Direction::kGreaterThanOrEqual || direction == Direction::kGreaterThan)
    {
        partial_value = clamp(1.0 - (num_other_cards - num_cards_won) / 3.0);
    }
    else if (direction == Direction::kLessThanOrEqual || direction == Direction::kLessThan)
    {
        partial_value = clamp(1.0 - (num_cards_won - num_other_cards) / 3.0);
    }
    else
    {
        partial_value = clamp(1.0 - std::abs(num_cards_won - num_other_cards) / 3.0);
    }
}

void CumCardCond::on_end()
{
    if (status == Status::kUnresolved)
    {
        status = compare_dir(num_cards_won, get_num_other_cards(), direction)
                     ? Status::kSuccess
                     : Status::kFail;
    }
}

// Implementation of WithCond methods
void WithCond::on_trick_end(const State &state)
{
    if (status != Status::kUnresolved)
        return;

    if (state.trick_winner == player)
    {
        // Find the player's card in this trick
        for (const auto &pair : state.active_cards)
        {
            if (pair.second == player)
            {
                if (card_filter(pair.first))
                {
                    status = Status::kSuccess;
                }
                break;
            }
        }
    }
}

// Implementation of SweepCond methods
void SweepCond::on_trick_end(const State &state)
{
    if (status != Status::kUnresolved)
        return;

    if (state.trick_winner == player)
    {
        // Add all cards from this trick to their respective suit sets
        for (const auto &pair : state.active_cards)
        {
            if (!pair.first.is_trump())
            {
                cards_won_per_suit[pair.first.suit]++;
            }
        }

        int num_sweeps_ = 0;
        for (const auto &count : cards_won_per_suit)
        {
            if (count == settings.side_suit_length)
            {
                num_sweeps_++;
            }
        }

        partial_value = std::min(1.0, static_cast<double>(num_sweeps_) / num_sweeps);

        if (num_sweeps_ >= num_sweeps)
        {
            status = Status::kSuccess;
        }
    }
}

// Implementation of ConsecCond methods
void ConsecCond::on_trick_end(const State &state)
{
    if (status != Status::kUnresolved)
        return;

    if (state.trick_winner == player)
    {
        if (cur_consec_end.has_value() && state.trick == cur_consec_end.value() + 1)
        {
            cur_consec_end = state.trick;
        }
        else
        {
            cur_consec_start = state.trick;
            cur_consec_end = state.trick;
        }
    }
    else
    {
        cur_consec_start = std::nullopt;
        cur_consec_end = std::nullopt;
    }

    if (no)
    {
        if (cur_consec_start.has_value() && cur_consec_end.has_value() &&
            (cur_consec_end.value() - cur_consec_start.value() + 1) >= num_consec)
        {
            status = Status::kFail;
        }
        else
        {
            partial_value = static_cast<double>(state.trick + 1) / static_cast<double>(settings.num_tricks);
        }
    }
    else
    {
        if (cur_consec_start.has_value() && cur_consec_end.has_value())
        {
            int num_consec_ = cur_consec_end.value() - cur_consec_start.value() + 1;
            partial_value = std::max(partial_value, static_cast<double>(num_consec_) / num_consec);

            if (num_consec_ >= num_consec)
            {
                status = Status::kSuccess;
            }
        }
    }
}

void ConsecCond::on_end()
{
    if (status == Status::kUnresolved)
    {
        status = no ? Status::kSuccess : Status::kFail;
    }
}

// Implementation of SumCond methods
void SumCond::on_trick_end(const State &state)
{
    if (status != Status::kUnresolved)
        return;

    if (state.trick_winner == player)
    {
        int trick_sum = 0;
        for (const auto &pair : state.active_cards)
        {
            trick_sum += pair.first.rank;
        }

        if (compare_dir(trick_sum, sum, direction))
        {
            status = Status::kSuccess;
        }
    }
}

// Implementation of NoLeadCond methods
void NoLeadCond::on_trick_end(const State &state)
{
    // If we're the leader and we lead a forbidden suit, fail
    if (state.leader == player && !state.active_cards.empty())
    {
        int lead_suit = state.active_cards[0].first.suit;
        if (std::find(suits.begin(), suits.end(), lead_suit) != suits.end())
        {
            status = Status::kFail;
            num_bad_lead++;
        }
    }

    partial_value = clamp(1.0 - num_bad_lead / 2.0);
}

// Implementation of parse_token function
std::shared_ptr<Condition> parse_token(const std::string &token, const Settings &settings, int player)
{
    std::string orig_token = token;

    if (token.starts_with("T"))
    {
        std::string t = token.substr(1);
        int trick;
        if (t == "-1")
        {
            trick = settings.num_tricks - 1;
        }
        else
        {
            trick = std::stoi(t);
        }
        return std::make_shared<TrickCond>(settings, player, trick);
    }
    else if (token.starts_with("#T"))
    {
        std::string t = token.substr(2);
        Direction direction;
        std::string remaining;
        std::tie(direction, remaining) = parse_dir(t);

        if (std::isdigit(remaining[0]))
        {
            return std::make_shared<CumTrickCond>(settings, player, direction, std::stoi(remaining));
        }
        else if (remaining.starts_with("#T("))
        {
            std::string type_str = remaining.substr(3, remaining.size() - 4);
            TrickCountType type;
            if (type_str == "capt")
                type = TrickCountType::kCapt;
            else if (type_str == "sumothers")
                type = TrickCountType::kSumOthers;
            else if (type_str == "anyother")
                type = TrickCountType::kAnyOther;
            else
                throw std::runtime_error("Unhandled token: " + orig_token);

            return std::make_shared<CumTrickCond>(settings, player, direction, type);
        }
        else
        {
            throw std::runtime_error("Unhandled token: " + orig_token);
        }
    }
    else if (token.starts_with("#sweep"))
    {
        std::string t = token.substr(8); // Remove "#sweep>="
        return std::make_shared<SweepCond>(settings, player, std::stoi(t));
    }
    else if (token.starts_with("#"))
    {
        std::string t = token.substr(1);
        auto [card_filter, remaining] = parse_card_filter(t);
        if (!card_filter.has_value())
        {
            throw std::runtime_error("Expected card filter in: " + token);
        }

        Direction direction;
        std::string after_dir;
        std::tie(direction, after_dir) = parse_dir(remaining);

        if (after_dir.starts_with("#"))
        {
            std::string other_t = after_dir.substr(1);
            auto [other_card_filter, other_remaining] = parse_card_filter(other_t);
            if (!other_card_filter.has_value() || !other_remaining.empty())
            {
                throw std::runtime_error("Expected other card filter in: " + token);
            }
            return std::make_shared<CumCardCond>(settings, player, direction, card_filter.value(), std::nullopt, other_card_filter.value());
        }
        else
        {
            return std::make_shared<CumCardCond>(settings, player, direction, card_filter.value(), std::stoi(after_dir));
        }
    }
    else if (token.starts_with("no("))
    {
        std::string inner = token.substr(3, token.size() - 4);
        auto cond = parse_token(inner, settings, player);

        if (auto trick_cond = std::dynamic_pointer_cast<TrickCond>(cond))
        {
            trick_cond->no = true;
            return trick_cond;
        }
        else if (auto consec_cond = std::dynamic_pointer_cast<ConsecCond>(cond))
        {
            consec_cond->no = true;
            return consec_cond;
        }
        else
        {
            throw std::runtime_error("no() can only be applied to TrickCond or ConsecCond");
        }
    }
    else if (token.starts_with("consec("))
    {
        std::string inner = token.substr(7, token.size() - 8);
        return std::make_shared<ConsecCond>(settings, player, std::stoi(inner));
    }
    else if (token.starts_with("nolead("))
    {
        std::string inner = token.substr(7, token.size() - 8);
        std::vector<int> suits;
        std::stringstream ss(inner);
        std::string suit_str;

        while (std::getline(ss, suit_str, ','))
        {
            if (TO_SUIT_NUM.find(suit_str[0]) != TO_SUIT_NUM.end())
            {
                suits.push_back(TO_SUIT_NUM.at(suit_str[0]));
            }
        }

        return std::make_shared<NoLeadCond>(settings, player, suits);
    }
    else if (token.starts_with("with("))
    {
        std::string inner = token.substr(5, token.size() - 6);
        auto [card_filter, remaining] = parse_card_filter(inner);
        if (!card_filter.has_value() || !remaining.empty())
        {
            throw std::runtime_error("Expected card filter in: " + token);
        }
        return std::make_shared<WithCond>(settings, player, card_filter.value());
    }
    else if (token.starts_with("sum"))
    {
        std::string t = token.substr(3);
        Direction direction;
        std::string remaining;
        std::tie(direction, remaining) = parse_dir(t);
        return std::make_shared<SumCond>(settings, player, direction, std::stoi(remaining));
    }
    else
    {
        auto [card, remaining] = parse_card(token);
        if (card.has_value() && remaining.empty())
        {
            return std::make_shared<CardCond>(settings, player, card.value());
        }
    }

    throw std::runtime_error("Unhandled token: " + orig_token);
}

// Implementation of AssignedTask methods
void AssignedTask::on_trick_end(const State &state)
{
    if (in_one_trick && status != Status::kUnresolved)
    {
        return;
    }

    for (auto &cond : conds)
    {
        cond->on_trick_end(state);
        if (in_one_trick)
        {
            cond->on_end();
        }
    }

    bool all_success = true;
    for (const auto &cond : conds)
    {
        if (cond->status != Status::kSuccess)
        {
            all_success = false;
            break;
        }
    }

    if (all_success && (!in_one_trick || state.trick_winner == player))
    {
        assert(status == Status::kUnresolved || status == Status::kSuccess);
        status = Status::kSuccess;
    }

    if (in_one_trick && status != Status::kSuccess)
    {
        for (auto &cond : conds)
        {
            cond->reset();
            assert(cond->status == Status::kUnresolved);
        }
    }

    if (!in_one_trick)
    {
        bool any_fail = false;
        for (const auto &cond : conds)
        {
            if (cond->status == Status::kFail)
            {
                any_fail = true;
                break;
            }
        }

        if (any_fail)
        {
            assert(status == Status::kUnresolved || status == Status::kFail);
            status = Status::kFail;
        }
    }

    compute_value();
}

void AssignedTask::on_game_end()
{
    if (in_one_trick)
    {
        assert(status != Status::kFail);
        if (status == Status::kUnresolved)
        {
            status = Status::kFail;
        }
        compute_value();
        return;
    }

    for (auto &cond : conds)
    {
        cond->on_end();
    }

    bool all_resolved = true;
    for (const auto &cond : conds)
    {
        if (cond->status == Status::kUnresolved)
        {
            all_resolved = false;
            break;
        }
    }
    assert(all_resolved);

    bool all_success = true;
    for (const auto &cond : conds)
    {
        if (cond->status != Status::kSuccess)
        {
            all_success = false;
            break;
        }
    }

    if (all_success)
    {
        assert(status == Status::kUnresolved || status == Status::kSuccess);
        status = Status::kSuccess;
    }
    else
    {
        assert(status == Status::kUnresolved || status == Status::kFail);
        status = Status::kFail;
    }

    compute_value();
}
