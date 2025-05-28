#pragma once

#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <variant>
#include <cassert>
#include <memory>
#include <tuple>
#include <map>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <numeric>

#include "types.h"
#include "settings.h"

// Forward declarations
struct State;
struct Condition;
struct Task;
struct AssignedTask;
using TaskPtr = std::shared_ptr<Task>;

// Enums to replace Python literals
enum class Status
{
    kSuccess,
    kFail,
    kUnresolved
};

enum class Direction
{
    kGreaterThan,
    kLessThan,
    kGreaterThanOrEqual,
    kEqual,
    kLessThanOrEqual
};

enum class TrickCountType
{
    kCapt,
    kSumOthers,
    kAnyOther
};

// Helper functions
bool compare_dir(int a, int b, Direction dir);

// Card filter type
using CardFilter = std::function<bool(const Card &)>;

// Helper function to clamp a value between 0 and 1
inline double clamp(double x)
{
    return std::max(std::min(x, 1.0), 0.0);
}

// Base class for all conditions
struct Condition
{
    Condition(const Settings &settings_, int player_)
        : settings(settings_), player(player_), status(Status::kUnresolved), partial_value(0.0) {}

    virtual ~Condition() = default;

    virtual void reset() {}
    virtual void on_trick_end([[maybe_unused]] const State &state) {}
    virtual void on_end() {}

    const Settings &settings;
    int player;
    Status status;
    double partial_value;
};
using ConditionPtr = std::shared_ptr<Condition>;

// Condition for checking trick-related conditions
struct TrickCond : public Condition
{
    explicit TrickCond(const Settings &settings_, int player_, int trick_, bool no_ = false)
        : Condition(settings_, player_), trick(trick_), no(no_) {}

    void reset() override
    {
        status = Status::kUnresolved;
    }

    void on_trick_end(const State &state) override;

    void on_end() override
    {
        if (status == Status::kUnresolved)
        {
            status = Status::kFail;
        }
    }

    int trick;
    bool no;
};

// Condition for checking card-related conditions
struct CardCond : public Condition
{
    explicit CardCond(const Settings &settings_, int player_, const Card &card_)
        : Condition(settings_, player_), card(card_) {}

    void reset() override
    {
        status = Status::kUnresolved;
    }

    void on_trick_end(const State &state) override;

    void on_end() override
    {
        if (status == Status::kUnresolved)
        {
            status = Status::kFail;
        }
    }

    Card card;
};

// Condition for cumulative trick-related conditions
struct CumTrickCond : public Condition
{
    explicit CumTrickCond(const Settings &settings_, int player_, Direction direction_,
                          const std::variant<int, TrickCountType> &num_tricks_)
        : Condition(settings_, player_), direction(direction_), num_tricks(num_tricks_),
          num_tricks_won(0), num_other_tricks_won(0)
    {
        if (std::holds_alternative<TrickCountType>(num_tricks) &&
            std::get<TrickCountType>(num_tricks) == TrickCountType::kAnyOther)
        {
            num_other_tricks_won_per_player = std::vector<int>(settings.num_players, 0);
        }
    }

    void reset() override
    {
        status = Status::kUnresolved;
        partial_value = 0.0;
        num_tricks_won = 0;

        if (std::holds_alternative<TrickCountType>(num_tricks) &&
            std::get<TrickCountType>(num_tricks) == TrickCountType::kAnyOther)
        {
            num_other_tricks_won_per_player = std::vector<int>(settings.num_players, 0);
        }
        else
        {
            num_other_tricks_won = 0;
        }
    }

    int get_num_other_tricks() const
    {
        if (std::holds_alternative<int>(num_tricks))
        {
            return std::get<int>(num_tricks);
        }
        else if (std::get<TrickCountType>(num_tricks) == TrickCountType::kCapt ||
                 std::get<TrickCountType>(num_tricks) == TrickCountType::kSumOthers)
        {
            return num_other_tricks_won;
        }
        else
        {
            assert(std::get<TrickCountType>(num_tricks) == TrickCountType::kAnyOther);

            int num_other_tricks = -1;

            for (size_t i = 0; i < num_other_tricks_won_per_player.size(); ++i)
            {
                if (i == static_cast<size_t>(player))
                    continue;

                if (direction == Direction::kGreaterThanOrEqual || direction == Direction::kGreaterThan)
                {
                    if (num_other_tricks == -1)
                        num_other_tricks = num_other_tricks_won_per_player[i];
                    else
                        num_other_tricks = std::max(num_other_tricks, num_other_tricks_won_per_player[i]);
                }
                else
                {
                    if (num_other_tricks == -1)
                        num_other_tricks = num_other_tricks_won_per_player[i];
                    else
                        num_other_tricks = std::min(num_other_tricks, num_other_tricks_won_per_player[i]);
                }
            }

            return num_other_tricks;
        }
    }

    void on_trick_end(const State &state) override;
    void on_end() override;

    Direction direction;
    std::variant<int, TrickCountType> num_tricks;
    int num_tricks_won;
    int num_other_tricks_won;
    std::vector<int> num_other_tricks_won_per_player;
};

// Condition for cumulative card-related conditions
struct CumCardCond : public Condition
{
    explicit CumCardCond(const Settings &settings_, int player_, Direction direction_,
                         const CardFilter &card_filter_,
                         const std::optional<int> &num_cards_ = std::nullopt,
                         const std::optional<CardFilter> &other_card_filter_ = std::nullopt)
        : Condition(settings_, player_), direction(direction_), card_filter(card_filter_),
          num_cards(num_cards_), other_card_filter(other_card_filter_),
          num_cards_won(0), num_other_cards_won(0) {}

    void reset() override
    {
        status = Status::kUnresolved;
        partial_value = 0.0;
        num_cards_won = 0;
        num_other_cards_won = 0;
    }

    int get_num_other_cards() const
    {
        return num_cards.has_value() ? num_cards.value() : num_other_cards_won;
    }

    void on_trick_end(const State &state) override;

    void on_end() override;

    Direction direction;
    CardFilter card_filter;
    std::optional<int> num_cards;
    std::optional<CardFilter> other_card_filter;
    int num_cards_won;
    int num_other_cards_won;
};

// Condition for checking if a card was won with a specific filter
struct WithCond : public Condition
{
    explicit WithCond(const Settings &settings_, int player_, const CardFilter &card_filter_)
        : Condition(settings_, player_), card_filter(card_filter_) {}

    void reset() override
    {
        status = Status::kUnresolved;
    }

    void on_trick_end(const State &state) override;

    void on_end() override
    {
        if (status != Status::kSuccess)
        {
            status = Status::kFail;
        }
    }

    CardFilter card_filter;
};

// Condition for checking if all cards of a suit were won
struct SweepCond : public Condition
{
    explicit SweepCond(const Settings &settings_, int player_, int num_sweeps_)
        : Condition(settings_, player_), num_sweeps(num_sweeps_) {}

    void reset() override
    {
        status = Status::kUnresolved;
        partial_value = 0.0;
        cards_won_per_suit = std::vector<int>(settings.num_side_suits, 0);
    }

    void on_trick_end(const State &state) override;

    void on_end() override
    {
        if (status != Status::kSuccess)
        {
            status = Status::kFail;
        }
    }

    int num_sweeps;
    std::vector<int> cards_won_per_suit;
};

// Condition for checking consecutive tricks
struct ConsecCond : public Condition
{
    explicit ConsecCond(const Settings &settings_, int player_, int num_consec_, bool no_ = false)
        : Condition(settings_, player_), num_consec(num_consec_), no(no_),
          cur_consec_start(std::nullopt), cur_consec_end(std::nullopt)
    {
        assert(num_consec >= 2);
    }

    void reset() override
    {
        status = Status::kUnresolved;
        partial_value = 0.0;
        cur_consec_start = std::nullopt;
        cur_consec_end = std::nullopt;
    }

    void on_trick_end(const State &state) override;

    void on_end() override;

    int num_consec;
    bool no;
    std::optional<int> cur_consec_start;
    std::optional<int> cur_consec_end;
};

// Condition for checking sum of card ranks
struct SumCond : public Condition
{
    explicit SumCond(const Settings &settings_, int player_, Direction direction_, int sum_)
        : Condition(settings_, player_), direction(direction_), sum(sum_) {}

    void reset() override
    {
        status = Status::kUnresolved;
    }

    void on_trick_end(const State &state) override;

    void on_end() override
    {
        if (status != Status::kSuccess)
        {
            status = Status::kFail;
        }
    }

    Direction direction;
    int sum;
};

// Condition for checking if a player leads with certain suits
struct NoLeadCond : public Condition
{
    explicit NoLeadCond(const Settings &settings_, int player_, const std::vector<int> &suits_)
        : Condition(settings_, player_), suits(suits_), num_bad_lead(0) {}

    void reset() override
    {
        status = Status::kUnresolved;
        partial_value = 0.0;
        num_bad_lead = 0;
    }

    void on_trick_end(const State &state) override;

    void on_end() override
    {
        if (status == Status::kUnresolved)
        {
            status = Status::kSuccess;
        }
    }

    std::vector<int> suits;
    int num_bad_lead;
};

// Helper functions for parsing and comparing
inline std::pair<Direction, std::string> parse_dir(const std::string &token)
{
    if (token.substr(0, 2) == "<=" || token.substr(0, 2) == ">=")
    {
        Direction dir = token.substr(0, 2) == "<=" ? Direction::kLessThanOrEqual : Direction::kGreaterThanOrEqual;
        return {dir, token.substr(2)};
    }

    assert(token[0] == '>' || token[0] == '<' || token[0] == '=');
    Direction dir;
    switch (token[0])
    {
    case '>':
        dir = Direction::kGreaterThan;
        break;
    case '<':
        dir = Direction::kLessThan;
        break;
    case '=':
        dir = Direction::kEqual;
        break;
    default:
        throw std::runtime_error("Invalid direction");
    }
    return {dir, token.substr(1)};
}

inline std::pair<std::optional<Card>, std::string> parse_card(const std::string &token)
{
    if (token.length() >= 2 && std::isdigit(token[0]) && TO_SUIT_NUM.find(token[1]) != TO_SUIT_NUM.end())
    {
        int rank = token[0] - '0';
        int suit = TO_SUIT_NUM.at(token[1]);
        return {Card(rank, suit), token.substr(2)};
    }
    return {std::nullopt, token};
}

inline bool compare_dir(int a, int b, Direction dir)
{
    switch (dir)
    {
    case Direction::kGreaterThan:
        return a > b;
    case Direction::kLessThan:
        return a < b;
    case Direction::kGreaterThanOrEqual:
        return a >= b;
    case Direction::kEqual:
        return a == b;
    case Direction::kLessThanOrEqual:
        return a <= b;
    default:
        throw std::runtime_error("Invalid direction");
    }
}

inline std::pair<std::optional<CardFilter>, std::string> parse_card_filter(const std::string &token)
{
    if (token.substr(0, 3) == "odd" || token.substr(0, 4) == "even")
    {
        bool is_odd = token.substr(0, 3) == "odd";
        std::string remaining = token.substr(is_odd ? 3 : 4);

        auto filt = [is_odd](const Card &card) -> bool
        {
            return is_odd ? (card.rank % 2 == 1) : (card.rank % 2 == 0);
        };

        return {filt, remaining};
    }
    else if (token.substr(0, 5) == "rank(")
    {
        std::string rank_filter = token.substr(5);
        size_t close_paren = rank_filter.find(')');
        if (close_paren == std::string::npos)
        {
            throw std::runtime_error("Missing closing parenthesis in rank filter");
        }

        std::string dir_str = rank_filter.substr(0, close_paren);
        std::string remaining = rank_filter.substr(close_paren + 1);

        auto [direction, num_str] = parse_dir(dir_str);
        int num = std::stoi(num_str);

        auto filt = [direction, num](const Card &card) -> bool
        {
            return compare_dir(card.rank, num, direction);
        };

        return {filt, remaining};
    }
    else
    {
        std::optional<int> rank;
        std::optional<int> suit;

        if (!token.empty() && std::isdigit(token[0]))
        {
            rank = token[0] - '0';
            if (token.length() > 1 && TO_SUIT_NUM.find(token[1]) != TO_SUIT_NUM.end())
            {
                suit = TO_SUIT_NUM.at(token[1]);
                return {
                    [rank, suit](const Card &card) -> bool
                    {
                        return card.rank == rank && card.suit == suit;
                    },
                    token.substr(2)};
            }
            return {
                [rank](const Card &card) -> bool
                {
                    return card.rank == rank;
                },
                token.substr(1)};
        }
        else if (!token.empty() && TO_SUIT_NUM.find(token[0]) != TO_SUIT_NUM.end())
        {
            suit = TO_SUIT_NUM.at(token[0]);
            return {
                [suit](const Card &card) -> bool
                {
                    return card.suit == suit;
                },
                token.substr(1)};
        }

        return {std::nullopt, token};
    }
}

// Function to parse a token into a condition
std::shared_ptr<Condition> parse_token(const std::string &token, const Settings &settings, int player);

// Task class representing a game task
struct Task
{
    Task(const std::string &formula_, const std::string &desc_, int difficulty_, int task_idx_)
        : formula(formula_), desc(desc_.empty() ? formula_ : desc_),
          difficulty(difficulty_), task_idx(task_idx_) {}

    bool operator==(const Task &other) const
    {
        return formula == other.formula;
    }

    std::string to_string() const
    {
        return desc;
    }

    std::string to_repr() const
    {
        return "Task(" + desc + ")";
    }

    std::string formula;
    std::string desc;
    int difficulty;
    int task_idx;
};

// Assigned task class representing a task assigned to a player
struct AssignedTask : public Task
{
    AssignedTask(const std::string &formula_, const std::string &desc_, int difficulty_,
                 int task_idx_, int player_, const Settings &settings_)
        : Task(formula_, desc_, difficulty_, task_idx_), player(player_), settings(settings_),
          status(Status::kUnresolved), value(0.0)
    {
        parse_formula();
    }

    void parse_formula()
    {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream iss(formula);

        while (iss >> token)
        {
            tokens.push_back(token);
        }

        try
        {
            in_one_trick = false;
            if (!tokens.empty() && tokens[0] == "1T")
            {
                in_one_trick = true;
                tokens.erase(tokens.begin());
            }

            for (const auto &token : tokens)
            {
                conds.push_back(parse_token(token, settings, player));
            }
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Could not parse " + formula + ": " + e.what());
        }

        for (auto &cond : conds)
        {
            cond->reset();
        }
    }

    void compute_value()
    {
        if (in_one_trick)
        {
            value = (status == Status::kSuccess) ? 1.0 : (status == Status::kFail) ? -1.0
                                                                                   : 0.0;
        }
        else
        {
            std::vector<double> cond_values;
            int i = 0;
            for (const auto &cond : conds)
            {
                assert(0 <= cond->partial_value && cond->partial_value <= 1);
                double partial_value = (cond->status == Status::kSuccess) ? 1.0 : cond->partial_value * 2.0 - 1.0;
                cond_values.push_back(partial_value);
                i++;
            }

            double avg_cond_value = 0.0;
            if (!cond_values.empty())
            {
                avg_cond_value = std::accumulate(cond_values.begin(), cond_values.end(), 0.0) /
                                 cond_values.size();
            }

            assert(-1.0 <= avg_cond_value && avg_cond_value <= 1.0);

            double task_bonus = (status == Status::kSuccess) ? settings.task_bonus : (status == Status::kFail) ? -settings.task_bonus
                                                                                                               : 0.0;

            value = (avg_cond_value + task_bonus) / (settings.task_bonus + 1.0);
        }

        assert(-1.0 <= value && value <= 1.0);
    }

    void on_trick_end(const State &state);

    void on_game_end();

    int player;
    const Settings &settings;
    Status status;
    double value;
    bool in_one_trick;
    std::vector<std::shared_ptr<Condition>> conds;
};

inline const std::vector<std::tuple<std::string, std::string, int>> TASK_DEFS = {
    {"1T 7p with(t)", "I will win 7p with a submarine.", 3},
    {
        "1T #rank(>6)=0 #t=0",
        "I will win a trick of which the card values are all less than 7. Submarines are not allowed in the trick.",
        3,
    },
    {
        "1T #p=#b #p>0",
        "I will win as many pink as blue cards in one trick. 0 pink/blue cards is not allowed.",
        3,
    },
    {
        "1T #8>=1 with(4)",
        "I will win an 8 with a 4.",
        4,
    },
    {
        "1T with(5)",
        "I will win a trick using a 5",
        3,
    },
    {
        "1T with(3)",
        "I will win a trick using a 3",
        4,
    },
    {
        "1T sum>28 #t=0",
        "I will win a trick with a total value greater than 28. Submarines are not allowed in the trick.",
        3,
    },
    {
        "1T sum<12 #t=0",
        "I will win a trick with a total value less than 12. Submarines are not allowed in the trick.",
        3,
    },
    {
        "1T sum>=22 sum<=23 #t=0",
        "I will win a trick with a total value of 22 or 23. Submarines are not allowed in the trick.",
        3,
    },
    {
        "1T #6>=2 with(6)",
        "I will win a 6 with another 6.",
        3,
    },
    {
        "1T #odd=0",
        "I will win a trick that contains only even-numbered cards.",
        5,
    },
    {
        "1T #g=#y #g>0",
        "I will win as many green as yellow cards in one trick. 0 green/yellow cards is not allowed.",
        3,
    },
    {
        "1T with(2)",
        "I will win a trick using a 2.",
        4,
    },
    {
        "1T #rank(<=5)=0",
        "I will win a trick of which the card values are all greater than 5.",
        4,
    },
    {
        "1T #even=0",
        "I will win a trick that contains only odd-numbered cards.",
        4,
    },
    {
        "1T T-1 2g",
        "I will win 2g in the final trick of the game.",
        4,
    },
    {"1T with(6)", "I will win a trick using a 6.", 3},
    {"1T #5>=1 with(7)", "I will win a 5 with a 7.", 2},
    {
        "1T 9g with(t)",
        "I will the 9g with a submarine.",
        3,
    },
    {
        "T0 T-1",
        "I will win the frist and the last trick.",
        4,
    },
    // {
    //     "#T<#T(capt)",
    //     "I will win fewer tricks than the captain. I am not the captain",
    //     2,
    // },
    {
        "T0 T1 T2",
        "I will win the first 3 tricks.",
        3,
    },
    {
        "6g",
        "I will win 6g",
        1,
    },
    {
        "#7>=2",
        "I will win at least 2 7's.",
        2,
    },
    {
        "5p 6y",
        "I will win 5p 6y.",
        2,
    },
    {
        "#T=2",
        "I will win exactly 2 tricks.",
        2,
    },
    {"#g=2", "I will win exactly 2 greens.", 4},
    {
        "#sweep>=1",
        "I will win all the cards in at least one of the 4 colors.",
        4,
    },
    {"#T>#T(sumothers)", "I will more tricks than everyone else combined.", 4},
    {
        "#t=0",
        "I will win no submarines.",
        1,
    },
    // {
    //     "#T=#T(capt)",
    //     "I will win as many tricks as the captain. I am not the captain.",
    //     3,
    // },
    {
        "8p 5b",
        "I will win 8p and 5b.",
        2,
    },
    {
        "#p>=5",
        "I will at least 5 pinks.",
        3,
    },
    {"#t=2", "I will win exactly 2 submarines.", 3},
    {
        "consec(2) #T=2",
        "I will win exactly 2 tricks and they will be in a row.",
        3,
    },
    {
        "#t=3",
        "I will win exactly 3 submarines.",
        4,
    },
    {
        "T-1",
        "I will win the last trick.",
        3,
    },
    {
        "9p 8y",
        "I will win 9p 8y.",
        3,
    },
    {"3p", "I will win 3p.", 1},
    {"9y 7b", "I will win 9y and 7b.", 3},
    {
        "#T=1",
        "I will win exactly 1 trick.",
        2,
    },
    {
        "consec(3)",
        "I will win 3 tricks in a row.",
        3,
    },
    {
        "#T=0",
        "I will 0 tricks.",
        3,
    },
    {
        "no(T0) no(T1) no(T2)",
        "I will win none of the first 3 tricks.",
        2,
    },
    {"2t #t=1", "I will 2t and no other submarine.", 3},
    {
        "#p>=1 #g>=1 #y>=1 #b>=1",
        "I will win at least one card of each color.",
        3,
    },
    {"#b=2", "I will win exactly 2 blues.", 4},
    {
        "#p=1",
        "I will win exactly 1 pink.",
        3,
    },
    {
        "#5=0",
        "I will no 5",
        2,
    },
    {"3t", "I will win 3t.", 1},
    // {
    //     "#T>#T(capt)",
    //     "I will win more tricks than the captain. I am not the captain.",
    //     2,
    // },
    {
        "T0",
        "I will win the first trick.",
        1,
    },
    {
        "1y",
        "I will win 1y.",
        1,
    },
    {
        "#T>#T(anyother)",
        "I will win more tricks than anyone else.",
        3,
    },
    {"consec(3) #T=3", "I will win exactly 3 tricks and they will be in a row.", 3},
    {
        "#6=3",
        "I will win exactly 3 6's.",
        4,
    },
    {
        "T0 T1",
        "I will win the first 2 tricks.",
        1,
    },
    {
        "#t=1",
        "I will win exactly 1 submarine.",
        3,
    },
    {"#p=0", "I will win no pink.", 2},
    {
        "1t #t=1",
        "I will win 1t and no other submarine.",
        3,
    },
    {
        "consec(2)",
        "I will win 2 tricks in a row.",
        1,
    },
    {
        "1p 7g",
        "I will win 1p and 7g.",
        2,
    },
    {"#9>=3", "I will win at least 3 9's.", 4},
    {
        "T-1 #T=1",
        "I will win only the last trick.",
        4,
    },
    {
        "1b 2b 3b",
        "I will win 1b 2b 3b.",
        3,
    },
    {"T0 #T=1", "I will win only the first trick.", 3},
    {
        "#9=2",
        "I will win exactly 2 9's.",
        3,
    },
    {
        "nolead(y,p,b)",
        "I will not open a trick with yellow, pink, or blue.",
        3,
    },
    {
        "#p>#g",
        "I will win more pink than green cards. 0 green cards is allowed.",
        1,
    },
    {"#y>=7", "I will win at least 7 yellows.", 3},
    {
        "#p=0 #b=0",
        "I will win no pink or blues.",
        3,
    },
    {
        "no(consec(2))",
        "I will never win 2 tricks in a row.",
        2,
    },
    {
        "#8=0 #9=0",
        "I will win no 8 or 9's.",
        3,
    },
    {"#1=0", "I will win no 1.", 2},
    {
        "#p=1 #g=1",
        "I will win exactly 1 pink and 1 green.",
        4,
    },
    {
        "3g 4y 5y",
        "I will 3g 4y and 5y.",
        4,
    },
    {"no(T0) no(T1) no(T2) no(T3) no(T4)", "I will none of the first 5 tricks.", 3},
    {"3b 3g 3y 3p", "I will win 3b 3g 3y 3p.", 4},
    {
        "#y>#b",
        "I will more yellow than blue cards. 0 blue cards is allowed",
        1,
    },
    {
        "#g=0",
        "I will win no greens.",
        2,
    },
    {"#y=0", "I will no yellows.", 2},
    {
        "#1=0 #2=0 #3=0",
        "I will no win 1, 2, or 3's.",
        3,
    },
    {
        "#T=4",
        "I will win exactly 4 tricks.",
        3,
    },
    {
        "#5>=3",
        "I will win at least 3 5's.",
        4,
    },
    {
        "#p=#y #p>0",
        "I will win as many pink as yellow cards. 0 pink/yellow cards is not allowed.",
        4,
    },
    {
        "nolead(p,g)",
        "I will not open with a pink or green.",
        1,
    },
    {
        "#T<#T(anyother)",
        "I will win fewer tricks than anyone else.",
        2,
    },
    {
        "#9=0",
        "I will win no 9.",
        1,
    },
    {
        "#y=0 #g=0",
        "I will win no yellow or greens.",
        3,
    },
    {"no(T0) no(T1) no(T2) no(T3)", "I will win none of the first 4 tricks.", 2},
    {
        "6b 7y",
        "I will the 6b and 7y.",
        2,
    },
    {
        "5g 8b",
        "I will win the 5g and 8b.",
        2,
    },
    {
        "9b 9p 9y 9g",
        "I will win 9b 9p 9y 9g.",
        5,
    },
    {"4b", "I will win 4b.", 1},
};

inline const std::vector<std::tuple<std::string, std::string, int>> EASY_TASK_DEFS = {
    {
        "#T>=1",
        "I will win at least one trick.",
        1,
    },
    {
        "#T>=2",
        "I will win at least two tricks.",
        1,
    },
    {
        "#T>=3",
        "I will win at least three tricks.",
        2,
    },
};

inline const std::vector<std::tuple<std::string, std::string, int>> MED_TASK_DEFS = {{"6g", "I will win 6g", 1}, {"#7>=2", "I will win at least 2 7's.", 2}, {"#T=2", "I will win exactly 2 tricks.", 2}, {"#t=0", "I will win no submarines.", 1}, {"T-1", "I will win the last trick.", 3}, {"3p", "I will win 3p.", 1}, {"consec(3)", "I will win 3 tricks in a row.", 3}, {"no(T0) no(T1) no(T2)", "I will win none of the first 3 tricks.", 2}, {"#p>=1 #g>=1 #y>=1 #b>=1", "I will win at least one card of each color.", 3}, {"#5=0", "I will no 5", 2}, {"3t", "I will win 3t.", 1}, {"T0", "I will win the first trick.", 1}, {"1y", "I will win 1y.", 1}, {"#T>#T(anyother)", "I will win more tricks than anyone else.", 3}, {"#p=0", "I will win no pink.", 2}, {"consec(2)", "I will win 2 tricks in a row.", 1}, {"#9=2", "I will win exactly 2 9's.", 3}, {"nolead(y,p,b)", "I will not open a trick with yellow, pink, or blue.", 3}, {"#p>#g", "I will win more pink than green cards. 0 green cards is allowed.", 1}, {"#p=0 #b=0", "I will win no pink or blues.", 3}, {"no(consec(2))", "I will never win 2 tricks in a row.", 2}, {"#1=0", "I will win no 1.", 2}, {"no(T0) no(T1) no(T2) no(T3) no(T4)", "I will none of the first 5 tricks.", 3}, {"#y>#b", "I will more yellow than blue cards. 0 blue cards is allowed", 1}, {"#g=0", "I will win no greens.", 2}, {"#y=0", "I will no yellows.", 2}, {"nolead(p,g)", "I will not open with a pink or green.", 1}, {"#9=0", "I will win no 9.", 1}, {"no(T0) no(T1) no(T2) no(T3)", "I will win none of the first 4 tricks.", 2}, {"4b", "I will win 4b.", 1}};

// Function to get task definitions
inline const std::vector<std::tuple<std::string, std::string, int>> &
get_task_defs(const std::string &bank)
{
    if (bank == "all")
    {
        return TASK_DEFS;
    }
    else if (bank == "easy")
    {
        return EASY_TASK_DEFS;
    }
    else if (bank == "med")
    {
        return MED_TASK_DEFS;
    }
    else
    {
        throw std::runtime_error("Invalid bank: " + bank);
    }
}