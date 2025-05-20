#include "settings.h"
#include "tasks.h"

Settings::Settings(int num_players_, int num_side_suits_, bool use_trump_suit_, int side_suit_length_, int trump_suit_length_, bool use_signals_, bool cheating_signal_, bool single_signal_, std::string bank_, std::string task_distro_, std::vector<int> task_idxs_, std::optional<int> min_difficulty_, std::optional<int> max_difficulty_, std::optional<int> max_num_tasks_, bool use_drafting_, int num_draft_tricks_, double task_bonus_, double win_bonus_) : num_players(num_players_), num_side_suits(num_side_suits_), use_trump_suit(use_trump_suit_), side_suit_length(side_suit_length_), trump_suit_length(trump_suit_length_), use_signals(use_signals_), cheating_signal(cheating_signal_), single_signal(single_signal_), bank(bank_), task_distro(task_distro_), task_idxs(task_idxs_), min_difficulty(min_difficulty_), max_difficulty(max_difficulty_), max_num_tasks(max_num_tasks_), use_drafting(use_drafting_), num_draft_tricks(num_draft_tricks_), task_bonus(task_bonus_), win_bonus(win_bonus_), task_defs(get_task_defs(bank))
{
    num_cards = num_side_suits * side_suit_length + (use_trump_suit ? trump_suit_length : 0);
    num_tricks = num_cards / num_players;
    max_hand_size = (num_cards - 1) / num_players + 1;
    num_task_defs = task_defs.size();
    use_nosignal = use_signals && !single_signal && !cheating_signal;
    max_suit_length = use_trump_suit ? std::max(side_suit_length, trump_suit_length) : side_suit_length;
    num_suits = num_side_suits + (use_trump_suit ? 1 : 0);
    num_phases = 1 + (use_signals ? 1 : 0) + (use_drafting ? 1 : 0);

    if (!task_idxs.empty())
    {
        resolved_max_num_tasks = task_idxs.size();
    }
    else
    {
        resolved_max_num_tasks = max_num_tasks.value();
    }

    max_num_actions = max_hand_size + (use_signals ? 1 : 0);
    if (use_drafting)
    {
        max_num_actions = std::max(max_num_actions, resolved_max_num_tasks + 1);
    }

    seq_length = num_players * (num_tricks * (use_signals && !single_signal ? 2 : 1) + (single_signal ? 1 : 0) + (use_drafting ? num_draft_tricks : 0));

    for (int i = 0; i < num_side_suits; ++i)
    {
        suits.push_back(i);
    }
    if (use_trump_suit)
    {
        suits.push_back(TRUMP_SUIT_NUM);
    }
}