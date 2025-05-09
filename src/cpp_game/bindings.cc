#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>

#include "types.h"
#include "settings.h"
#include "state.h"
#include "tasks.h"
#include "utils.h"
#include "engine.h"
#include "rollout.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_game, m)
{
    m.doc() = "Python bindings for the C++ game engine"; // optional module docstring

    // Bind Card class
    py::class_<Card>(m, "Card")
        .def(py::init<int, int>())
        .def_readwrite("rank", &Card::rank)
        .def_readwrite("suit", &Card::suit)
        .def("is_trump", &Card::is_trump)
        .def("__str__", &Card::to_string)
        .def("__repr__", &Card::to_string)
        .def(py::self == py::self)
        .def(py::self != py::self);

    // Bind Action class
    py::class_<Action>(m, "Action")
        .def(py::init<int, ActionType, const std::optional<Card> &>())
        .def_readwrite("player", &Action::player)
        .def_readwrite("type", &Action::type)
        .def_readwrite("card", &Action::card)
        .def("__str__", &Action::to_string)
        .def("__repr__", &Action::to_string)
        .def(py::self == py::self)
        .def(py::self != py::self);

    // Bind Signal class
    py::class_<Signal>(m, "Signal")
        .def(py::init<const Card &, SignalValue, int>())
        .def_readwrite("card", &Signal::card)
        .def_readwrite("value", &Signal::value)
        .def_readwrite("trick", &Signal::trick)
        .def("__str__", &Signal::to_string)
        .def("__repr__", &Signal::to_string)
        .def(py::self == py::self)
        .def(py::self != py::self);

    // Bind Settings class
    py::class_<Settings>(m, "Settings")
        .def(py::init<>())
        .def_readwrite("num_players", &Settings::num_players)
        .def_readwrite("num_side_suits", &Settings::num_side_suits)
        .def_readwrite("use_trump_suit", &Settings::use_trump_suit)
        .def_readwrite("side_suit_length", &Settings::side_suit_length)
        .def_readwrite("trump_suit_length", &Settings::trump_suit_length)
        .def_readwrite("use_signals", &Settings::use_signals)
        .def_readwrite("bank", &Settings::bank)
        .def_readwrite("task_distro", &Settings::task_distro)
        .def_readwrite("task_idxs", &Settings::task_idxs)
        .def_readwrite("min_difficulty", &Settings::min_difficulty)
        .def_readwrite("max_difficulty", &Settings::max_difficulty)
        .def_readwrite("max_num_tasks", &Settings::max_num_tasks)
        .def_readwrite("task_bonus", &Settings::task_bonus)
        .def_readwrite("win_bonus", &Settings::win_bonus)
        .def("num_tricks", &Settings::num_tricks)
        .def("max_hand_size", &Settings::max_hand_size)
        .def("get_suit_idx", &Settings::get_suit_idx)
        .def("get_suits", &Settings::get_suits)
        .def("get_suit_length", &Settings::get_suit_length)
        .def("max_suit_length", &Settings::max_suit_length)
        .def("num_suits", &Settings::num_suits)
        .def("num_phases", &Settings::num_phases)
        .def("get_max_num_tasks", &Settings::get_max_num_tasks)
        .def("validate", &Settings::validate)
        .def("to_string", &Settings::to_string)
        .def("__str__", &Settings::to_string)
        .def("__repr__", &Settings::to_string);

    // Bind State class
    py::class_<State>(m, "State")
        .def(py::init<int, Phase, const std::vector<std::vector<Card>> &,
                      const std::vector<Action> &, int, int, int, int,
                      const std::vector<std::pair<Card, int>> &,
                      const std::vector<std::pair<std::vector<Card>, int>> &,
                      const std::vector<std::optional<Signal>> &,
                      const std::optional<int> &,
                      const std::vector<std::vector<AssignedTask>> &,
                      Status, double>())
        .def_readwrite("num_players", &State::num_players)
        .def_readwrite("phase", &State::phase)
        .def_readwrite("hands", &State::hands)
        .def_readwrite("actions", &State::actions)
        .def_readwrite("trick", &State::trick)
        .def_readwrite("leader", &State::leader)
        .def_readwrite("captain", &State::captain)
        .def_readwrite("cur_player", &State::cur_player)
        .def_readwrite("active_cards", &State::active_cards)
        .def_readwrite("past_tricks", &State::past_tricks)
        .def_readwrite("signals", &State::signals)
        .def_readwrite("trick_winner", &State::trick_winner)
        .def_readonly("assigned_tasks", &State::assigned_tasks)
        .def_readwrite("status", &State::status)
        .def_readwrite("value", &State::value)
        .def("get_next_player", &State::get_next_player)
        .def("get_player_idx", &State::get_player_idx)
        .def("phase_idx", &State::phase_idx)
        .def("get_player", &State::get_player)
        .def("get_turn", &State::get_turn)
        .def("lead_suit", &State::lead_suit)
        .def("to_string", &State::to_string)
        .def("__str__", &State::to_string)
        .def("__repr__", &State::to_string);

    // Bind Engine class
    py::class_<Engine>(m, "Engine")
        .def(py::init<const Settings &, const std::optional<int> &>(),
             py::arg("settings"), py::arg("seed") = std::nullopt)
        .def("gen_hands", &Engine::gen_hands)
        .def("gen_tasks", &Engine::gen_tasks)
        .def("reset_state", &Engine::reset_state)
        .def("calc_trick_winner", &Engine::calc_trick_winner)
        .def("skip_to_next_unsignaled", &Engine::skip_to_next_unsignaled)
        .def("move", &Engine::move)
        .def("valid_actions", &Engine::valid_actions)
        .def_readonly("state", &Engine::state);

    // Bind utils functions
    m.def("split_by_suit", &split_by_suit, "Split a hand of cards by suit");

    // Bind enums
    py::enum_<Phase>(m, "Phase")
        .value("signal", Phase::kSignal)
        .value("play", Phase::kPlay)
        .value("end", Phase::kEnd)
        .export_values();

    py::enum_<Status>(m, "Status")
        .value("success", Status::kSuccess)
        .value("fail", Status::kFail)
        .value("unresolved", Status::kUnresolved)
        .export_values();

    py::enum_<Direction>(m, "Direction")
        .value("kGreaterThan", Direction::kGreaterThan)
        .value("kLessThan", Direction::kLessThan)
        .value("kGreaterThanOrEqual", Direction::kGreaterThanOrEqual)
        .value("kEqual", Direction::kEqual)
        .value("kLessThanOrEqual", Direction::kLessThanOrEqual)
        .export_values();

    py::enum_<TrickCountType>(m, "TrickCountType")
        .value("kCapt", TrickCountType::kCapt)
        .value("kSumOthers", TrickCountType::kSumOthers)
        .value("kAnyOther", TrickCountType::kAnyOther)
        .export_values();

    // Bind Condition class and its derived classes
    py::class_<Condition, std::shared_ptr<Condition>>(m, "Condition")
        .def("reset", &Condition::reset)
        .def("on_trick_end", &Condition::on_trick_end)
        .def("on_end", &Condition::on_end)
        .def_readwrite("status", &Condition::status)
        .def_readwrite("partial_value", &Condition::partial_value)
        .def_readwrite("player", &Condition::player);

    py::class_<TrickCond, Condition, std::shared_ptr<TrickCond>>(m, "TrickCond")
        .def(py::init<const Settings &, int, int, bool>())
        .def("reset", &TrickCond::reset)
        .def("on_trick_end", &TrickCond::on_trick_end)
        .def("on_end", &TrickCond::on_end);

    py::class_<CardCond, Condition, std::shared_ptr<CardCond>>(m, "CardCond")
        .def(py::init<const Settings &, int, const Card &>())
        .def("reset", &CardCond::reset)
        .def("on_trick_end", &CardCond::on_trick_end)
        .def("on_end", &CardCond::on_end);

    py::class_<CumTrickCond, Condition, std::shared_ptr<CumTrickCond>>(m, "CumTrickCond")
        .def(py::init<const Settings &, int, Direction, const std::variant<int, TrickCountType> &>())
        .def("reset", &CumTrickCond::reset)
        .def("get_num_other_tricks", &CumTrickCond::get_num_other_tricks)
        .def("on_trick_end", &CumTrickCond::on_trick_end)
        .def("on_end", &CumTrickCond::on_end);

    py::class_<CumCardCond, Condition, std::shared_ptr<CumCardCond>>(m, "CumCardCond")
        .def(py::init<const Settings &, int, Direction, const std::function<bool(const Card &)> &,
                      const std::optional<int> &, const std::optional<std::function<bool(const Card &)>> &>())
        .def("reset", &CumCardCond::reset)
        .def("get_num_other_cards", &CumCardCond::get_num_other_cards)
        .def("on_trick_end", &CumCardCond::on_trick_end)
        .def("on_end", &CumCardCond::on_end);

    py::class_<WithCond, Condition, std::shared_ptr<WithCond>>(m, "WithCond")
        .def(py::init<const Settings &, int, const std::function<bool(const Card &)> &>())
        .def("reset", &WithCond::reset)
        .def("on_trick_end", &WithCond::on_trick_end)
        .def("on_end", &WithCond::on_end);

    py::class_<SweepCond, Condition, std::shared_ptr<SweepCond>>(m, "SweepCond")
        .def(py::init<const Settings &, int, int>())
        .def("reset", &SweepCond::reset)
        .def("on_trick_end", &SweepCond::on_trick_end)
        .def("on_end", &SweepCond::on_end);

    py::class_<ConsecCond, Condition, std::shared_ptr<ConsecCond>>(m, "ConsecCond")
        .def(py::init<const Settings &, int, int, bool>())
        .def_readonly("num_consec", &ConsecCond::num_consec)
        .def_readonly("no", &ConsecCond::no)
        .def_readonly("cur_consec_start", &ConsecCond::cur_consec_start)
        .def_readonly("cur_consec_end", &ConsecCond::cur_consec_end)
        .def("reset", &ConsecCond::reset)
        .def("on_trick_end", &ConsecCond::on_trick_end)
        .def("on_end", &ConsecCond::on_end);

    py::class_<SumCond, Condition, std::shared_ptr<SumCond>>(m, "SumCond")
        .def(py::init<const Settings &, int, Direction, int>())
        .def("reset", &SumCond::reset)
        .def("on_trick_end", &SumCond::on_trick_end)
        .def("on_end", &SumCond::on_end);

    py::class_<NoLeadCond, Condition, std::shared_ptr<NoLeadCond>>(m, "NoLeadCond")
        .def(py::init<const Settings &, int, const std::vector<int> &>())
        .def("reset", &NoLeadCond::reset)
        .def("on_trick_end", &NoLeadCond::on_trick_end)
        .def("on_end", &NoLeadCond::on_end);

    // Bind Task class
    py::class_<Task>(m, "Task")
        .def(py::init<const std::string &, const std::string &, int, int>())
        .def_readonly("formula", &Task::formula)
        .def_readonly("desc", &Task::desc)
        .def_readonly("difficulty", &Task::difficulty)
        .def_readonly("task_idx", &Task::task_idx)
        .def("__str__", &Task::to_string)
        .def("__repr__", &Task::to_string);

    // Bind AssignedTask class
    py::class_<AssignedTask, Task>(m, "AssignedTask")
        .def(py::init<const std::string &, const std::string &, int, int, int, const Settings &>())
        .def_readonly("formula", &AssignedTask::formula)
        .def_readonly("desc", &AssignedTask::desc)
        .def_readonly("difficulty", &AssignedTask::difficulty)
        .def_readonly("task_idx", &AssignedTask::task_idx)
        .def_readonly("player", &AssignedTask::player)
        .def_readonly("status", &AssignedTask::status)
        .def_readonly("value", &AssignedTask::value)
        .def_readonly("conds", &AssignedTask::conds)
        .def_readonly("in_one_trick", &AssignedTask::in_one_trick)
        .def("on_trick_end", &AssignedTask::on_trick_end)
        .def("on_game_end", &AssignedTask::on_game_end)
        .def("compute_value", &AssignedTask::compute_value);

    // Bind utility functions
    m.def("parse_dir", &parse_dir, "Parse direction from string");
    m.def("parse_card", &parse_card, "Parse card from string");
    m.def("compare_dir", &compare_dir, "Compare two integers based on direction");
    m.def("parse_card_filter", &parse_card_filter, "Parse card filter from string");
    m.def("parse_token", &parse_token, "Parse token into condition");
    m.def("get_task_defs", &get_task_defs, "Get task definitions");
    m.def("get_preset", &get_preset, "Get preset");

    // Bind enums
    py::enum_<ActionType>(m, "ActionType")
        .value("play", ActionType::kPlay)
        .value("signal", ActionType::kSignal)
        .value("nosignal", ActionType::kNoSignal);

    py::enum_<SignalValue>(m, "SignalValue")
        .value("singleton", SignalValue::kSingleton)
        .value("highest", SignalValue::kHighest)
        .value("lowest", SignalValue::kLowest);

    // Bind MoveInputs struct
    py::class_<MoveInputs>(m, "MoveInputs")
        .def(py::init<int, int, int>())
        .def_readwrite("hist_player_idxs", &MoveInputs::hist_player_idxs)
        .def_readwrite("hist_tricks", &MoveInputs::hist_tricks)
        .def_readwrite("hist_cards", &MoveInputs::hist_cards)
        .def_readwrite("hist_turns", &MoveInputs::hist_turns)
        .def_readwrite("hist_phases", &MoveInputs::hist_phases)
        .def_readwrite("hand", &MoveInputs::hand)
        .def_readwrite("player_idx", &MoveInputs::player_idx)
        .def_readwrite("trick", &MoveInputs::trick)
        .def_readwrite("turn", &MoveInputs::turn)
        .def_readwrite("phase", &MoveInputs::phase)
        .def_readwrite("valid_actions", &MoveInputs::valid_actions)
        .def_readwrite("task_idxs", &MoveInputs::task_idxs);

    // Bind RolloutResults struct
    py::class_<RolloutResults>(m, "RolloutResults")
        .def(py::init<int, int, int, int>())
        .def_readwrite("hist_player_idxs", &RolloutResults::hist_player_idxs)
        .def_readwrite("hist_tricks", &RolloutResults::hist_tricks)
        .def_readwrite("hist_cards", &RolloutResults::hist_cards)
        .def_readwrite("hist_turns", &RolloutResults::hist_turns)
        .def_readwrite("hist_phases", &RolloutResults::hist_phases)
        .def_readwrite("hand", &RolloutResults::hand)
        .def_readwrite("player_idx", &RolloutResults::player_idx)
        .def_readwrite("trick", &RolloutResults::trick)
        .def_readwrite("turn", &RolloutResults::turn)
        .def_readwrite("phase", &RolloutResults::phase)
        .def_readwrite("valid_actions", &RolloutResults::valid_actions)
        .def_readwrite("task_idxs", &RolloutResults::task_idxs)
        .def_readwrite("probs", &RolloutResults::probs)
        .def_readwrite("log_probs", &RolloutResults::log_probs)
        .def_readwrite("actions", &RolloutResults::actions)
        .def_readwrite("rewards", &RolloutResults::rewards)
        .def_readwrite("num_success_tasks_pp", &RolloutResults::num_success_tasks_pp)
        .def_readwrite("win", &RolloutResults::win);

    // Bind Rollout class
    py::class_<Rollout>(m, "Rollout")
        .def(py::init<const Settings &, int>(),
             py::arg("settings"), py::arg("engine_seed"))
        .def("move", &Rollout::move,
             py::arg("action_idx"), py::arg("probs"), py::arg("log_probs"));

    // Bind BatchRollout class
    py::class_<BatchRollout>(m, "BatchRollout")
        .def(py::init<const Settings &, int, const std::vector<int>>(),
             py::arg("settings"), py::arg("num_rollouts"),
             py::arg("engine_seeds"))
        .def("get_move_inputs", &BatchRollout::get_move_inputs)
        .def("move", &BatchRollout::move,
             py::arg("action_indices"), py::arg("probs"), py::arg("log_probs"))
        .def("is_done", &BatchRollout::is_done)
        .def("get_results", &BatchRollout::get_results);

    py::class_<Rng>(m, "Rng")
        .def(py::init<const std::optional<int> &>(),
             py::arg("seed") = std::nullopt)
        .def("randint", &Rng::randint)
        .def("shuffle_idxs", &Rng::shuffle_idxs);
}