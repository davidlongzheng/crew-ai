#pragma once

#include "settings.h"
#include <unordered_map>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "types.h"
#include "rollout.h"
#include "featurizer.h"
namespace py = pybind11;

struct Node
{
    Node(const std::vector<Action> &valid_actions, int player, int idx, int move = -1, Node *parent = nullptr);

    std::vector<float> child_U(float c_puct_base, float c_puct_init);
    std::vector<float> child_Q();
    float N();
    void set_N(float N);
    float W();
    void set_W(float W);
    float Q();
    float V();
    void set_V(float V);

    const std::vector<Action> valid_actions;
    int player;
    int move = -1;
    Node *parent = nullptr;
    bool is_expanded = false;

    int num_actions;
    // sum of values. Q = W/N
    std::vector<float> child_W;
    // counts
    std::vector<float> child_N;
    // priors
    std::vector<float> child_P;
    // virtual counts
    std::vector<float> child_V;

    // For only the root node.
    float _W = 0.0;
    float _N = 0.0;
    float _V = 0.0;

    float default_Q = 0.0;

    int idx;

    std::vector<std::unique_ptr<Node>> children;
};

struct TreeSearch
{
    TreeSearch(const Settings &settings, int num_rollouts, float c_puct_base, float c_puct_init, int num_parallel, bool root_noise, bool all_noise, bool cheating, float noise_eps, float noise_alpha, int seed);

    int best_move(Node *node);
    int random_move(Node *node);
    void expand(Node *node, const py::array_t<float> &prior_probs, float value);
    void backup(Node *node, float value);

    void add_virtual_loss(Node *node);
    void revert_virtual_loss(Node *node);

    void reset(const std::vector<State> &states, const py::array_t<float> &prior_probs, const py::array_t<float> &values);
    void print_select(Node *node, int move, std::ostream &stream);
    const MoveInputs &select_nodes();
    void expand_nodes(const py::array_t<float> &prior_probs, const py::array_t<float> &values);
    bool is_done();
    void record_move_inputs(const State &state);

    std::tuple<py::array_t<float>, py::array_t<float>> get_final_pv();

    const Settings settings;
    Engine engine;
    const int num_rollouts;
    const float c_puct_base;
    const float c_puct_init;
    const int num_parallel;
    const bool root_noise;
    const bool all_noise;
    const bool cheating;
    const float noise_eps;
    const float noise_alpha;
    std::mt19937 rng;

    std::vector<State> root_states;
    std::vector<std::unique_ptr<Node>> root_nodes;
    int cur_node_idx;
    std::vector<Node *> leaves;
    std::vector<float> rewards;
    std::vector<std::tuple<int, int>> leaf_node_idxs;
    Featurizer featurizer;
    py::array_t<float> final_probs;
    py::array_t<float> final_values;
};