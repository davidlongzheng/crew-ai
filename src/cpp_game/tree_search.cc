#include "tree_search.h"
#include "state.h"
#include <iomanip>

Node::Node(const std::vector<Action> &valid_actions_, int player_, int idx_, int move_, Node *parent_) : valid_actions(valid_actions_), player(player_), idx(idx_), move(move_), parent(parent_), num_actions(valid_actions.size()), child_W(num_actions, 0.0), child_N(num_actions, 0.0), child_P(num_actions, 0.0), child_V(num_actions, 0.0)
{
    assert((parent == nullptr) == (move == -1));
    for (int i = 0; i < num_actions; i++)
    {
        children.push_back(nullptr);
    }
}

std::vector<float> Node::child_U(float c_puct_base, float c_puct_init)
{
    assert(N() > 0);
    float pb_c = std::log((1 + N() + c_puct_base) / c_puct_base) + c_puct_init;
    float sqrt_N = std::sqrt(N());
    std::vector<float> ret(num_actions, 0.0);
    for (int i = 0; i < num_actions; i++)
    {
        ret[i] = pb_c * child_P[i] * (sqrt_N / (1.0 + child_N[i]));
    }
    return ret;
}

std::vector<float> Node::child_Q()
{
    std::vector<float> ret(num_actions, 0.0);
    for (int i = 0; i < num_actions; i++)
    {
        if (child_N[i] == 0)
        {
            ret[i] = default_Q;
        }
        else
        {
            ret[i] = child_W[i] / child_N[i];
        }
    }
    return ret;
}

float Node::N()
{
    if (parent)
    {
        return parent->child_N[move];
    }
    return _N;
}

void Node::set_N(float N)
{
    if (parent)
    {
        parent->child_N[move] = N;
    }
    else
    {
        _N = N;
    }
}

float Node::V()
{
    if (parent)
    {
        return parent->child_V[move];
    }
    return _V;
}

void Node::set_V(float V)
{
    if (parent)
    {
        parent->child_V[move] = V;
    }
    else
    {
        _V = V;
    }
}

float Node::W()
{
    if (parent)
    {
        return parent->child_W[move];
    }
    return _W;
}

void Node::set_W(float W)
{
    if (parent)
    {
        parent->child_W[move] = W;
    }
    else
    {
        _W = W;
    }
}

float Node::Q()
{
    assert(N() > 0);
    return W() / N();
}

void add_dirichlet_noise(Node *node, float eps, float alpha, std::mt19937 &rng)
{
    std::vector<float> noise(node->num_actions);

    // Generate dirichlet noise
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < node->num_actions; i++)
    {
        noise[i] = gamma(rng);
        sum += noise[i];
    }

    // Normalize
    for (float &n : noise)
    {
        n /= sum;
    }

    // Apply noise to child_P
    for (size_t i = 0; i < node->child_P.size(); i++)
    {
        node->child_P[i] = node->child_P[i] * (1.0f - eps) + noise[i] * eps;
    }
}

TreeSearch::TreeSearch(const Settings &settings_, int num_rollouts_, float c_puct_base_, float c_puct_init_, int num_parallel_, bool root_noise_, bool all_noise_, bool cheating_, float noise_eps_, float noise_alpha_, int seed_)
    : settings(settings_), engine(settings), num_rollouts(num_rollouts_), c_puct_base(c_puct_base_), c_puct_init(c_puct_init_), num_parallel(num_parallel_), root_noise(root_noise_), all_noise(all_noise_), cheating(cheating_), noise_eps(noise_eps_), noise_alpha(noise_alpha_), featurizer(settings, num_rollouts * num_parallel), final_probs({num_rollouts, settings.max_num_actions}), final_values(num_rollouts)
{
    if (seed_ == -1)
    {
        std::random_device rd;
        rng = std::mt19937(rd());
    }
    else
    {
        rng = std::mt19937(seed_);
    }
}

int TreeSearch::best_move(Node *node)
{
    if (!node->is_expanded)
    {
        throw std::runtime_error("Expand leaf node first.");
    }
    std::vector<float> child_Q = node->child_Q();
    std::vector<float> child_U = node->child_U(c_puct_base, c_puct_init);
    std::vector<float> ucb_scores(child_Q.size());
    for (int i = 0; i < ucb_scores.size(); i++)
    {
        ucb_scores[i] = child_Q[i] + child_U[i];
    }
    return std::max_element(ucb_scores.begin(), ucb_scores.end()) - ucb_scores.begin();
}

int TreeSearch::random_move(Node *node)
{
    if (!node->is_expanded)
    {
        throw std::runtime_error("Expand leaf node first.");
    }

    std::discrete_distribution<> dist(node->child_P.begin(), node->child_P.end());

    // Sample from the distribution
    return dist(rng);
}

void TreeSearch::add_virtual_loss(Node *node)
{
    while (node)
    {
        node->set_V(node->V() + 1.0);
        node = node->parent;
    }
}

void TreeSearch::revert_virtual_loss(Node *node)
{
    while (node)
    {
        node->set_V(0.0);
        node = node->parent;
    }
}

void TreeSearch::expand(Node *node, const py::array_t<float> &prior_probs, float value)
{
    if (node->is_expanded)
    {
        throw std::runtime_error("Node already expanded.");
    }

    assert(prior_probs.size() == settings.max_num_actions);
    node->is_expanded = true;

    auto buffer = prior_probs.data();
    node->child_P.assign(buffer, buffer + node->num_actions);
    node->default_Q = value;
    // std::cout << "expand " << node->idx << " " << value << std::endl;
}

void TreeSearch::backup(Node *node, float value)
{
    while (node)
    {
        node->set_N(node->N() + 1);
        node->set_W(node->W() + value);
        node = node->parent;
    }
}

void TreeSearch::reset(const std::vector<State> &states, const py::array_t<float> &prior_probs, const py::array_t<float> &values)
{
    assert(states.size() <= num_rollouts);
    root_states = states;
    cur_node_idx = 0;
    root_nodes.clear();
    for (int i = 0; i < root_states.size(); ++i)
    {
        engine.state = root_states[i];
        if (!cheating)
        {
            engine.set_private(engine.state.cur_player);
        }
        py::slice slice_(i, i + 1, 1);
        py::array_t<float> prior_probs_i = prior_probs[slice_].cast<py::array_t<float>>();
        float value = values.at(i);
        root_nodes.push_back(make_unique<Node>(engine.valid_actions(), engine.state.cur_player, cur_node_idx));
        cur_node_idx++;
        expand(root_nodes[i].get(), prior_probs_i, value);
        if (root_noise || all_noise)
        {
            add_dirichlet_noise(root_nodes[i].get(), noise_eps, noise_alpha, rng);
        }
        backup(root_nodes[i].get(), value);
    }
    leaves.clear();
    rewards.clear();
}

void TreeSearch::print_select(Node *node, int move, std::ostream &stream)
{
    stream << "P" << node->player << " node " << node->idx;
    std::vector<std::tuple<int, float, float, float, float, float>> sorted;

    std::vector<float> child_Q = node->child_Q();
    std::vector<float> child_U = node->child_U(c_puct_base, c_puct_init);

    for (int i = 0; i < node->num_actions; i++)
    {
        if (node->child_P[i] >= 0.05 || i == move)
        {
            sorted.push_back(std::make_tuple(i, node->child_P[i], node->child_N[i], node->child_W[i], child_Q[i], child_U[i]));
        }
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b)
              { return std::get<1>(a) > std::get<1>(b); });
    stream << " choices: [";
    for (const auto &[idx, P, N, W, Q, U] : sorted)
    {
        stream << idx << ":p=" << std::fixed << std::setprecision(3) << P << ",n=" << N << ",w=" << W << ",q=" << Q << ",u=" << U << " ";
    }

    stream << "] move: " << move << std::endl;
}

const MoveInputs &TreeSearch::select_nodes()
{
    // Select nodes
    featurizer.reset();
    leaves.clear();
    rewards.clear();
    leaf_node_idxs.clear();
    for (int rollout_idx = 0; rollout_idx < root_states.size(); rollout_idx++)
    {
        int failsafe = 0;
        Node *node;
        std::unordered_set<Node *> visited;

        while (visited.size() < num_parallel && failsafe < num_parallel * 2)
        {
            failsafe++;
            node = root_nodes[rollout_idx].get();
            engine.state = root_states[rollout_idx];
            int cur_player = engine.state.cur_player;
            double cum_reward = 0.0;
            // std::cout << "path: " << failsafe << " phase: " << phase_to_string(engine.state.phase)
            //           << " player: " << engine.state.cur_player
            //           << " trick: " << engine.state.trick
            //           << " turn: " << engine.state.get_turn() << std::endl;

            while (node->is_expanded && engine.state.phase != Phase::kEnd)
            {
                int move = node->player == cur_player ? best_move(node) : random_move(node);
                // print_select(node, move, std::cout);
                const Action &action = node->valid_actions[move];
                cum_reward += engine.move(action);

                std::vector<Action> valid_actions;
                while ((valid_actions = engine.state.phase == Phase::kEnd ? std::vector<Action>() : engine.valid_actions()).size() == 1)
                {
                    cum_reward += engine.move(valid_actions[0]);
                }

                if (!node->children[move])
                {
                    node->children[move] = std::make_unique<Node>(valid_actions, engine.state.cur_player, cur_node_idx, move, node);
                    cur_node_idx++;
                }

                node = node->children[move].get();
            }

            if (engine.state.phase == Phase::kEnd)
            {
                backup(node, cum_reward);
            }
            else
            {
                add_virtual_loss(node);
                if (!visited.contains(node))
                {
                    visited.insert(node);
                    leaves.push_back(node);
                    rewards.push_back(cum_reward);
                    featurizer.record_move_inputs(engine);
                    leaf_node_idxs.push_back(std::make_tuple(node->idx, node->parent->idx));
                }
            }
        }
    }

    return featurizer.move_inputs;
}

void TreeSearch::expand_nodes(const py::array_t<float> &prior_probs, const py::array_t<float> &values)
{
    for (int i = 0; i < leaves.size(); i++)
    {
        py::slice leaf_slice(i, i + 1, 1);
        py::array_t<float> prior_probs_i = prior_probs[leaf_slice].cast<py::array_t<float>>();
        float value = values.at(i) + rewards[i];
        revert_virtual_loss(leaves[i]);
        expand(leaves[i], prior_probs_i, value);
        if (all_noise)
        {
            add_dirichlet_noise(leaves[i], noise_eps, noise_alpha, rng);
        }
        backup(leaves[i], value);
    }
    // std::cout << std::string(50, '-') << std::endl;
}

std::tuple<py::array_t<float>, py::array_t<float>> TreeSearch::get_final_pv()
{
    auto final_probs_ptr = static_cast<float *>(final_probs.mutable_data());
    auto final_values_ptr = static_cast<float *>(final_values.mutable_data());

    for (int rollout_idx = 0; rollout_idx < root_nodes.size(); ++rollout_idx)
    {
        std::vector<float> probs = root_nodes[rollout_idx]->child_N;
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0);
        for (int i = 0; i < probs.size(); i++)
        {
            probs[i] = probs[i] / sum;
        }
        std::copy(probs.begin(), probs.end(), final_probs_ptr);
        final_probs_ptr += probs.size();
        for (int i = 0; i < settings.max_num_actions - probs.size(); i++)
        {
            *final_probs_ptr = 0.0;
            final_probs_ptr++;
        }
        *final_values_ptr = root_nodes[rollout_idx]->Q();
        final_values_ptr++;
    }

    return std::make_tuple(final_probs, final_values);
}