#pragma once

#include <random>
#include <vector>

struct Rng
{
    Rng(std::optional<int> seed)
    {
        if (seed.has_value())
        {
            rng = std::mt19937(*seed);
        }
        else
        {
            std::random_device rd;
            rng = std::mt19937(rd());
        }
    }

    std::vector<int> shuffle_idxs(int n)
    {
        std::vector<int> vec;
        for (int i = 0; i < n; i++)
        {
            vec.push_back(i);
        }

        std::shuffle(vec.begin(), vec.end(), rng);

        return vec;
    }

    template <typename T>
    std::vector<T> shuffle(const std::vector<T> &vec)
    {
        auto idxs = shuffle_idxs(vec.size());
        std::vector<T> shuffled_vec;
        for (int i = 0; i < (int)vec.size(); i++)
        {
            shuffled_vec.push_back(vec[idxs[i]]);
        }

        return shuffled_vec;
    }

    int randint(int min, int max)
    {
        return std::uniform_int_distribution<int>(min, max)(rng);
    }

    std::mt19937 rng;
};