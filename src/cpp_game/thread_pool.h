#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <future>
#include <functional>
#include <condition_variable>

class ThreadPool
{
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    template <typename F, typename... Args>
    auto enqueue(F &&f, Args &&...args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using return_type = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::lock_guard lock(queue_mutex);
            tasks.emplace([task]
                          { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;
};

inline void parallel_exec(ThreadPool &pool, const std::vector<std::function<void()>> &tasks)
{
    std::vector<std::future<void>> futures;
    for (const auto &task : tasks)
    {
        futures.push_back(pool.enqueue(task));
    }
    for (auto &fut : futures)
    {
        fut.get(); // wait for all to finish
    }
}