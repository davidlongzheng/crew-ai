#include <atomic>
#include <cstdint>
#include <iostream>
#include <functional>
#include <thread>
#include <vector>

class FixedThreadPool
{
public:
    explicit FixedThreadPool(std::vector<std::vector<std::function<void()>>> funcs)
        : funcs_(std::move(funcs)),
          n_(funcs_.size()),
          epoch_(0),
          stage_(0),
          remaining_(0),
          done_(false)
    {
        threads_.reserve(n_);
        for (std::size_t i = 0; i < n_; ++i)
        {
            threads_.emplace_back(
                [this, i]
                {
                    std::int32_t local_epoch = 0;
                    while (true)
                    {
                        std::int32_t e = epoch_.load(std::memory_order_acquire);
                        if (e == -1)
                            return; // global shutdown
                        if (e == local_epoch)
                        {
                            epoch_.wait(e, std::memory_order_acquire);
                            continue;
                        }
                        local_epoch = e; // new generation

                        funcs_[i][stage_.load(std::memory_order_acquire)]();
                        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                            done_.store(true, std::memory_order_release);
                    }
                });
        }
    }

    /** Execute every worker’s fixed function in parallel, then return */
    void run(int stage)
    {
        stage_.store(stage, std::memory_order_release);
        remaining_.store(n_, std::memory_order_release);
        done_.store(false, std::memory_order_release);
        // wake everyone up
        epoch_.fetch_add(1, std::memory_order_release);
        epoch_.notify_all();

        while (!done_.load(std::memory_order_acquire))
            std::this_thread::yield(); // spin until last finishes
    }

    ~FixedThreadPool()
    {
        epoch_.store(-1, std::memory_order_release);
        epoch_.notify_all();
        for (auto &t : threads_)
            t.join();
    }

private:
    const std::vector<std::vector<std::function<void()>>> funcs_;
    const std::size_t n_;

    /* --- shared run-time state --- */
    std::atomic<std::int32_t> epoch_;
    std::atomic<std::int32_t> stage_;
    std::atomic<std::size_t> remaining_;
    std::atomic<bool> done_;

    std::vector<std::thread> threads_;
};