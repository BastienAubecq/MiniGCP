#ifndef TEST_UTILS
#define TEST_UTILS

#include <cstdint>
#include <vector>
#include <span>
#include "sycl/sycl.hpp"
#include "MiniGCP/Model.hpp"
#include "MiniGCP/solver/Context.hpp"

/**
 * @brief Run a kernel with a group of threads on the device and get the result of each thread
 * @tparam R Result type
 * @tparam K Kernel type
 * @param allocator Allocator containing some variables
 * @param nThreads Number of threads in the group
 * @param kernel The kernel
 */
template<typename R, typename K>
inline std::vector<R> runOnDevice(cp::Allocator& allocator, size_t nThreads, K kernel) {
    std::span<std::byte const> state = allocator.getData()[2];
    sycl::queue q(sycl::cpu_selector_v, { sycl::property::queue::in_order() });
    std::byte* deviceState = (std::byte*)malloc_device(state.size(), q);
    R* deviceResults = (R*)sycl::malloc_device(nThreads * sizeof(R), q);
    q.memcpy(deviceState, state.data(), state.size()).wait();

    q.submit([=](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(nThreads, nThreads), [=](sycl::nd_item<1> item) {
            cp::Context ctx(item, { nullptr, nullptr, deviceState });
            deviceResults[item.get_global_id(0)] = kernel(ctx);
        });
    }).wait();

    std::vector<R> results(nThreads);
    q.memcpy(results.data(), deviceResults, nThreads * sizeof(R)).wait();
    q.memcpy(const_cast<std::byte*>(state.data()), deviceState, state.size()).wait();
    sycl::free(deviceState, q);
    sycl::free(deviceResults, q);
    return results;
}

#endif