#ifndef TEST_UTILS
#define TEST_UTILS

#include <cstdint>
#include <array>
#include <vector>
#include <span>
#include "sycl/sycl.hpp"
#include "MiniGCP/Model.hpp"
#include "MiniGCP/solver/Context.hpp"

/**
 * @brief Run a kernel with a group of threads on the device and get the result of each thread
 * @tparam R Result type
 * @tparam K Kernel type
 * @param allocator Allocator containing some data
 * @param nThreads Number of threads in the group
 * @param kernel The kernel
 */
template<typename R, typename K>
inline std::vector<R> runOnDevice(cp::Allocator& allocator, size_t nThreads, K kernel) {
    auto data = allocator.getData();
    sycl::queue q(sycl::cpu_selector_v, { sycl::property::queue::in_order() });
    std::array<std::byte*, 3> deviceData{};
    for (size_t i = 0; i < data.size(); i++) {
        if (!data[i].empty()) {
            deviceData[i] = reinterpret_cast<std::byte*>(sycl::malloc_device(data[i].size(), q));
            q.memcpy(deviceData[i], data[i].data(), data[i].size()).wait();
        }
    }
    uint32_t temporarySize = allocator.getTemporarySize();
    R* deviceResults = (R*)sycl::malloc_device(nThreads * sizeof(R), q);

    q.submit([=](sycl::handler& h) {
        sycl::stream error(8192, 1024, h);
        sycl::local_accessor<std::byte> temporary(temporarySize + 16, h);
        h.parallel_for(sycl::nd_range<1>(nThreads, nThreads), [=](sycl::nd_item<1> item) {
            cp::Context ctx(item, error, deviceData, temporary.get_multi_ptr<sycl::access::decorated::no>().get() + 16);
            deviceResults[item.get_global_id(0)] = kernel(ctx);
        });
    }).wait();

    std::vector<R> results(nThreads);
    q.memcpy(results.data(), deviceResults, nThreads * sizeof(R)).wait();
    if (!data[2].empty()) {
        q.memcpy(const_cast<std::byte*>(data[2].data()), deviceData[2], data[2].size()).wait();
    }
    for (std::byte* pointer : deviceData) {
        if (pointer != nullptr) sycl::free(pointer, q);
    }
    sycl::free(deviceResults, q);
    return results;
}

template<typename K>
inline void runOnDevice(cp::Allocator& allocator, size_t nThreads, K kernel) {
    runOnDevice<char>(allocator, nThreads, [=](cp::Context ctx) {
        kernel(ctx);
        return 0;
    });
}


/**
 * @brief Combine domain changes
 * @param changes The changes
 * @returns Combined change
 */
inline cp::DomainChange combineChanges(std::vector<cp::DomainChange> const& changes) {
    cp::DomainChange change = cp::DomainChange::none;
    for (cp::DomainChange c : changes) change |= c;
    return change;
}

#endif