#ifndef MINIGCP_DEVICE_PARAMS
#define MINIGCP_DEVICE_PARAMS

#include <cstdint>
#include <functional>
#include <stdexcept>
#include "sycl/sycl.hpp"

namespace cp {

    class DeviceParams {
    public:
        std::function<int(sycl::device const&)> selector;
        uint32_t groupSize;
        uint32_t nGroups;

        DeviceParams(std::function<int(sycl::device const&)> selector, uint32_t groupSize, uint32_t nGroups) :
            selector(selector), groupSize(groupSize), nGroups(nGroups) {}
    };

}

#endif