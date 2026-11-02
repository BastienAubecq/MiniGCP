#ifndef MINIGCP_DEVICE_PARAMS
#define MINIGCP_DEVICE_PARAMS

#include <cstdint>
#include <functional>
#include <stdexcept>
#include "sycl/sycl.hpp"

namespace cp {

    /**
     * @brief Device-related parameters.
     * Some parameters are provided as template parameters so that
     * the compiler can use their value to improve performance.
     * @tparam groupSize Work group size
     * @tparam localLocal Whether to store local data in local memory
     * @tparam localState Whether to store state data in local memory
     * @param selector Device selector
     * @param nGroups Number of work groups
     */
    template<uint32_t _groupSize, bool _localLocal = true, bool _localState = true>
    class DeviceParams {
    public:
        constexpr static uint32_t groupSize = _groupSize;
        constexpr static uint32_t localLocal = _localLocal;
        constexpr static uint32_t localState = _localState;

        std::function<int(sycl::device const&)> selector;
        uint32_t nGroups;

        DeviceParams(std::function<int(sycl::device const&)> selector, uint32_t nGroups) :
            selector(selector), nGroups(nGroups) {}
    };


    template<typename T>
    inline constexpr bool is_device_params_v = false;

    template<uint32_t groupSize, bool localLocal, bool localState>
    inline constexpr bool is_device_params_v<DeviceParams<groupSize, localLocal, localState>> = true;

    template<typename T>
    concept IsDeviceParams = is_device_params_v<T>;

}

#endif