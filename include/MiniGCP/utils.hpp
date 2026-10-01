#ifndef MINIGCP_UTILS
#define MINIGCP_UTILS

#include <cstdint>
#include "sycl/sycl.hpp"

namespace cp {

    /**
     * @brief Argmin reduction over a group of threads
     * @param group The group
     * @param value Current thread's value
     * @param index Index of the value
     * @return Index of the minimum value in the group
     */
    inline int32_t argmin_over_group(sycl::group<1> group, int32_t value, int32_t index) {
        int64_t packed = static_cast<int64_t>(value) << 32 | static_cast<uint32_t>(index);
        return static_cast<int32_t>(sycl::reduce_over_group(group, packed, sycl::minimum<int64_t>()));
    }

    /**
     * @brief Argmax reduction over a group of threads
     * @param group The group
     * @param value Current thread's value
     * @param index Index of the value
     * @return Index of the maximum value in the group
     */
    inline int32_t argmax_over_group(sycl::group<1> group, int32_t value, int32_t index) {
        int64_t packed = static_cast<int64_t>(value) << 32 | static_cast<uint32_t>(index);
        return static_cast<int32_t>(sycl::reduce_over_group(group, packed, sycl::maximum<int64_t>()));
    }

}

#endif