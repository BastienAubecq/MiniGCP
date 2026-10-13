#ifndef MINIGCP_UTILS
#define MINIGCP_UTILS

#include <cstdint>
#include <optional>
#include "sycl/sycl.hpp"
#include "MiniGCP/solver/Context.hpp"

namespace cp {

    /**
     * @brief Argmin reduction over a group of threads
     * @param ctx Current context
     * @param value Current thread's value
     * @param index Index of the value
     * @return Index of the minimum value in the group
     */
    inline int32_t groupArgmin(Context ctx, int32_t value, int32_t index) {
        int64_t packed = static_cast<int64_t>(value) << 32 | static_cast<uint32_t>(index);
        return static_cast<int32_t>(sycl::reduce_over_group(ctx.item.get_group(), packed, sycl::minimum<int64_t>()));
    }

    /**
     * @brief Argmax reduction over a group of threads
     * @param ctx Current context
     * @param value Current thread's value
     * @param index Index of the value
     * @return Index of the maximum value in the group
     */
    inline int32_t groupArgmax(Context ctx, int32_t value, int32_t index) {
        int64_t packed = static_cast<int64_t>(value) << 32 | static_cast<uint32_t>(index);
        return static_cast<int32_t>(sycl::reduce_over_group(ctx.item.get_group(), packed, sycl::maximum<int64_t>()));
    }


    /**
     * @brief Broadcast a value from any thread that provides one to a group of threads
     * @param ctx Current context
     * @param value Current thread's value if it provides one.
     * @return The broadcasted value if any thread provides a value
     */
    inline std::optional<int32_t> groupBroadcastAny(Context ctx, std::optional<int32_t> value) {
        int32_t* shared = ctx.getTemporary<int32_t>() - 1;
        if (value) sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>(*shared).store(*value);
        if (!sycl::any_of_group(ctx.item.get_group(), value.has_value())) return {};
        int32_t broadcasted = *shared;
        sycl::group_barrier(ctx.item.get_group());
        return broadcasted;
    }

}

#endif