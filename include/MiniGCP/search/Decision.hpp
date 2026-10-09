#ifndef MINIGCP_DECISION
#define MINIGCP_DECISION

#include <concepts>
#include <type_traits>
#include "sycl/sycl.hpp"
#include "MiniGCP/solver/Context.hpp"

namespace cp {

    /**
     * @brief Decision taken by the search strategy at a node.
     * It can be applied to modify the state of the node for each branch in order.
     */
    template<typename T>
    concept Decision = sycl::is_device_copyable_v<T> && requires(T decision, Context ctx) {
        /**
         * @brief Apply the decision to move from the parent to the left child
         * @param ctx Current context
         */
        { decision.applyLeft(ctx) } -> std::same_as<void>;

        /**
         * @brief Apply the decision to move from the parent to the right child
         * @param ctx Current context
         */
        { decision.applyRight(ctx) } -> std::same_as<void>;
    };

}

#endif