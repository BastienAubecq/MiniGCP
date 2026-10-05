#ifndef MINIGCP_DECISION
#define MINIGCP_DECISION

#include <concepts>
#include <type_traits>
#include "MiniGCP/solver/Context.hpp"

namespace cp {

    /**
     * @brief Decision taken by the search strategy at a node.
     * It can be applied to modify the state of the node for each branch in order.
     */
    template<typename T>
    concept Decision = std::is_trivially_copyable_v<T> && requires(T decision, Context ctx) {
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