#ifndef MINIGCP_SEARCH
#define MINIGCP_SEARCH

#include <cstdint>
#include <concepts>
#include <type_traits>
#include "MiniGCP/search/Decision.hpp"
#include "MiniGCP/solver/Context.hpp"

namespace cp {

    template<typename T>
    concept SearchStrategy = std::is_trivially_copyable_v<T> && requires(T search, Context ctx) {
        /**
         * @brief Decide how to split the search space at a node.
         * This function is called by a group of threads.
         * All threads must return the same value.
         * @param ctx Current context
         * @return The decision
         */
        { search.split(ctx) } -> Decision;

        /**
         * @brief Get the maximum depth of the search tree with this search strategy.
         * This function is called on the host.
         * The depth of a path from the root to a leaf is defined as the number of branches
         * in that path that are not the right-most ones.
         */
        { search.maxDepth() } -> std::same_as<uint32_t>;
    };

}

#endif