#ifndef MINIGCP_SEARCH
#define MINIGCP_SEARCH

#include <cstdint>
#include <optional>
#include <concepts>
#include <type_traits>
#include "MiniGCP/search/Decision.hpp"
#include "MiniGCP/solver/Context.hpp"

namespace cp {

    template<typename T>
    struct IsOptionalDecision : std::false_type {};

    template<typename T>
    struct IsOptionalDecision<std::optional<T>> : std::true_type {
        using Decision = T;
    };

    template<typename T>
    concept OptionalDecision = IsOptionalDecision<T>::value;


    template<typename T>
    concept SearchStrategy = std::is_trivially_copyable_v<T> && requires(T search, Context ctx) {
        /**
         * @brief Decide how to split the search space at a node.
         * This function is called by a group of threads.
         * All threads must return the same value.
         * @param ctx Current context
         * @return The decision
         */
        { search.split(ctx) } -> OptionalDecision;

        /**
         * @brief Get the maximum depth of the search tree with this search strategy.
         * This function is called on the host.
         * The depth of a path from the root to a leaf is defined as the number of branches
         * in that path that are not the right-most ones.
         */
        { search.maxDepth() } -> std::same_as<uint32_t>;
    };


    template<SearchStrategy Strategy>
    using DecisionOf = typename IsOptionalDecision<decltype(std::declval<Strategy>().split(std::declval<Context>()))>::Decision;

}

#endif