#ifndef MINIGCP_SEARCH_LIMIT
#define MINIGCP_SEARCH_LIMIT

#include <concepts>
#include <type_traits>
#include "MiniGCP/search/SearchStatistics.hpp"

namespace cp {

    /**
     * @brief Search limit function.
     * It takes a reference to a `SearchStatistics` instance and returns whether the search should be stopped.
     */
    template<typename T>
    concept SearchLimit = std::is_trivially_copyable_v<T> &&
        std::invocable<T, SearchStatistics> &&
        std::same_as<std::invoke_result_t<T, SearchStatistics>, bool>;


    /**
     * @brief Limit : fully explore the search tree
     */
    inline auto noLimit = [](SearchStatistics) {
        return false;
    };

    /**
     * @brief Limit : stop when the first solution is found
     */
    inline auto firstSolution = [](SearchStatistics stats) {
        return stats.nSolutions > 0;
    };

}

#endif