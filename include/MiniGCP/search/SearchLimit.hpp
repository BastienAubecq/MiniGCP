#ifndef MINIGCP_SEARCH_LIMIT
#define MINIGCP_SEARCH_LIMIT

#include <concepts>
#include <type_traits>
#include "sycl/sycl.hpp"
#include "MiniGCP/search/SearchStatistics.hpp"

namespace cp {

    /**
     * @brief Search limit function.
     * It determines whether the search should be stopped based on the current search statistics.
     */
    template<typename T>
    concept SearchLimit = sycl::is_device_copyable_v<T> &&
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