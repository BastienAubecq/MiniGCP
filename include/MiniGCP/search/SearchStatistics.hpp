#ifndef MINIGCP_SEARCH_STATISTICS
#define MINIGCP_SEARCH_STATISTICS

#include <cstdint>

namespace cp {

    /**
     * @brief Statistics collected by the solver during the search
     */
    class SearchStatistics {
    public:
        /**
         * @brief Best objective value found for an optimization problem
         */
        int32_t objective;

        /**
         * @brief Number of nodes explored
         */
        int32_t nNodes = 0;

        /**
         * @brief Number of solutions found
         */
        int32_t nSolutions = 0;

        /**
         * @brief Number of failures
         */
        int32_t nFailures = 0;
    };

}

#endif