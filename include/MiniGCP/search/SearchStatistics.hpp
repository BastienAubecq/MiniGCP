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
        uint32_t nNodes = 0;

        /**
         * @brief Number of solutions found
         */
        uint32_t nSolutions = 0;

        /**
         * @brief Number of failures
         */
        uint32_t nFailures = 0;
    };

}

#endif