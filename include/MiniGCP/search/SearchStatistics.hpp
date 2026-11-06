#ifndef MINIGCP_SEARCH_STATISTICS
#define MINIGCP_SEARCH_STATISTICS

#include <cstdint>
#include <ostream>

namespace cp {

    /**
     * @brief Statistics collected by the solver during the search
     */
    class SearchStatistics {
    public:
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

        /**
         * @brief Best objective value found for an optimization problem
         */
        int32_t objective = INT32_MAX;

        /**
         * @brief Index of the next subproblem to solve
         */
        uint32_t nextSubproblem = 0;

        /**
         * @brief Whether the search tree was fully explored
         */
        bool completed = false;


        /**
         * @brief Increment the node counter
         */
        void addNode() {
            //printf("Node\n");
            u32_atomic(nNodes).fetch_add(1);
        }

        /**
         * @brief Increment the solution counter
         */
        void addSolution() {
            //printf("Solution\n");
            u32_atomic(nSolutions).fetch_add(1);
        }

        /**
         * @brief Increment the failure counter
         */
        void addFailure() {
            //printf("Failure\n");
            u32_atomic(nFailures).fetch_add(1);
        }
        
        /**
         * @brief Update the objective value
         * @param value New value
         */
        void updateObjective(int32_t value) {
            i32_atomic(objective).fetch_min(value);
        }

        /**
         * @brief Get the next subproblem
         */
        uint32_t getSubproblem() {
            return u32_atomic(nextSubproblem).fetch_add(1);
        }


    private:
        using i32_atomic = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;
        using u32_atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>; 
    };


    inline std::ostream& operator<<(std::ostream& stream, SearchStatistics const& stats) {
        return stream << 
            "Solutions: " << stats.nSolutions <<
            "\nNodes: " << stats.nNodes <<
            "\nFailures: " << stats.nFailures <<
            "\nObjective: " << stats.objective <<
            "\nSubproblem: " << stats.nextSubproblem <<
            "\nCompleted: " << stats.completed;
    }

}

#endif