#ifndef MINIGCP_EPS_SOLVER
#define MINIGCP_EPS_SOLVER

#include <cstdint>
#include <vector>
#include <span>
#include <optional>
#include "sycl/sycl.hpp"
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/Constraint.hpp"
#include "MiniGCP/search/ProblemType.hpp"
#include "MiniGCP/search/SearchLimit.hpp"
#include "MiniGCP/search/SearchStatistics.hpp"
#include "MiniGCP/search/SearchStrategy.hpp"
#include "MiniGCP/solver/ConstraintStore.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/solver/DataStore.hpp"
#include "MiniGCP/solver/DeviceParams.hpp"
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"

namespace cp {

    /**
     * @brief Embarrassingly parallel search solver
     */
    template<SearchStrategy Strategy, SearchLimit Limit, Constraint... Constraints>
    class EPSSolver {
    public:
        /**
         * @brief Create and initialize a solver
         * @param model CP model
         * @param type Problem type
         * @param strategy Search strategy
         * @param limit Search limit
         * @param epsDepth Depth of the subproblems that will be solved in parallel
         * @param params Device parameters
         * @param output Output variables
         * @param objective Objective variable
         */
        EPSSolver(
            Model<Constraints...> model, ProblemType type, Strategy strategy, Limit limit, uint32_t epsDepth,
            DeviceParams params, ArrayIdx<Scope::global, IntVarIdx> output, IntVarIdx objective = IntVarIdx(0, 0)
        ) :
            strategy(strategy), epsDepth(epsDepth), params(params),
            queue(params.selector, sycl::property_list()),
            data(queue, model.getData(), limit, type, objective, output, params.nGroups, strategy.maxDepth()),
            constraints(queue, model.getConstraints()),
            solution(output.size()) {}

        ~EPSSolver() {
            data.destroy(queue);
            constraints.destroy(queue);
        }


        /**
         * @brief Solve the problem
         */
        void solve() {
            // Copy fields passed to the kernel
            Strategy strategy = this->strategy;
            uint32_t epsDepth = this->epsDepth;
            DeviceParams params = this->params;
            DataStore<Decision, Limit> data = this->data;
            ConstraintStore<Constraints...> constraints = this->constraints;

            // Launch kernel
            queue.submit([=](sycl::handler& handler) {
                sycl::local_accessor<std::byte> shared(data.getSharedSize(), handler);
                sycl::stream err(8192, 1024, handler);

                handler.parallel_for(sycl::nd_range<1>(params.nGroups * params.groupSize, params.groupSize), [=](sycl::nd_item<1> item) {
                    Context ctx = data.threadInit(item, shared);

                    while (true) {
                        uint32_t subproblem = data.nextSubproblem(ctx);
                        if (subproblem >= 1 << epsDepth) break; // All subproblems solved
                        data.subproblemInit(ctx);
                        
                        // Dive until reaching the subproblem
                        bool hasSubproblem = true;
                        for (uint32_t i = 0; i < epsDepth; i++) {
                            bool right = subproblem >> (epsDepth - i - 1) & 1; // Each bit of the subproblem index indicates a direction to take
                            sycl::group_barrier(item.get_group());
                            std::optional<Decision> decision = strategy.split(ctx);
                            if (!decision) { // Can't reach subproblem -> the first thread to reach this node explores it
                                hasSubproblem = subproblem >> (epsDepth - i - 1) == 0;
                                break;
                            }
                            if (item.get_local_id(0) == 0) {
                                if (right) decision->applyRight(ctx);
                                else decision->applyLeft(ctx);
                            }
                        }
                        if (!hasSubproblem) continue;

                        // Iterative DFS search
                        uint32_t top = -1;
                        while (data.continueSearch(item)) {
                            sycl::group_barrier(item.get_group());
                            data.addNode(item);
                            data.getObjective(ctx);
                            sycl::group_barrier(item.get_group());

                            //printf("Fixpoint\n");
                            bool inconsistent = constraints.fixpoint(ctx);
                            sycl::group_barrier(item.get_group());
                            if (inconsistent) {
                                //printf("Inconsistency\n");
                                data.addFailure(item);
                            }
                            else {
                                std::optional<Decision> decision = strategy.split(ctx);
                                if (decision) { // Move to left child
                                    if (!data.checkIndex(++top)) {
                                        err << "Stack overflow. This probably means that the search strategy specified an incorrect maximum depth.\n";
                                        break;
                                    }
                                    data.saveState(ctx, top, *decision);
                                    sycl::group_barrier(item.get_group());
                                    if (item.get_local_id(0) == 0) decision->applyLeft(ctx);
                                    continue;
                                }
                                else { // Solution
                                    //printf("Solution\n");
                                    data.addSolution(ctx);
                                    sycl::group_barrier(item.get_group());
                                }
                            }
                            
                            // Backtrack and move to right child
                            if (top == -1) { // Tree fully explored
                                break;
                            }
                            Decision decision = data.restoreState(ctx, top--);
                            sycl::group_barrier(item.get_group());
                            if (item.get_local_id(0) == 0) decision.applyRight(ctx);
                        }
                    }
                });
            }).wait();

            data.getOutput(queue, params.nGroups, statistics, solution.data());
        }


        /**
         * @brief Get the search statistics
         */
        SearchStatistics getStatistics() const {
            return statistics;
        }

        /**
         * @brief Get the values of the output variables
         */
        std::span<int32_t const> getSolution() const {
            return solution;
        }


    private:
        using Decision = DecisionOf<Strategy>;

        Strategy strategy;
        uint32_t epsDepth;
        DeviceParams params;
        sycl::queue queue;
        DataStore<Decision, Limit> data;
        ConstraintStore<Constraints...> constraints;
        SearchStatistics statistics;
        std::vector<int32_t> solution;
    };

}

#endif