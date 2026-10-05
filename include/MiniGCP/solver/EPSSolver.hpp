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
        EPSSolver(
            Model<Constraints...> model, ProblemType type, Strategy strategy, Limit limit,
            DeviceParams params, ArrayIdx<Scope::global, IntVarIdx> output, IntVarIdx objective = IntVarIdx(0, 0)
        ) :
            strategy(strategy), params(params),
            queue(params.selector, { sycl::property::queue::in_order() }),
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
            DeviceParams params = this->params;
            DataStore<Decision, Limit> data = this->data;
            ConstraintStore<Constraints...> constraints = this->constraints;

            // Launch kernel
            queue.submit([=](sycl::handler& handler) {
                handler.parallel_for(sycl::nd_range<1>(params.nGroups * params.groupSize, params.groupSize), [=](sycl::nd_item<1> item) {
                    Context ctx = data.getContext(item);
                    data.initLocal(item);
                    data.initState(item);
                    data.initOutput(item);

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
                                    printf("Stack overflow\n");
                                    break;
                                }
                                data.saveState(item, top, *decision);
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
                        Decision decision = data.restoreState(item, top--);
                        sycl::group_barrier(item.get_group());
                        if (item.get_local_id(0) == 0) decision.applyRight(ctx);
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
        DeviceParams params;
        sycl::queue queue;
        DataStore<Decision, Limit> data;
        ConstraintStore<Constraints...> constraints;
        SearchStatistics statistics;
        std::vector<int32_t> solution;
    };

}

#endif