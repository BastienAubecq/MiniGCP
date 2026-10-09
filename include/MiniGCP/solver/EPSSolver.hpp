#ifndef MINIGCP_EPS_SOLVER
#define MINIGCP_EPS_SOLVER

#include "MiniGCP/solver/Solver.hpp"

namespace cp {

    /**
     * @brief Embarrassingly parallel search solver
     */
    template<SearchStrategy Strategy, Constraint... Constraints>
    class EPSSolver : public Solver<Strategy, Constraints...> {
    public:
        EPSSolver(Model<Constraints...> model, Strategy strategy, DeviceParams params, uint32_t epsDepth) :
            Solver<Strategy, Constraints...>(model, strategy, params), epsDepth(epsDepth) {}

        void solve(SearchLimit auto limit) {
            uint32_t epsDepth = this->epsDepth;

            this->parallelFor(this->solver, this->params.nGroups, this->params.groupSize, [=](Context ctx, DeviceSolver<Strategy, Constraints...> solver) {
                bool hasSubproblem = !solver.firstSearch;
                solver.firstSearch = false;

                while (true) {
                    // DFS solve
                    if (hasSubproblem) {
                        while (true) {
                            if (!solver.continueSearch(ctx, limit)) return;
                            if (solver.exploreCurrent(ctx) || solver.moveLeft(ctx)) {
                                if (solver.moveRight(ctx)) break;
                            }
                        }
                    }

                    // Get next subproblem
                    uint32_t subproblem = ctx.item.get_local_id(0) == 0 ? solver.statistics->getSubproblem() : 0;
                    subproblem = sycl::group_broadcast(ctx.item.get_group(), subproblem, 0);
                    if (subproblem >= 1 << epsDepth) { // All subproblems solved
                        solver.statistics->completed = true;
                        return;
                    }
                    
                    // Dive until reaching the subproblem
                    solver.data.reset(ctx);
                    hasSubproblem = true;
                    for (uint32_t i = 0; i < epsDepth; i++) {
                        bool right = subproblem >> (epsDepth - i - 1) & 1; // Each bit of the subproblem index indicates a direction to take
                        sycl::group_barrier(ctx.item.get_group());
                        std::optional<DecisionOf<Strategy>> decision = solver.strategy.split(ctx);
                        if (!decision) { // Can't reach subproblem -> the first thread to reach this node explores it
                            hasSubproblem = subproblem >> (epsDepth - i - 1) == 0;
                            break;
                        }
                        if (ctx.item.get_local_id(0) == 0) {
                            if (right) decision->applyRight(ctx);
                            else decision->applyLeft(ctx);
                        }
                    }
                }
            });
        }


    private:
        uint32_t epsDepth;
    };

}

#endif