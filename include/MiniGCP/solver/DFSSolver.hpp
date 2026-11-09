#ifndef MINIGCP_DFS_SOLVER
#define MINIGCP_DFS_SOLVER

#include <stdexcept>
#include "MiniGCP/solver/Solver.hpp"

namespace cp {

    /**
     * @brief Depth-first search solver
     */
    template<Objective Objective, IntVarPtr Output, SearchStrategy Strategy, IsDeviceParams DeviceParams, Constraint... Constraints>
    class DFSSolver : public Solver<Objective, Output, Strategy, DeviceParams, Constraints...> {
    public:
        DFSSolver(Model<Constraints...> model, Objective objective, Output output, uint32_t outputSize, Strategy strategy, DeviceParams params) :
            Solver<Objective, Output, Strategy, DeviceParams, Constraints...>(model, objective, output, outputSize, strategy, params) {
            if (params.nGroups != 1) throw std::invalid_argument("DFSSolver uses a single group of threads");
        }

        void solve(SearchLimit auto limit) {
            this->parallelFor(this->solver, 1, [=](Context ctx, auto solver) {
                if (solver.firstSearch) solver.data.reset(ctx);
                while (solver.continueSearch(ctx, limit)) {
                    if (solver.exploreCurrent(ctx) || solver.moveLeft(ctx)) {
                        if (solver.moveRight(ctx)) {
                            solver.statistics->completed = true;
                            break;
                        }
                    }
                }
                //if (ctx.item.get_local_id(0) == 0) printf("Passes: %ld\n", solver.constraints.nPasses);
            }); 
        }
    };

}

#endif