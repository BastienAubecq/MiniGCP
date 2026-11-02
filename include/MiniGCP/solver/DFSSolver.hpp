#ifndef MINIGCP_DFS_SOLVER
#define MINIGCP_DFS_SOLVER

#include <stdexcept>
#include "MiniGCP/solver/Solver.hpp"

namespace cp {

    /**
     * @brief Depth-first search solver
     */
    template<SearchStrategy Strategy, IsDeviceParams DeviceParams, Constraint... Constraints>
    class DFSSolver : public Solver<Strategy, DeviceParams, Constraints...> {
    public:
        DFSSolver(Model<Constraints...> model, Strategy strategy, DeviceParams params) :
            Solver<Strategy, DeviceParams, Constraints...>(model, strategy, params) {
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
            }); 
        }
    };

}

#endif