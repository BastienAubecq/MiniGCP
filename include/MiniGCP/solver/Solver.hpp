#ifndef MINIGCP_SOLVER
#define MINIGCP_SOLVER

#include <cstdint>
#include <optional>
#include <utility>
#include "sycl/sycl.hpp"
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/Constraint.hpp"
#include "MiniGCP/search/SearchLimit.hpp"
#include "MiniGCP/search/SearchStatistics.hpp"
#include "MiniGCP/search/SearchStrategy.hpp"
#include "MiniGCP/solver/ConstraintStore.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/solver/DataStore.hpp"
#include "MiniGCP/solver/DeviceParams.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Objective.hpp"

namespace cp {

    /**
     * @brief Device-side solver
     */
    template<Objective Objective, IntVarPtr Output, SearchStrategy Strategy, IsDeviceParams DeviceParams, Constraint... Constraints>
    class DeviceSolver {
    public:
        Objective objective;
        Strategy strategy;
        DataStore<DecisionOf<Strategy>, Output, DeviceParams::localLocal, DeviceParams::localState> data;
        ConstraintStore<Constraints...> constraints;
        SearchStatistics* statistics;
        bool firstSearch = true;

        DeviceSolver(sycl::queue& queue, Model<Constraints...> model, Objective objective, Output output, uint32_t outputSize, Strategy strategy, uint32_t nGroups) :
            objective(objective), strategy(strategy),
            data(queue, model.getData(), model.getTemporarySize(), output, outputSize, nGroups, strategy.maxDepth()),
            constraints(queue, model.getConstraints()) {
            statistics = sycl::malloc_device<SearchStatistics>(1, queue);
            SearchStatistics stats;
            queue.memcpy(statistics, &stats, sizeof(SearchStatistics)).wait();
        }

        void destroy(sycl::queue& queue) const {
            data.destroy(queue);
            constraints.destroy(queue);
            sycl::free(statistics, queue);
        }

        void invalidate() {
            data.invalidate();
            constraints.invalidate();
            statistics = nullptr;
        }


        /**
         * @brief Explore the current node
         * @param ctx Current context
         * @return Whether the current node is inconsistent
         */
        bool exploreCurrent(Context ctx) const {
            if (ctx.item.get_local_id(0) == 0) statistics->addNode();
            bool inconsistent = false;
            if (ctx.item.get_local_id(0) == 0) {
                DomainChange change = objective.update(ctx, statistics->objective);
                if ((change & DomainChange::inconsistency) == DomainChange::inconsistency) inconsistent = true;
            }
            inconsistent = sycl::group_broadcast(ctx.item.get_group(), inconsistent, 0);
            sycl::group_barrier(ctx.item.get_group());
            if (!inconsistent) inconsistent = constraints.fixpoint(ctx);
            if (inconsistent && ctx.item.get_local_id(0) == 0) statistics->addFailure();
            return inconsistent;
        }

        /**
         * @brief Move to the left child of the current node
         * @param ctx Current context
         * @return Whether the current node is a solution
         */
        bool moveLeft(Context ctx) const {
            //if (ctx.item.get_local_id(0) == 0) printf("Move left\n");
            std::optional<DecisionOf<Strategy>> decision = strategy.split(ctx);
            if (decision) {
                if (!data.push(ctx, *decision)) {
                    ctx.error << "Stack overflow. This probably means that the search strategy specified an incorrect maximum depth.\n";
                }
                if (ctx.item.get_local_id(0) == 0) decision->applyLeft(ctx);
                sycl::group_barrier(ctx.item.get_group());
            }
            else {
                int32_t obj = objective.value(ctx);
                data.solution(ctx, obj);
                if (ctx.item.get_local_id(0) == 0) {
                    //printf("Objective: %d\n", obj);
                    statistics->addSolution();
                    statistics->updateObjective(obj);
                }
            }
            return !decision;
        }

        /**
         * @brief Backtrack and move to the right child of the previous node
         * @param ctx Current context
         * @return Whether the search is completed
         */
        bool moveRight(Context ctx) const {
            //if (ctx.item.get_local_id(0) == 0) printf("Move right\n");
            std::optional<DecisionOf<Strategy>> decision = data.pop(ctx);
            if (decision) {
                if (ctx.item.get_local_id(0) == 0) decision->applyRight(ctx);
                sycl::group_barrier(ctx.item.get_group());
            }
            return !decision;
        }

        /**
         * @brief Evaluate a search limit
         * @param ctx Current context
         * @param limit The limit
         * @return Whether the search should continue
         */
        bool continueSearch(Context ctx, SearchLimit auto limit) const {
            // Check only with first thread to make sure all threads get the same value
            bool stop = false;
            if (ctx.item.get_local_id(0) == 0 && limit(*statistics)) stop = true;
            return !sycl::group_broadcast(ctx.item.get_group(), stop, 0);
        }
    };



    /**
     * @brief CP solver base class
     */
    template<Objective Objective, IntVarPtr Output, SearchStrategy Strategy, IsDeviceParams DeviceParams, Constraint... Constraints>
    class Solver {
    public:
        Solver(Model<Constraints...> model, Objective objective, Output output, uint32_t outputSize, Strategy strategy, DeviceParams params) :
            queue(params.selector, sycl::property_list()),
            params(params),
            solver(queue, model, objective, output, outputSize, strategy, params.nGroups),
            solution(outputSize) {}

        ~Solver() {
            solver.destroy(queue);
        }

        Solver(Solver const& other) = delete;

        Solver(Solver&& other) :
            queue(std::move(other.queue)),
            params(std::move(other.params)),
            solver(other.solver),
            statistics(std::move(other.statistics)),
            solution(std::move(other.solution)) {
            other.solver.invalidate();
        }

        Solver& operator=(Solver other) {
            std::swap(queue, other.queue);
            std::swap(params, other.params);
            std::swap(solver, other.solver);
            std::swap(statistics, other.statistics);
            std::swap(solution, other.solution);
            return *this;
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


    protected:
        sycl::queue queue;
        DeviceParams params;
        DeviceSolver<Objective, Output, Strategy, DeviceParams, Constraints...> solver;
        SearchStatistics statistics;
        std::vector<int32_t> solution;

        /**
         * @brief Execute a kernel on the device
         * @param solver Device solver
         * @param nGroups Number of work groups
         * @param kernel The kernel
         */
        template<typename Kernel>
        void parallelFor(DeviceSolver<Objective, Output, Strategy, DeviceParams, Constraints...>& solver, uint32_t nGroups, Kernel kernel) {
            queue.submit([=](sycl::handler& handler) {
                auto local = solver.data.getLocalMemory(handler);
                sycl::stream error(8192, 1024, handler);

                handler.parallel_for(sycl::nd_range<1>(nGroups * DeviceParams::groupSize, DeviceParams::groupSize), [=](sycl::nd_item<1> item) {
                    if (solver.statistics->completed) return;

                    Context ctx = solver.data.getContext(item, error, local);
                    if (!solver.firstSearch) solver.data.restore(ctx);
                    kernel(ctx, solver);
                    solver.data.save(ctx);
                });
            }).wait();

            solver.firstSearch = false;
            queue.memcpy(&statistics, solver.statistics, sizeof(SearchStatistics)).wait();
            solver.data.getOutput(queue, nGroups, statistics.objective, solution.data());
        }
    };

}

#endif