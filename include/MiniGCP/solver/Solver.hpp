#ifndef MINIGCP_SOLVER
#define MINIGCP_SOLVER

#include <cstdint>
#include <optional>
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
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"

namespace cp {

    /**
     * @brief Device-side solver
     */
    template<SearchStrategy Strategy, Constraint... Constraints>
    class DeviceSolver {
    public:
        ProblemType type;
        IntVarIdx objective;
        Strategy strategy;
        DataStore<DecisionOf<Strategy>> data;
        ConstraintStore<Constraints...> constraints;
        SearchStatistics* statistics;
        bool firstSearch = true;

        DeviceSolver(sycl::queue& queue, Model<Constraints...> model, Strategy strategy, uint32_t nGroups) :
            type(model.type), objective(model.objective), strategy(strategy),
            data(queue, model.getData(), model.output, nGroups, strategy.maxDepth(), SearchStatistics(model.type).objective),
            constraints(queue, model.getConstraints()) {
            statistics = sycl::malloc_device<SearchStatistics>(1, queue);
            SearchStatistics stats(model.type);
            queue.memcpy(statistics, &stats, sizeof(SearchStatistics)).wait();
        }

        void destroy(sycl::queue& queue) const {
            data.destroy(queue);
            constraints.destroy(queue);
            sycl::free(statistics, queue);
        }


        /**
         * @brief Explore the current node
         * @param ctx Current context
         * @return Whether the current node is inconsistent
         */
        bool exploreCurrent(Context ctx) const {
            if (ctx.item.get_local_id(0) == 0) {
                statistics->addNode();
                if (type == ProblemType::minimization) ctx[objective].removeAbove(statistics->objective - 1);
                if (type == ProblemType::maximization) ctx[objective].removeBelow(statistics->objective + 1);
            }
            sycl::group_barrier(ctx.item.get_group());
            bool inconsistent = constraints.fixpoint(ctx);
            if (inconsistent && ctx.item.get_local_id(0) == 0) statistics->addFailure();
            return inconsistent;
        }

        /**
         * @brief Move to the left child of the current node
         * @param ctx Current context
         * @return Whether the current node is a solution
         */
        bool moveLeft(Context ctx) const {
            //data.print(ctx);
            std::optional<DecisionOf<Strategy>> decision = strategy.split(ctx);
            if (decision) {
                if (!data.push(ctx, *decision)) {
                    ctx.error << "Stack overflow. This probably means that the search strategy specified an incorrect maximum depth.\n";
                }
                if (ctx.item.get_local_id(0) == 0) decision->applyLeft(ctx);
                sycl::group_barrier(ctx.item.get_group());
            }
            else {
                int32_t obj = type == ProblemType::satisfaction ? 1 : ctx[objective].min();
                data.solution(ctx, obj);
                if (ctx.item.get_local_id(0) == 0) {
                    statistics->addSolution();
                    statistics->updateObjective(type, obj);
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
            std::optional<DecisionOf<Strategy>> decision = data.pop(ctx);
            //data.print(ctx);
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
    template<SearchStrategy Strategy, Constraint... Constraints>
    class Solver {
    public:
        Solver(Model<Constraints...> model, Strategy strategy, DeviceParams params) :
            queue(params.selector, sycl::property_list()),
            params(params),
            solver(queue, model, strategy, params.nGroups),
            statistics(model.type),
            solution(model.output.size()) {}

        ~Solver() {
            solver.destroy(queue);
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
        DeviceSolver<Strategy, Constraints...> solver;
        SearchStatistics statistics;
        std::vector<int32_t> solution;

        /**
         * @brief Execute a kernel on the device
         * @param solver Device solver
         * @param nGroups Number of work groups
         * @param groupSize Size of the work groups
         * @param kernel The kernel
         */
        template<typename Kernel>
        void parallelFor(DeviceSolver<Strategy, Constraints...>& solver, uint32_t nGroups, uint32_t groupSize, Kernel kernel) {
            queue.submit([=](sycl::handler& handler) {
                sycl::local_accessor<std::byte> shared(solver.data.getSharedSize(), handler);
                sycl::stream error(8192, 1024, handler);

                handler.parallel_for(sycl::nd_range<1>(nGroups * groupSize, groupSize), [=](sycl::nd_item<1> item) {
                    if (solver.statistics->completed) return;

                    Context ctx = solver.data.getContext(item, error, shared);
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