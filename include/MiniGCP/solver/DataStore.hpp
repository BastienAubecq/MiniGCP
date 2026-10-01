#ifndef MINIGCP_DATA_STORE
#define MINIGCP_DATA_STORE

#include <cstdint>
#include <span>
#include <array>
#include <algorithm>
#include <stdexcept>
#include "sycl/sycl.hpp"
#include "MiniGCP/search/Decision.hpp"
#include "MiniGCP/search/ProblemType.hpp"
#include "MiniGCP/search/SearchLimit.hpp"
#include "MiniGCP/search/SearchStatistics.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"

#include <iostream>

namespace cp {

    /**
     * @brief Device-side data
     * @tparam D Type of decisions
     */
    template<Decision Decision, SearchLimit Limit>
    class DataStore {
    public:
        DataStore(
            sycl::queue& queue, std::array<std::span<std::byte const>, 3> data,
            Limit limit, ProblemType type, IntVarIdx objective, ArrayIdx<Scope::global, IntVarIdx> output,
            uint32_t nGroups, uint32_t maxDepth
        ) : limit(limit), type(type), objective(objective), output(output) {
            globalSize = data[0].size();
            localSize = data[1].size();
            stateSize = data[2].size();
            stackSize = maxDepth;
            size_t allocLocalSize = localSize * (1 + nGroups);
            size_t allocStateSize = stateSize * (1 + (1 + stackSize) * nGroups);
            if (allocLocalSize > UINT32_MAX || allocStateSize > UINT32_MAX) {
                throw std::range_error("Allocated size exceeds 32-bit integer range");
            }
            globalData = globalSize == 0 ? nullptr : sycl::malloc_device<std::byte>(globalSize, queue);
            localData = localSize == 0 ? nullptr : sycl::malloc_device<std::byte>(allocLocalSize, queue);
            stateData = stateSize == 0 ? nullptr : sycl::malloc_device<std::byte>(allocStateSize, queue);
            outputData = sycl::malloc_device<int32_t>((output.size() + 1) * nGroups, queue);
            decisions = sycl::malloc_device<Decision>(stackSize * nGroups, queue);
            statistics = sycl::malloc_device<SearchStatistics>(1, queue);
            if (globalSize > 0) queue.memcpy(globalData, data[0].data(), globalSize).wait();
            if (localSize > 0) queue.memcpy(localData, data[1].data(), localSize).wait();
            if (stateSize > 0) queue.memcpy(stateData, data[2].data(), stateSize).wait();
            SearchStatistics stats(initialObjective());
            queue.memcpy(statistics, &stats, sizeof(SearchStatistics)).wait();
        }

        void destroy(sycl::queue& queue) const {
            sycl::free(globalData, queue);
            sycl::free(localData, queue);
            sycl::free(stateData, queue);
            sycl::free(outputData, queue);
            sycl::free(decisions, queue);
            sycl::free(statistics, queue);
        }

        
        /**
         * @brief Get solver output
         * @param queue SYCL queue
         * @param nGroups Number of work groups
         * @param statistics Search statistics
         * @param solution Values of the output variables
         */
        void getOutput(sycl::queue& queue, uint32_t nGroups, SearchStatistics& statistics, int32_t* solution) const {
            queue.memcpy(&statistics, this->statistics, sizeof(SearchStatistics)).wait();
            int32_t* outputData = new int32_t[(output.size() + 1) * nGroups];
            queue.memcpy(outputData, this->outputData, (output.size() + 1) * nGroups * sizeof(int32_t)).wait();
            int32_t obj = type == ProblemType::satisfaction ? 1 : statistics.objective;
            for (uint32_t i = 0; i < nGroups; i++) { // Take the solution of the first group that has the best objective
                if (outputData[(output.size() + 1) * i + output.size()] == obj) {
                    std::copy(outputData + (output.size() + 1) * i, outputData + (output.size() + 1) * i + output.size(), solution);
                    break;
                }
            }
            delete[] outputData;
        }


        /**
         * @brief Get the context of a thread
         * @param item Current thread
         */
        Context getContext(sycl::nd_item<1> item) const {
            return Context(item, {
                globalData,
                localData + localSize * (1 + item.get_group(0)),
                stateData + stateSize * (1 + item.get_group(0))
            });
        }


        /**
         * @brief Initialize local data for a group of threads
         * @param item Current thread
         */
        void initLocal(sycl::nd_item<1> item) const {
            copy(item, localData, localData + localSize * (1 + item.get_group(0)), localSize);
        }

        /**
         * @brief Initialize state data for a group of threads
         * @param item Current thread
         */
        void initState(sycl::nd_item<1> item) const {
            copy(item, stateData, stateData + stateSize * (1 + item.get_group(0)), stateSize);
        }

        /**
         * @brief Initialize output data for a group of threads
         * @param item Current thread
         */
        void initOutput(sycl::nd_item<1> item) const {
            if (item.get_local_id(0) == 0) {
                outputData[(output.size() + 1) * item.get_group(0) + output.size()] = initialObjective();
            }
        }

        /**
         * @brief Save state data and decision for a node to a stack frame
         * @param item Current thread
         * @param index Index of the frame
         * @return Pointer to the decision
         */
        Decision* saveState(sycl::nd_item<1> item, uint32_t index) const {
            copy(item,
                stateData + stateSize * (1 + item.get_group(0)),
                stateData + stateSize * (1 + item.get_group_range(0) * (1 + index) + item.get_group(0)),
                stateSize
            );
            return decisions + item.get_group_range(0) * index + item.get_group(0);
        }

        /**
         * @brief Restore state data and decision for a node from a stack frame
         * @param item Current thread
         * @param index Index of the frame
         * @return Pointer to the decision
         */
        Decision* restoreState(sycl::nd_item<1> item, uint32_t index) const {
            copy(item,
                stateData + stateSize * (1 + item.get_group_range(0) * (1 + index)),
                stateData + stateSize * (1 + item.get_group(0)),
                stateSize
            );
            return decisions + item.get_group_range(0) * index + item.get_group(0);
        }


        /**
         * @brief Whether the search should continue
         * @param item Current thread
         */
        bool continueSearch(sycl::nd_item<1> item) const {
            bool stop = false;
            if (item.get_local_id(0) == 0) { // Check only with first thread to make sure all threads get the same value
                if (limit(*statistics)) stop = true;
            }
            stop = sycl::group_broadcast(item.get_group(), stop, 0);
            return !stop;
        }

        /**
         * @brief Get global objective value to update current bound
         * @param ctx Current context
         */
        void getObjective(Context ctx) const {
            if (ctx.item.get_local_id(0) == 0) {
                if (type == ProblemType::minimum) ctx[objective].removeAbove(statistics->objective - 1);
                if (type == ProblemType::maximum) ctx[objective].removeBelow(statistics->objective + 1);
            }
        }

        /**
         * @brief Add a node to the statistics
         * @param thread Current thread
         */
        void addNode(sycl::nd_item<1> item) const {
            if (item.get_local_id(0) == 0) atomic(statistics->nNodes).fetch_add(1);
        }

        /**
         * @brief Add a failure to the statistics
         * @param thread Current thread
         */
        void addFailure(sycl::nd_item<1> item) const {
            if (item.get_local_id(0) == 0) atomic(statistics->nFailures).fetch_add(1);
        }

        /**
         * @brief Add a solution to the statistics and store solution
         * @param ctx Current context
         */
        void addSolution(Context ctx) const {
            if (ctx.item.get_local_id(0) == 0) {
                atomic(statistics->nSolutions).fetch_add(1);
                if (type == ProblemType::satisfaction) {
                    outputData[(output.size() + 1) * ctx.item.get_group(0) + output.size()] = 1;
                }
                else {
                    int32_t obj = ctx[objective].min();
                    if (type == ProblemType::minimum) atomic(statistics->objective).fetch_min(obj);
                    else atomic(statistics->objective).fetch_max(obj);
                    outputData[(output.size() + 1) * ctx.item.get_group(0) + output.size()] = obj;
                }
            }
            for (uint32_t i = ctx.item.get_local_id(0); i < output.size(); i += ctx.item.get_local_range(0)) {
                outputData[(output.size() + 1) * ctx.item.get_group(0) + i] = ctx[ctx[output][i]].min();
            }
        }


    private:
        using atomic = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>;

        uint32_t globalSize;
        uint32_t localSize;
        uint32_t stateSize;
        uint32_t stackSize;
        std::byte* globalData;
        std::byte* localData;
        std::byte* stateData;
        int32_t* outputData;
        Decision* decisions;
        SearchStatistics* statistics;
        Limit limit;
        ProblemType type;
        IntVarIdx objective;
        ArrayIdx<Scope::global, IntVarIdx> output;

        /**
         * @brief Copy some data from a source to a destination with a group of threads
         * @param item Current thread
         * @param src The source
         * @param dst The destination
         * @param size Number of bytes to copy
         */
        void copy(sycl::nd_item<1> item, std::byte const* src, std::byte* dst, uint32_t size) const {
            for (uint32_t i = item.get_local_id(0); i < size; i += item.get_local_range(0)) {
                dst[i] = src[i];
            }
        }

        /**
         * @brief Get the initial value of the objective depending on the problem type
         * @return The initial value
         */
        int32_t initialObjective() const {
            if (type == ProblemType::minimum) return INT32_MAX;
            if (type == ProblemType::maximum) return INT32_MIN;
            return 0;
        }
    };

}

#endif