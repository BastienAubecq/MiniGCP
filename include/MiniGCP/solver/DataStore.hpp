#ifndef MINIGCP_DATA_STORE
#define MINIGCP_DATA_STORE

#include <cstdint>
#include <span>
#include <array>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include "sycl/sycl.hpp"
#include "MiniGCP/Model.hpp"
#include "MiniGCP/search/Decision.hpp"
#include "MiniGCP/search/SearchLimit.hpp"
#include "MiniGCP/search/SearchStatistics.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"

namespace cp {

    /**
     * @brief Device-side data
     * @tparam Decision Decision type
     * @tparam Limit Search limit type
     */
    template<Decision Decision>
    class DataStore {
    public:
        DataStore(
            sycl::queue& queue, std::array<std::span<std::byte const>, 3> data, uint32_t temporarySize,
            ArrayIdx<Scope::global, IntVarIdx> output, uint32_t nGroups, uint32_t maxDepth, int32_t objective
        ) : temporarySize(temporarySize), output(output) {
            globalSize = data[0].size();
            localSize = data[1].size();
            stateSize = data[2].size();
            if (localSize % sizeof(uint32_t) != 0) localSize += sizeof(uint32_t) - localSize % sizeof(uint32_t);
            if (stateSize % sizeof(uint32_t) != 0) stateSize += sizeof(uint32_t) - stateSize % sizeof(uint32_t);
            stackSize = maxDepth;
            size_t localMemorySize = queue.get_device().get_info<sycl::info::device::local_mem_size>();
            if (temporarySize > localMemorySize) throw std::range_error("Temporary size exceeds local memory size");
            localCurrentState =
                queue.get_device().get_info<sycl::info::device::local_mem_type>() == sycl::info::local_mem_type::local &&
                localMemorySize > temporarySize + stateSize;
            
            size_t allocLocalSize = localSize * (1 + nGroups);
            size_t allocStateSize = stateSize * (1 + (stackSize + 1) * nGroups);
            if (allocLocalSize > UINT32_MAX || allocStateSize > UINT32_MAX) {
                throw std::range_error("Allocated size exceeds 32-bit integer range");
            }

            int32_t* outputInit = new int32_t[(output.size() + 1) * nGroups];
            for (uint32_t i = 0; i < nGroups; i++) {
                outputInit[(output.size() + 1) * i + output.size()] = objective;
            }

            globalData = globalSize == 0 ? nullptr : sycl::malloc_device<std::byte>(globalSize, queue);
            localData = localSize == 0 ? nullptr : sycl::malloc_device<std::byte>(allocLocalSize, queue);
            stateData = stateSize == 0 ? nullptr : sycl::malloc_device<std::byte>(allocStateSize, queue);
            outputData = sycl::malloc_device<int32_t>((output.size() + 1) * nGroups, queue);
            decisions = sycl::malloc_device<Decision>(stackSize * nGroups, queue);
            tops = sycl::malloc_device<uint32_t>(nGroups, queue);
            if (globalSize > 0) queue.memcpy(globalData, data[0].data(), globalSize);
            if (localSize > 0) queue.memcpy(localData, data[1].data(), localSize);
            if (stateSize > 0) queue.memcpy(stateData, data[2].data(), stateSize);
            queue.memcpy(outputData, outputInit, sizeof(int32_t) * (output.size() + 1) * nGroups);
            queue.memset(tops, -1, sizeof(uint32_t) * nGroups);
            queue.wait();

            delete[] outputInit;
        }

        void destroy(sycl::queue& queue) const {
            sycl::free(globalData, queue);
            sycl::free(localData, queue);
            sycl::free(stateData, queue);
            sycl::free(outputData, queue);
            sycl::free(decisions, queue);
            sycl::free(tops, queue);
        }

        
        /**
         * @brief Get solver output
         * @param queue SYCL queue
         * @param nGroups Number of work groups
         * @param objective Best objective value
         * @param solution Values of the output variables
         */
        void getOutput(sycl::queue& queue, uint32_t nGroups, int32_t objective, int32_t* solution) const {
            int32_t* outputData = new int32_t[(output.size() + 1) * nGroups];
            queue.memcpy(outputData, this->outputData, (output.size() + 1) * nGroups * sizeof(int32_t)).wait();
            for (uint32_t i = 0; i < nGroups; i++) { // Take the solution of the first group that has the best objective
                if (outputData[(output.size() + 1) * i + output.size()] == objective) {
                    std::copy(outputData + (output.size() + 1) * i, outputData + (output.size() + 1) * i + output.size(), solution);
                    break;
                }
            }
            delete[] outputData;
        }


        /**
         * @brief Get local memory
         */
        std::array<sycl::local_accessor<std::byte>, 2> getLocalMemory(sycl::handler& handler) const {
            return {
                sycl::local_accessor<std::byte>(temporarySize, handler),
                sycl::local_accessor<std::byte>(localCurrentState ? stateSize : 0, handler)
            };
        }

        /**
         * @brief Get the context of a thread
         * @param item Current thread
         * @param error Error stream
         * @param local Local memory
         * @return Current context
         */
        Context getContext(sycl::nd_item<1> item, sycl::stream error, std::array<sycl::local_accessor<std::byte>, 2> local) const {
            return Context(item, error, {
                globalData,
                localData + localSize * (1 + item.get_group(0)),
                localCurrentState ?
                    std::get<1>(local).get_multi_ptr<sycl::access::decorated::no>().get() :
                    stateData + stateSize * (1 + item.get_group(0))
            }, std::get<0>(local).get_multi_ptr<sycl::access::decorated::no>().get());
        }


        /**
         * @brief Reset local and state data
         * @param ctx Current context
         */
        void reset(Context ctx) const {
            copy(ctx.item, localData, ctx.getData<Scope::local>(), localSize);
            copy(ctx.item, stateData, ctx.getData<Scope::state>(), stateSize);
        }

        /**
         * @brief Save current state in global memory
         * @param ctx Current context
         */
        void save(Context ctx) const {
            if (localCurrentState) {
                copy(ctx.item, ctx.getData<Scope::state>(), stateData + stateSize * (1 + ctx.item.get_group(0)), stateSize);
            }
        }

        /**
         * @brief Restore current state from global memory
         * @param ctx Current context
         */
        void restore(Context ctx) const {
            if (localCurrentState) {
                copy(ctx.item, stateData + stateSize * (1 + ctx.item.get_group(0)), ctx.getData<Scope::state>(), stateSize);
            }
        }

    
        /**
         * @brief Push state data and a decision onto the stack
         * @param ctx Current context
         * @param decision The decision
         * @return Whether the push succeeded
         */
        bool push(Context ctx, Decision decision) const {
            uint32_t index = tops[ctx.item.get_group(0)] + 1;
            if (index >= stackSize) return false;
            sycl::group_barrier(ctx.item.get_group());
            if (ctx.item.get_local_id(0) == 0) {
                //printf("Push %d\n", index);
                tops[ctx.item.get_group(0)] = index;
                decisions[ctx.item.get_group_range(0) * index + ctx.item.get_group(0)] = decision;
            }
            std::byte* frame = stateData + stateSize * (1 + ctx.item.get_group_range(0) * (1 + index) + ctx.item.get_group(0));
            copy(ctx.item, ctx.getData<Scope::state>(), frame, stateSize);
            return true;
        }

        /**
         * @brief Pop state data and a decision from the stack
         * @param ctx Current context
         * @return The decision if the pop succeeded
         */
        std::optional<Decision> pop(Context ctx) const {
            uint32_t index = tops[ctx.item.get_group(0)];
            if (index == -1) return {};
            sycl::group_barrier(ctx.item.get_group());
            if (ctx.item.get_local_id(0) == 0) {
                //printf("Pop %d\n", index);
                tops[ctx.item.get_group(0)] = index - 1;
            }
            std::byte* frame = stateData + stateSize * (1 + ctx.item.get_group_range(0) * (1 + index) + ctx.item.get_group(0));
            copy(ctx.item, frame, ctx.getData<Scope::state>(), stateSize);
            return decisions[ctx.item.get_group_range(0) * index + ctx.item.get_group(0)];
        }

        /**
         * @brief Set the current group's best solution
         * @param ctx Current context
         * @param objective Objective value
         */
        void solution(Context ctx, int32_t objective) const {
            for (uint32_t i = ctx.item.get_local_id(0); i < output.size(); i += ctx.item.get_local_range(0)) {
                outputData[(output.size() + 1) * ctx.item.get_group(0) + i] = ctx[ctx[output][i]].min();
            }
            if (ctx.item.get_local_id(0) == 0) {
                outputData[(output.size() + 1) * ctx.item.get_group(0) + output.size()] = objective;
            }
            sycl::group_barrier(ctx.item.get_group());
        }


        void print(Context ctx) const {
            if (ctx.item.get_local_id(0) == 0) {
                printf("--- State ---\n");
                for (uint32_t i = 0; i < output.size(); i++) {
                    ctx[ctx[output][i]].print();
                }
                printf("--- End State ---\n");
            }
        }


    private:
        uint32_t globalSize;
        uint32_t localSize;
        uint32_t stateSize;
        uint32_t stackSize;
        uint32_t temporarySize;
        bool localCurrentState; // Whether the current state is stored in local memory
        std::byte* globalData;
        std::byte* localData;
        std::byte* stateData; // Stacks of states
        int32_t* outputData; // Values of the output variables
        Decision* decisions; // Stack of decisions
        uint32_t* tops; // Top of the stacks
        ArrayIdx<Scope::global, IntVarIdx> output;

        /**
         * @brief Copy some data from a source to a destination with a group of threads
         * @param item Current thread
         * @param src The source
         * @param dst The destination
         * @param size Number of bytes to copy
         */
        void copy(sycl::nd_item<1> item, std::byte const* src, std::byte* dst, uint32_t size) const {
            uint32_t const* src32 = reinterpret_cast<uint32_t const*>(src);
            uint32_t* dst32 = reinterpret_cast<uint32_t*>(dst);
            size /= sizeof(uint32_t);
            for (uint32_t i = item.get_local_id(0); i < size; i += item.get_local_range(0)) {
                dst32[i] = src32[i];
            }
            sycl::group_barrier(item.get_group());
        }
    };

}

#endif