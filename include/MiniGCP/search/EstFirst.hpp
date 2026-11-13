#ifndef MINIGCP_EST_FIRST
#define MINIGCP_EST_FIRST

#include <cstdint>
#include <limits>
#include <optional>
#include "sycl/sycl.hpp"
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/utils.hpp"
#include "MiniGCP/search/FixRemove.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntVar.hpp"

namespace cp {

    /**
     * @brief Scheduling search strategy (Smallest Earliest Start Time / EST-first):
     * Select the unfixed task variable with the smallest minimum value (earliest start time).
     * Fix the variable to its minimum value in the left branch.
     * Remove that value (increase lower bound) in the right branch.
     * @tparam T Type of the variable pointer
     */
    template<IntVarPtr T>
    class EstFirst {
    public:
        EstFirst(T variables, uint32_t n) : variables(variables), n(n) {}

        std::optional<FixRemove<IntVarIn<T>>> split(Context ctx) const {
            int32_t smallestMin = std::numeric_limits<int32_t>::max();
            int32_t smallestIndex = -1;
            for (uint32_t i = ctx.item.get_local_id(0); i < n; i += ctx.item.get_local_range(0)) {
                auto& v = variables[ctx, i][ctx];
                if (v.size() > 1) {
                    int32_t m = v.min();
                    if (m < smallestMin) {
                        smallestMin = m;
                        smallestIndex = i;
                    }
                }
            }
            smallestIndex = groupArgmin(ctx, smallestMin, smallestIndex);
            if (smallestIndex == -1) return {};
            IntVarIn<T> smallestVar = variables[ctx, smallestIndex];
            return FixRemove(smallestVar, smallestVar[ctx].min());
        }

        uint32_t maxDepth() const {
            return n;
        }

    private:
        T variables;
        uint32_t n;
    };

}

#endif
