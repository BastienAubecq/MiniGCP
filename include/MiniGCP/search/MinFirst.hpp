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
     * @brief Search strategy :
     * Select the unfixed variable with the smallest minimum value.
     * Fix the variable to its minimum value in the left branch.
     * Remove that value in the right branch.
     * @tparam T Type of the variable pointer
     */
    template<IntVarPtr T>
    class MinFirst {
    public:
        MinFirst(T variables, uint32_t n) : variables(variables), n(n) {}

        std::optional<FixRemove<IntVarIn<T>>> split(Context ctx) const {
            int32_t smallestMin = std::numeric_limits<int32_t>::max();
            int32_t smallestIndex = -1;
            for (uint32_t i = ctx.item.get_local_id(0); i < n; i += ctx.item.get_local_range(0)) {
                if (!variables[ctx, i][ctx].value()) {
                    int32_t m = variables[ctx, i][ctx].min();
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
