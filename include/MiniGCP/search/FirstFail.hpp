#ifndef MINIGCP_FIRST_FAIL
#define MINIGCP_FIRST_FAIL

#include <cstdint>
#include <limits>
#include <span>
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
     * Select the variable with the smallest non-empty domain.
     * Fix the variable to its minimum value in the left branch.
     * Remove that value in the right branch.
     * @tparam T Type of the variable array
     */
    template<IntVarArray T>
    class FirstFail {
    public:
        FirstFail(T variables) : variables(variables) {}

        std::optional<FixRemove<IntVarIn<T>>> split(Context ctx) const {
            int32_t smallestSize = std::numeric_limits<int32_t>::max();
            int32_t smallestIndex = -1;
            for (uint32_t i = ctx.item.get_local_id(0); i < variables.size(); i += ctx.item.get_local_range(0)) {
                int32_t s = variables[ctx, i][ctx].size();
                if (s > 1 && s < smallestSize) {
                    smallestSize = s;
                    smallestIndex = i;
                }
            }
            smallestIndex = groupArgmin(ctx, smallestSize, smallestIndex);
            if (smallestIndex == -1) return {};
            IntVarIn<T> smallestVar = variables[ctx, smallestIndex];
            //if (ctx.item.get_local_id(0) == 0) printf("Split %d-%d\n", smallestVar.index, ctx[smallestVar].min());
            return FixRemove(smallestVar, smallestVar[ctx].min());
        }

        uint32_t maxDepth() const {
            return variables.size();
        }
        
    private:
        T variables;
    };

}

#endif