#ifndef MINIGCP_FIX_REMOVE
#define MINIGCP_FIX_REMOVE

#include <cstdint>
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"

namespace cp {

    /**
     * @brief Decision :
     * Fix the domain of a variable to a value in the left branch.
     * Remove that value in the right branch.
     * Leaf node if the variable is already fixed.
     */
    class FixRemove {
    public:
        FixRemove(IntVarIdx x, int32_t v) : x(x), v(v) {}

        void applyLeft(Context ctx) {
            //printf("Fix %d to %d\n", x.index, v);
            //ctx[x].print();
            ctx[x].fix(v);
            //ctx[x].print();
        }

        void applyRight(Context ctx) {
            //printf("Remove %d from %d\n", v, x.index);
            //ctx[x].print();
            ctx[x].remove(v);
            //ctx[x].print();
        }
    
    private:
        IntVarIdx x;
        int32_t v;
    };

}

#endif