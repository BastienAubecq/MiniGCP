#ifndef MINIGCP_NOT_EQUAL
#define MINIGCP_NOT_EQUAL

#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"

namespace cp {

    /**
     * @brief Constraint : x != y
     */
    class NotEqual {
    public:
        constexpr static bool parallel = false;

        NotEqual(IntVarIdx x, IntVarIdx y) :
            x(x), y(y) {}

        DomainChange propagate(Context ctx) const {
            if (ctx[x].fixed()) {
                return ctx[y].remove(ctx[x].min());
            }
            if (ctx[y].fixed()) {
                return ctx[x].remove(ctx[y].min());
            }
            return DomainChange::none;
        }

    private:
        IntVarIdx x;
        IntVarIdx y;
    };


    inline NotEqual operator!=(IntVarIdx x, IntVarIdx y) {
        return NotEqual(x, y);
    }

}

#endif