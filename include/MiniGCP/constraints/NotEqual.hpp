#ifndef MINIGCP_NOT_EQUAL
#define MINIGCP_NOT_EQUAL

#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVarBitsetIdx.hpp"

namespace cp {

    /**
     * @brief Constraint : x != y
     */
    class NotEqual {
    public:
        constexpr static bool parallel = false;

        NotEqual(IntVarBitsetIdx x, IntVarBitsetIdx y) :
            x(x), y(y) {}

        DomainChange propagate(Context ctx) const {
            std::optional<int32_t> v = ctx[x].value(); 
            if (v) return ctx[y].remove(*v);
            v = ctx[y].value();
            if (v) return ctx[x].remove(*v);
            return DomainChange::none;
        }

    private:
        IntVarBitsetIdx x;
        IntVarBitsetIdx y;
    };


    inline NotEqual operator!=(IntVarBitsetIdx x, IntVarBitsetIdx y) {
        return NotEqual(x, y);
    }

}

#endif