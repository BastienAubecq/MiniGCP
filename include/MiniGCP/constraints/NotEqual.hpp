#ifndef MINIGCP_NOT_EQUAL
#define MINIGCP_NOT_EQUAL

#include "MiniGCP/state/IntVar.hpp"

namespace cp {

    /**
     * @brief Constraint : x != y
     * @tparam X Type of x
     * @tparam Y Type of y
     */
    template<IntVar X, IntVar Y>
    class NotEqual {
    public:
        constexpr static bool parallel = false;

        NotEqual(X x, Y y) : x(x), y(y) {}

        DomainChange propagate(Context ctx) const {
            std::optional<int32_t> v = x[ctx].value(); 
            if (v) return y[ctx].remove(*v);
            v = y[ctx].value();
            if (v) return x[ctx].remove(*v);
            return DomainChange::none;
        }

    private:
        X x;
        Y y;
    };

    template<IntVar X, IntVar Y>
    inline NotEqual<X, Y> operator!=(X x, Y y) {
        return NotEqual(x, y);
    }


    template<IntVarPtr X, IntVarPtr Y>
    class NotEqualVec {
    public:
        constexpr static bool parallel = true;

        NotEqualVec(X x, Y y, uint32_t n) : x(x), y(y), n(n) {}

        DomainChange propagate(Context ctx) const {
            DomainChange change = DomainChange::none;
            for (uint32_t i = ctx.item.get_local_id(0); i < n; i += ctx.item.get_local_range(0)) {
                change |= NotEqual(x[ctx, i], y[ctx, i]).propagate(ctx);
            }
            return groupChange(ctx.item.get_group(), change);
        }

    private:
        X x;
        Y y;
        uint32_t n;
    };

}

#endif