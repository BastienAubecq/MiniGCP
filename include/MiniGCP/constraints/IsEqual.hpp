#ifndef MINIGCP_IS_EQUAL
#define MINIGCP_IS_EQUAL

#include "MiniGCP/Model.hpp"
#include "MiniGCP/state/Bool.hpp"
#include "MiniGCP/state/Const.hpp"
#include "MiniGCP/state/IntVar.hpp"

namespace cp {

    /**
     * @brief Constraint : x != y <=> r
     * @tparam X Type of x
     * @tparam Y Type of y
     * @tparam R Type of r
     */
    template<IntVar X, IntVar Y, IntVar R>
    class IsEqual {
    public:
        constexpr static bool parallel = false;

        IsEqual(X x, Y y, R r) : x(x), y(y), r(r) {}

        DomainChange propagate(Context ctx) const {
            int32_t xMin = x[ctx].min();
            int32_t xMax = x[ctx].max();
            int32_t yMin = y[ctx].min();
            int32_t yMax = y[ctx].max();

            if (r[ctx].value() == 1) { // Equal constraint
                DomainChange change = DomainChange::none;
                if (yMin > xMin) change |= x[ctx].removeBelow(yMin);
                if (xMin > yMin) change |= y[ctx].removeBelow(xMin);
                if (yMax < xMax) change |= x[ctx].removeAbove(yMax);
                if (xMax < yMax) change |= y[ctx].removeAbove(xMax);
                return change;
            }

            if (r[ctx].value() == 0) { // NotEqual constraint
                if (xMin == xMax) return y[ctx].remove(xMin);
                if (yMin == yMax) return x[ctx].remove(yMin);
                return DomainChange::none;
            }

            // Determine r if possible
            if (yMin > xMax || xMin > yMax) return r[ctx].fix(0);
            if (xMin == xMax && yMin == yMax && xMin == yMin) return r[ctx].fix(1);
            return DomainChange::none;
        }

    private:
        [[no_unique_address]] X x;
        [[no_unique_address]] Y y;
        [[no_unique_address]] R r;
    };

    template<Constraint... Constraints, IntVar X, IntVar Y>
    inline Bool isEqual(Model<Constraints...>& model, X x, Y y) {
        Bool r(model);
        model.constraint(IsEqual(x, y, r));
        return r;
    }


    template<IntVar X, IntVar Y>
    class Equal : public IsEqual<X, Y, Const<Lit<1>>> {
    public:
        Equal(X x, Y y) : IsEqual<X, Y, Const<Lit<1>>>(x, y, Const(Lit<1>())) {}
    };

    template<IntVar X, IntVar Y>
    inline Equal<X, Y> operator==(X x, Y y) {
        return Equal(x, y);
    }

    
    template<IntVar X, IntVar Y>
    class NotEqual : public IsEqual<X, Y, Const<Lit<0>>> {
    public:
        NotEqual(X x, Y y) : IsEqual<X, Y, Const<Lit<0>>>(x, y, Const(Lit<0>())) {}
    };

    template<IntVar X, IntVar Y>
    inline NotEqual<X, Y> operator!=(X x, Y y) {
        return NotEqual(x, y);
    }

}

#endif