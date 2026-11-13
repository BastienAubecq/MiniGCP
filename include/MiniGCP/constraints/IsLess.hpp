#ifndef MINIGCP_IS_LESS
#define MINIGCP_IS_LESS

#include "MiniGCP/Model.hpp"
#include "MiniGCP/state/Add.hpp"
#include "MiniGCP/state/Bool.hpp"
#include "MiniGCP/state/Const.hpp"
#include "MiniGCP/state/IntVar.hpp"

namespace cp {

    /**
     * @brief Constraint : x < y <=> r
     * @tparam X Type of x
     * @tparam Y Type of y
     * @tparam R Type of r
     */
    template<IntVar X, IntVar Y, IntVar R>
    class IsLess {
    public:
        constexpr static bool parallel = false;

        IsLess(X x, Y y, R r) : x(x), y(y), r(r) {}

        DomainChange propagate(Context ctx) const {
            if (r[ctx].value() == 1) { // Less constraint
                return x[ctx].removeAbove(y[ctx].max() - 1) | y[ctx].removeBelow(x[ctx].min() + 1);
            }

            if (r[ctx].value() == 0) { // GreaterOrEqual constraint
                return x[ctx].removeBelow(y[ctx].min()) | y[ctx].removeAbove(x[ctx].max());
            }

            // Determine r if possible
            if (x[ctx].max() < y[ctx].min()) return r[ctx].fix(1);
            if (x[ctx].min() >= y[ctx].max()) return r[ctx].fix(0);
            return DomainChange::none;
        }

    private:
        [[no_unique_address]] X x;
        [[no_unique_address]] Y y;
        [[no_unique_address]] R r;
    };

    template<Constraint... Constraints, IntVar X, IntVar Y>
    inline Bool isLess(Model<Constraints...>& model, X x, Y y) {
        Bool r(model);
        model.constraint(IsLess(x, y, r));
        return r;
    }


    template<IntVar X, IntVar Y>
    class Less : public IsLess<X, Y, Const<Lit<1>>> {
    public:
        Less(X x, Y y) : IsLess<X, Y, Const<Lit<1>>>(x, y, Const(Lit<1>())) {}
    };

    template<IntVar X, IntVar Y>
    inline Less<X, Y> operator<(X x, Y y) {
        return Less(x, y);
    }

    template<IntVar X, IntVar Y>
    inline Less<X, Y> operator>(X x, Y y) {
        return Less(y, x);
    }


    template<IntVar X, IntVar Y, IntVar R>
    class IsLessOrEqual : public IsLess<X, Add<Y, Lit<1>>, R> {
    public:
        IsLessOrEqual(X x, Y y, R r) : IsLess<X, Add<Y, Lit<1>>, R>(x, Add(y, Lit<1>()), r) {}
    };

    template<Constraint... Constraints, IntVar X, IntVar Y>
    inline Bool isLessOrEqual(Model<Constraints...>& model, X x, Y y) {
        Bool r(model);
        model.constraint(IsLessOrEqual(x, y, r));
        return r;
    }


    template<IntVar X, IntVar Y>
    class LessOrEqual : public IsLessOrEqual<X, Y, Const<Lit<1>>> {
    public:
        LessOrEqual(X x, Y y) : IsLessOrEqual<X, Y, Const<Lit<1>>>(x, y, Const(Lit<1>())) {}
    };

    template<IntVar X, IntVar Y>
    inline LessOrEqual<X, Y> operator<=(X x, Y y) {
        return LessOrEqual(x, y);
    }

    template<IntVar X, IntVar Y>
    inline LessOrEqual<X, Y> operator>=(X x, Y y) {
        return LessOrEqual(y, x);
    }

}

#endif