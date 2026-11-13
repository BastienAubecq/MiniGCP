#ifndef MINIGCP_IS_OR
#define MINIGCP_IS_OR

#include "MiniGCP/Model.hpp"
#include "MiniGCP/state/Bool.hpp"
#include "MiniGCP/state/Const.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Neg.hpp"

namespace cp {

    /**
     * @brief Constraint : x | y <=> r
     * @tparam X Type of x
     * @tparam Y Type of y
     * @tparam R Type of r
     */
    template<IntVar X, IntVar Y, IntVar R>
    class IsOr {
    public:
        constexpr static bool parallel = false;

        IsOr(X x, Y y, R r) : x(x), y(y), r(r) {}

        DomainChange propagate(Context ctx) const {
            std::optional<int32_t> xv = x[ctx].value();
            std::optional<int32_t> yv = y[ctx].value();

            if (r[ctx].value() == 1) { // Or constraint
                if (xv == 0) return y[ctx].fix(1);
                if (yv == 0) return x[ctx].fix(1);
            }

            if (r[ctx].value() == 0) { // Nor constraint
                return x[ctx].fix(0) | y[ctx].fix(0);
            }

            // Determine r if possible
            if (xv == 1 || yv == 1) {
                return r[ctx].fix(1);
            }
            if (xv == 0 && yv == 0) {
                return r[ctx].fix(0);
            }
            return DomainChange::none;
        }

    protected:
        [[no_unique_address]] X x;
        [[no_unique_address]] Y y;
        [[no_unique_address]] R r;
    };

    template<Constraint... Constraints, IntVar X, IntVar Y>
    inline Bool isOr(Model<Constraints...>& model, X x, Y y) {
        Bool r(model);
        model.constraint(IsOr(x, y, r));
        return r;
    }

    
    template<IntVar X, IntVar Y>
    class Nor : public IsOr<X, Y, Const<Lit<0>>> {
    public:
        Nor(X x, Y y) : IsOr<X, Y, Const<Lit<0>>>(x, y, Const(Lit<0>())) {}
    };


    template<IntVar X, IntVar Y>
    class Or : public IsOr<X, Y, Const<Lit<1>>> {
    public:
        Or(X x, Y y) : IsOr<X, Y, Const<Lit<1>>>(x, y, Const(Lit<1>())) {}

        Nor<X, Y> operator~() const {
            return Nor(this->x, this->y);
        }
    };

    template<IntVar X, IntVar Y>
    inline Or<X, Y> operator|(X x, Y y) {
        return Or(x, y);
    }


    template<IntVar X, IntVar Y, IntVar R>
    class IsAnd : public IsOr<Not<X>, Not<Y>, Not<R>> {
    public:
        IsAnd(X x, Y y, R r) : IsOr<Not<X>, Not<Y>, Not<R>>(~x, ~y, ~r) {}
    };

    template<Constraint... Constraints, IntVar X, IntVar Y>
    inline Bool isAnd(Model<Constraints...>& model, X x, Y y) {
        Bool r(model);
        model.constraint(IsAnd(x, y, r));
        return r;
    }

    
    template<IntVar X, IntVar Y>
    class Nand : public IsAnd<X, Y, Const<Lit<0>>> {
    public:
        Nand(X x, Y y) : IsAnd<X, Y, Const<Lit<0>>>(x, y, Const(Lit<0>())) {}
    };


    template<IntVar X, IntVar Y>
    class And : public IsAnd<X, Y, Const<Lit<1>>> {
    public:
        And(X x, Y y) : IsAnd<X, Y, Const<Lit<1>>>(x, y, Const(Lit<1>())) {}

        Nand<X, Y> operator~() const {
            return Nand(this->x.x, this->y.x);
        }
    };

    template<IntVar X, IntVar Y>
    inline And<X, Y> operator&(X x, Y y) {
        return And(x, y);
    }

}

#endif