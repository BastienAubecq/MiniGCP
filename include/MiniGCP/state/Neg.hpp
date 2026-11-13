#ifndef MINIGCP_NEG
#define MINIGCP_NEG

#include <functional>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/Add.hpp"
#include "MiniGCP/state/Const.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Ptr.hpp"

namespace cp {

    template<IntVarRef T>
    class NegRef {
    public:
        T x;

        int32_t min() const {
            return -x.max();
        }

        int32_t max() const {
            return -x.min();
        }

        int32_t size() const {
            return x.size();
        }

        std::optional<int32_t> value() const {
            return x.value().transform(std::negate<int32_t>());
        }

        bool contains(int32_t v) const {
            return x.contains(-v);
        }

        DomainChange remove(int32_t v) {
            return x.remove(-v);
        }

        DomainChange fix(int32_t v) {
            return x.fix(-v);
        }

        DomainChange removeBelow(int32_t v) {
            return x.removeAbove(-v);
        }

        DomainChange removeAbove(int32_t v) {
            return x.removeBelow(-v);
        }
    };


    /**
     * @brief Negate an integer variable
     * @tparam T Underlying variable type
     */
    template<IntVar T>
    class Neg {
    public:
        T x;

        auto operator[](Context ctx) const {
            return NegRef<decltype(x[ctx])>(x[ctx]);
        }

        auto operator[](Allocator& allocator) const {
            return NegRef<decltype(x[allocator])>(x[allocator]);
        }
    };

    template<IntVar T>
    inline Neg<T> operator-(T x) {
        return Neg(x);
    }


    /**
     * @brief Negate a vector
     * @tparam T Type of the vector
     */
    template<ContextPtr T>
    class NegVec {
    public:
        [[no_unique_address]] T x;

        auto operator[](Context ctx, uint32_t i) const {
            return -x[ctx, i];
        }

        auto operator[](Allocator& allocator, uint32_t i) const {
            return -x[allocator, i];
        }
    };

    template<ContextPtr T>
    inline NegVec<T> operator-(T x) {
        return NegVec(x);
    }


    template<IntVar T>
    class Not : public Add<Neg<T>, Lit<1>> {
    public:
        Not(T x) : Add<Neg<T>, Lit<1>>(Neg(x), Lit<1>()) {}
    };

    template<IntVar T>
    inline Not<T> operator~(T x) {
        return Not(x);
    }


    template<ContextPtr T>
    class NotVec : public AddVec<NegVec<T>, Same<Lit<1>>> {
    public:
        NotVec(T x) : AddVec<NegVec<T>, Same<Lit<1>>>(NegVec(x), Same(Lit<1>())) {}
    };

    template<ContextPtr T>
    inline NotVec<T> operator~(T x) {
        return NotVec(x);
    }

}

#endif