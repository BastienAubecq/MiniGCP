#ifndef MINIGCP_MUL
#define MINIGCP_MUL

#include <concepts>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Ptr.hpp"

namespace cp {

    template<IntVarRef V>
    class MulRef {
    public:
        V x;
        int32_t y;

        int32_t min() const {
            if (y >= 0) return x.min() * y;
            return x.max() * y;
        }

        int32_t max() const {
            if (y >= 0) return x.max() * y;
            return x.min() * y;
        }

        int32_t size() const {
            if (y == 0) return 1;
            return x.size();
        }

        std::optional<int32_t> value() const {
            if (y == 0) return 0;
            std::optional<int32_t> v = x.value();
            if (v) return *v * y;
            return {};
        }

        bool contains(int32_t v) const {
            if (y == 0) return v == 0;
            return v % y == 0 && x.contains(v / y);
        }

        DomainChange remove(int32_t v) {
            if (y == 0) return v == 0 ? DomainChange::inconsistency : DomainChange::none;
            return v % y == 0 ? x.remove(v / y) : DomainChange::none;
        }

        DomainChange fix(int32_t v) {
            if (y == 0) return v == 0 ? DomainChange::none : DomainChange::inconsistency;
            return v % y == 0 ? x.fix(v / y) : DomainChange::inconsistency;
        }

        DomainChange removeBelow(int32_t v) {
            if (y == 0) return v > 0 ? DomainChange::inconsistency : DomainChange::none;
            int32_t q = v / y;
            if (q * y < v) ++q;
            return x.removeBelow(q);
        }

        DomainChange removeAbove(int32_t v) {
            if (y == 0) return v < 0 ? DomainChange::inconsistency : DomainChange::none;
            int32_t q = v / y;
            if (q * y > v) --q;
            return x.removeAbove(q);
        }
    };


    /**
     * @brief Multiply an integer variable by a constant
     * @tparam V Underlying variable type
     * @tparam C Constant type
     */
    template<IntVar V, std::convertible_to<int32_t> C>
    class Mul {
    public:
        V x;
        [[no_unique_address]] C y;

        auto operator[](Context ctx) const {
            return MulRef<decltype(x[ctx])>(x[ctx], y);
        }

        auto operator[](Allocator& allocator) const {
            return MulRef<decltype(x[allocator])>(x[allocator], y);
        }
    };

    template<IntVar V, std::convertible_to<int32_t> C>
    inline Mul<V, C> operator*(V x, C y) {
        return Mul(x, y);
    }

    template<IntVar V, std::convertible_to<int32_t> C>
    inline Mul<V, C> operator*(C y, V x) {
        return Mul(x, y);
    }


    /**
     * @brief Multiply two vectors
     * @tparam X Type of the first vector
     * @tparam Y Type of the second vector
     */
    template<ContextPtr X, ContextPtr Y>
    class MulVec {
    public:
        [[no_unique_address]] X x;
        [[no_unique_address]] Y y;

        auto operator[](Context ctx, uint32_t i) const {
            return x[ctx, i] * y[ctx, i];
        }

        auto operator[](Allocator& allocator, uint32_t i) const {
            return x[allocator, i] * y[allocator, i];
        }
    };

    template<ContextPtr X, ContextPtr Y>
    inline MulVec<X, Y> operator*(X x, Y y) {
        return MulVec(x, y);
    }

}

#endif