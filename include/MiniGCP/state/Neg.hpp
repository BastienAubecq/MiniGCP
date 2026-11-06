#ifndef MINIGCP_NEG
#define MINIGCP_NEG

#include <functional>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVar.hpp"

namespace cp {

    template<IntVarIntervalRef T>
    class NegRef {
    public:
        explicit NegRef(T x) : x(x) {}

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

        bool contains(int32_t v) const requires IntVarRef<T> {
            return x.contains(-v);
        }

        DomainChange remove(int32_t v) const requires IntVarRef<T> {
            return x.remove(-v);
        }

        DomainChange fix(int32_t v) const {
            return x.fix(-v);
        }

        DomainChange removeBelow(int32_t v) const {
            return x.removeAbove(-v);
        }

        DomainChange removeAbove(int32_t v) const {
            return x.removeBelow(-v);
        }

    private:
        T x;
    };


    /**
     * @brief Negate an integer variable
     * @tparam T Underlying variable type
     */
    template<IntVarInterval T>
    class Neg {
    public:
        explicit Neg(T x) : x(x) {}

        auto operator[](Context ctx) const {
            return NegRef(x[ctx]);
        }

        auto operator[](Allocator& allocator) const {
            return NegRef(x[allocator]);
        }

        T operator-() {
            return x;
        }

    private:
        T x;
    };

    template<IntVarInterval T>
    inline Neg<T> operator-(T x) {
        return Neg(x);
    }


    /**
     * @brief Negate an integer variable array
     * @tparam T Underlying variable array type
     */
    template<IntVarIntervalArray T>
    class Neg1 {
    public:
        explicit Neg1(T x) : x(x) {}

        auto operator[](Context ctx, uint32_t i) const {
            return Neg(x[ctx, i]);
        }

        auto operator[](Allocator& allocator, uint32_t i) const {
            return Neg(x[allocator, i]);
        }

        uint32_t size() const {
            return x.size();
        }

        T operator-() {
            return x;
        }

    private:
        T x;
    };

    template<IntVarIntervalArray T>
    inline Neg1<T> operator-(T x) {
        return Neg1(x);
    }

}

#endif