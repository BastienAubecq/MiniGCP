#ifndef MINIGCP_NEG
#define MINIGCP_NEG

#include <functional>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/Array.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVar.hpp"

namespace cp {

    template<IntVarRef T>
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

        bool contains(int32_t v) const {
            return x.contains(-v);
        }

        DomainChange remove(int32_t v) const {
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
    template<IntVar T>
    class Neg {
    public:
        explicit Neg(T x) : x(x) {}

        auto operator[](Context ctx) const {
            return NegRef(x[ctx]);
        }

        auto operator[](Allocator& allocator) const {
            return NegRef(x[allocator]);
        }

    private:
        [[no_unique_address]] T x;
    };

    template<IntVar T>
    inline Neg<T> operator-(T x) {
        return Neg(x);
    }


    /**
     * @brief Negate an array
     * @tparam T Type of the array
     */
    template<ContextArray T>
    class NegArr {
    public:
        explicit NegArr(T x) : x(x) {}

        auto operator[](Context ctx, uint32_t i) const {
            return -x[ctx, i];
        }

        auto operator[](Allocator& allocator, uint32_t i) const {
            return -x[allocator, i];
        }

        uint32_t size() const requires Sized<T> {
            return x.size();
        }

    private:
        [[no_unique_address]] T x;
    };

    template<ContextArray T>
    inline NegArr<T> operator-(T x) {
        return NegArr(x);
    }

}

#endif