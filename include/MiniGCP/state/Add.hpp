#ifndef MINIGCP_ADD
#define MINIGCP_ADD

#include <concepts>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/Array.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVar.hpp"

namespace cp {

    template<IntVarRef V, std::convertible_to<int32_t> C>
    class AddRef {
    public:
        AddRef(V x, C y) : x(x), y(y) {}

        int32_t min() const {
            return x.min() + y;
        }

        int32_t max() const {
            return x.max() + y;
        }

        int32_t size() const {
            return x.size();
        }

        std::optional<int32_t> value() const {
            std::optional<int32_t> v = x.value();
            if (v) return *v + y;
            return {};
        }

        bool contains(int32_t v) const {
            return x.contains(v - y);
        }

        DomainChange remove(int32_t v) const {
            return x.remove(v - y);
        }

        DomainChange fix(int32_t v) const {
            return x.fix(v - y);
        }

        DomainChange removeBelow(int32_t v) const {
            return x.removeBelow(v - y);
        }

        DomainChange removeAbove(int32_t v) const {
            return x.removeAbove(v - y);
        }

    private:
        V x;
        C y;
    };


    /**
     * @brief Add a constant to an integer variable
     * @tparam V Underlying variable type
     * @tparam C Constant type
     */
    template<IntVar V, std::convertible_to<int32_t> C>
    class Add {
    public:
        Add(V x, C y) : x(x), y(y) {}

        auto operator[](Context ctx) const {
            return AddRef(x[ctx], y);
        }

        auto operator[](Allocator& allocator) const {
            return AddRef(x[allocator], y);
        }

    private:
        [[no_unique_address]] V x;
        [[no_unique_address]] C y;
    };

    template<IntVar V, std::convertible_to<int32_t> C>
    inline Add<V, C> operator+(V x, C y) {
        return Add(x, y);
    }

    template<IntVar V, std::convertible_to<int32_t> C>
    inline Add<V, C> operator+(C y, V x) {
        return Add(x, y);
    }

    template<IntVar V, std::convertible_to<int32_t> C>
    inline Add<V, C> operator-(V x, C y) {
        return Add(x, -y);
    }


    /**
     * @brief Add two arrays
     * @tparam X Type of the first array
     * @tparam Y Type of the second array
     */
    template<ContextArray X, ContextArray Y>
    class AddArr {
    public:
        AddArr(X x, Y y = {}) : x(x), y(y) {}

        auto operator[](Context ctx, uint32_t i) const {
            return x[ctx, i] + y[ctx, i];
        }

        auto operator[](Allocator& allocator, uint32_t i) const {
            return x[allocator, i] + y[allocator, i];
        }

        uint32_t size() const requires Sized<X, Y> {
            return cp::size(x, y);
        }

    private:
        [[no_unique_address]] X x;
        [[no_unique_address]] Y y;
    };

    template<ContextArray X, ContextArray Y>
    inline AddArr<X, Y> operator+(X x, Y y) {
        return AddArr(x, y);
    }

    template<ContextArray X, ContextArray Y>
    inline AddArr<X, Y> operator-(X x, Y y) {
        return AddArr(x, -y);
    }

}

#endif