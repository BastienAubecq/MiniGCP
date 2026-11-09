#ifndef MINIGCP_ADD
#define MINIGCP_ADD

#include <concepts>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Ptr.hpp"

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
        V x;
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
    inline auto operator-(V x, C y) {
        return Add(x, -y);
    }


    /**
     * @brief Add two vectors
     * @tparam X Type of the first vector
     * @tparam Y Type of the second vector
     */
    template<ContextPtr X, ContextPtr Y>
    class AddVec {
    public:
        AddVec(X x, Y y = {}) : x(x), y(y) {}

        auto operator[](Context ctx, uint32_t i) const {
            return x[ctx, i] + y[ctx, i];
        }

        auto operator[](Allocator& allocator, uint32_t i) const {
            return x[allocator, i] + y[allocator, i];
        }

    private:
        [[no_unique_address]] X x;
        [[no_unique_address]] Y y;
    };

    template<ContextPtr X, ContextPtr Y>
    inline AddVec<X, Y> operator+(X x, Y y) {
        return AddVec(x, y);
    }

    template<ContextPtr X, ContextPtr Y>
    inline auto operator-(X x, Y y) {
        return AddVec(x, -y);
    }

}

#endif