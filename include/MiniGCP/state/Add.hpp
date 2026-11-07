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
    template<IntVar V, std::convertible_to<int32_t> C = int32_t>
    class Add {
    public:
        Add(V x, C y = {}) : x(x), y(y) {}

        auto operator[](Context ctx) const {
            return AddRef(x[ctx], y);
        }

        auto operator[](Allocator& allocator) const {
            return AddRef(x[allocator], y);
        }

        Add<V>& operator+=(int32_t z) requires std::same_as<C, int32_t> {
            y += z;
            return *this;
        }

        Add<V>& operator-=(int32_t z) requires std::same_as<C, int32_t> {
            return *this += -y;
        }

    private:
        V x;
        C y;
    };

    template<IntVar V>
    inline Add<V> operator+(Add<V> x, int32_t y) {
        return x += y;
    }

    template<IntVar V>
    inline Add<V> operator+(V x, int32_t y) {
        return Add(x, y);
    }

    template<IntVar V>
    inline auto operator+(int32_t x, V y) {
        return y + x;
    }

    template<IntVar V>
    inline auto operator-(V x, int32_t y) {
        return x + -y;
    }


    /**
     * @brief Add the same constant to each variable in an integer variable array
     * @tparam V Underlying variable array type
     * @tparam C Constant type
     */
    template<IntVarArray V, std::convertible_to<int32_t> C = int32_t>
    class Add1 {
    public:
        Add1(V x, C y = {}) : x(x), y(y) {}

        auto operator[](Context ctx, uint32_t i) const {
            return Add(x[ctx, i], y);
        }

        auto operator[](Allocator& allocator, uint32_t i) const {
            return Add(x[allocator, i], y);
        }

        uint32_t size() const {
            return x.size();
        }

        Add1<V>& operator+=(int32_t z) requires std::same_as<C, int32_t> {
            y += z;
            return *this;
        }

        Add1<V>& operator-=(int32_t z) requires std::same_as<C, int32_t> {
            return *this += -y;
        }

    private:
        V x;
        C y;
    };

    template<IntVarArray V>
    inline Add1<V> operator+(Add1<V> x, int32_t y) {
        return x += y;
    }

    template<IntVarArray V>
    inline Add1<V> operator+(V x, int32_t y) {
        return Add1(x, y);
    }

    template<IntVarArray V>
    inline auto operator+(int32_t x, V y) {
        return y + x;
    }

    template<IntVarArray V>
    inline auto operator-(V x, int32_t y) {
        return x + -y;
    }


    /**
     * @brief Add a different constant to each variable in an integer variable array
     * @tparam T Underlying variable array type
     */
    template<IntVarArray T>
    class AddN {
    public:
        AddN(T x, Array<int32_t> y) : x(x), y(y) {}

        auto operator[](Context ctx, uint32_t i) const {
            return Add(x[ctx, i], y[ctx][i]);
        }

        auto operator[](Allocator& allocator, uint32_t i) const {
            return Add(x[allocator, i], y[allocator][i]);
        }

        uint32_t size() const {
            return x.size();
        }

    private:
        T x;
        Array<int32_t> y;
    };

    template<IntVarArray T>
    inline AddN<T> operator+(T x, Array<int32_t> y) {
        return AddN(x, y);
    }

    template<IntVarArray T>
    inline auto operator+(Array<int32_t> x, T y) {
        return y + x;
    }

}

#endif