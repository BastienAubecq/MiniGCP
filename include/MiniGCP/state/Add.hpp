#ifndef MINIGCP_ADD
#define MINIGCP_ADD

#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/Array.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVar.hpp"

namespace cp {

    template<IntVarRef T>
    class AddRef {
    public:
        AddRef(T x, int32_t y) : x(x), y(y) {}

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
        T x;
        int32_t y;
    };


    /**
     * @brief Add a constant to an integer variable
     * @tparam T Underlying variable type
     */
    template<IntVar T>
    class Add {
    public:
        Add(T x, int32_t y) : x(x), y(y) {}

        auto operator[](Context ctx) const {
            return AddRef(x[ctx], y);
        }

        auto operator[](Allocator& allocator) const {
            return AddRef(x[allocator], y);
        }

        Add<T>& operator+=(int32_t z) {
            y += z;
            return *this;
        }

        Add<T>& operator-=(int32_t z) {
            return *this += -y;
        }

    private:
        T x;
        int32_t y;
    };

    template<IntVar T>
    inline Add<T> operator+(Add<T> x, int32_t y) {
        return x += y;
    }

    template<IntVar T>
    inline Add<T> operator+(T x, int32_t y) {
        return Add<T>(x, y);
    }

    template<IntVar T>
    inline auto operator+(int32_t x, T y) {
        return y + x;
    }

    template<IntVar T>
    inline auto operator-(T x, int32_t y) {
        return x + -y;
    }


    /**
     * @brief Add the same constant to each variable in an integer variable array
     * @tparam T Underlying variable array type
     */
    template<IntVarArray T>
    class Add1 {
    public:
        Add1(T x, int32_t y) : x(x), y(y) {}

        auto operator[](Context ctx, uint32_t i) const {
            return Add(x[ctx, i], y);
        }

        auto operator[](Allocator& allocator, uint32_t i) const {
            return Add(x[allocator, i], y);
        }

        uint32_t size() const {
            return x.size();
        }

        Add1<T>& operator+=(int32_t z) {
            y += z;
            return *this;
        }

        Add1<T>& operator-=(int32_t z) {
            return *this += -y;
        }

    private:
        T x;
        int32_t y;
    };

    template<IntVarArray T>
    inline Add1<T> operator+(Add1<T> x, int32_t y) {
        return x += y;
    }

    template<IntVarArray T>
    inline Add1<T> operator+(T x, int32_t y) {
        return Add1(x, y);
    }

    template<IntVarArray T>
    inline auto operator+(int32_t x, T y) {
        return y + x;
    }

    template<IntVarArray T>
    inline auto operator-(T x, int32_t y) {
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