#ifndef MINIGCP_ARRAY
#define MINIGCP_ARRAY

#include <cstdint>
#include <span>
#include <algorithm>
#include <concepts>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/Const.hpp"
#include "MiniGCP/state/Scope.hpp"

namespace cp {

    /**
     * @brief Data array
     * @tparam scope Scope of the array
     * @tparam T Type of elements in the array
     */
    template<typename T, Scope scope = Scope::global>
    class Array {
    public:
        /**
         * @brief Allocate a zero-initialized array
         * @param allocator CP model allocator
         * @param count Number of elements in the array
         */
        Array(Allocator& allocator, uint32_t count) :
            index(allocator.allocate<T, scope>(count)),
            count(count) {}

        /**
         * @brief Allocate an array
         * @param allocator CP model allocator
         * @param values Initial values in the array
         */
        Array(Allocator& allocator, std::span<T const> values) : Array(allocator, values.size()) {
            std::copy(values.begin(), values.end(), (*this)[allocator]);
        }

        T* operator[](Context ctx) const {
            return ctx.getData<T, scope>(index);
        }

        T* operator[](Allocator& allocator) const {
            return allocator.getData<T, scope>(index);
        }

        T& operator[](Context ctx, uint32_t i) const {
            return (*this)[ctx][i];
        }

        T& operator[](Allocator& allocator, uint32_t i) const {
            return (*this)[allocator][i];
        }

        uint32_t size() const {
            return count;
        }

    private:
        uint32_t index;
        uint32_t count;
    };


    /**
     * @brief Array that contains the same value multiple times
     * @tparam T Type of the value
     */
    template<typename T>
    class Same {
    public:
        Same(T x) : x(x) {}

        T operator[](Context, uint32_t) const {
            return x;
        }

        T operator[](Allocator&, uint32_t) const {
            return x;
        }

    private:
        [[no_unique_address]] T x;
    };


    /**
     * @brief Range of values starting at 0
     */
    class Range {
    public:
        int32_t operator[](Context, uint32_t i) const {
            return i;
        }

        int32_t operator[](Allocator&, uint32_t i) const {
            return i;
        }
    };


    /**
     * @brief Array in a context
     */
    template<typename T>
    concept ContextArray = requires(T const& arr, Context ctx, Allocator& allocator, uint32_t i) {
        arr[ctx, i];
        arr[allocator, i];
    };


    /**
     * @brief At least one parameter has a size
     */
    template<typename... T>
    concept Sized = (requires(T const& x) {
        { x.size() } -> std::same_as<uint32_t>;
    } || ...);

    /**
     * @brief Get the size of any argument that has a size
     */
    template<typename T, typename... Ts>
    uint32_t size(T const& x, Ts const&...) requires Sized<T> {
        return x.size();
    }

    template<typename T, typename... Ts>
    uint32_t size(T const&, Ts const&... xs) {
        return size(xs...);
    }

}

#endif