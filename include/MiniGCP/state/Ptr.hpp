#ifndef MINIGCP_ARRAY
#define MINIGCP_ARRAY

#include <cstdint>
#include <span>
#include <algorithm>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/Const.hpp"
#include "MiniGCP/state/Scope.hpp"

namespace cp {

    /**
     * @brief Data pointer
     * @tparam scope Scope of the pointer
     * @tparam T Type of pointed elements
     */
    template<typename T, Scope scope = Scope::global>
    class Ptr {
    public:
        /**
         * @brief Allocate a zero-initialized array
         * @param allocator CP model allocator
         * @param count Number of elements in the array
         */
        Ptr(Allocator& allocator, uint32_t count) :
            index(allocator.allocate<T, scope>(count)) {}

        /**
         * @brief Allocate an array
         * @param allocator CP model allocator
         * @param values Initial values in the array
         */
        Ptr(Allocator& allocator, std::span<T const> values) : Ptr(allocator, values.size()) {
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

    private:
        uint32_t index;
    };


    /**
     * @brief Data reference
     * @tparam scope Scope of the reference
     * @tparam T Type of the value
     */
    template<typename T, Scope scope>
    class Ref : private Ptr<T, scope> {
    public:
        /**
         * @brief Allocate a value
         * @param allocator CP model allocator
         * @param value The value
         */
        Ref(Allocator& allocator, T value) :
            Ptr<T, scope>(allocator, std::span(&value, 1)) {}

        T& operator[](Context ctx) const {
            return *this->Ptr<T, scope>::operator[](ctx);
        }

        T& operator[](Allocator& allocator) const {
            return *this->Ptr<T, scope>::operator[](allocator);
        }
    };


    /**
     * @brief Same value multiple times
     * @tparam T Type of the value
     */
    template<typename T>
    class Same {
    public:
        explicit Same(T x) : x(x) {}

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
     * @brief Pointer in a context
     */
    template<typename T>
    concept ContextPtr = sycl::is_device_copyable_v<T> && requires(T const& ptr, Context ctx, Allocator& allocator, uint32_t i) {
        ptr[ctx, i];
        ptr[allocator, i];
    };

}

#endif