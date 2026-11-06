#ifndef MINIGCP_ARRAY
#define MINIGCP_ARRAY

#include <cstdint>
#include <span>
#include <algorithm>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
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

}

#endif