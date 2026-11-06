#ifndef MINIGCP_VALUE
#define MINIGCP_VALUE

#include <cstdint>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/Scope.hpp"

namespace cp {

    /**
     * @brief Data value
     * @tparam scope Scope of the value
     * @tparam T Type of the value
     */
    template<typename T, Scope scope = Scope::global>
    class Value {
    public:
        /**
         * @brief Allocate a value
         * @param allocator CP model allocator
         * @param value The value
         */
        Value(Allocator& allocator, int32_t value) {
            index = allocator.allocate<T, scope>(1);
            (*this)[allocator] = value;
        }

        T& operator[](Context ctx) const {
            return *ctx.getData<T, scope>(index);
        }

        T& operator[](Allocator& allocator) const {
            return *allocator.getData<T, scope>(index);
        }

    private:
        uint32_t index;
    };

}

#endif