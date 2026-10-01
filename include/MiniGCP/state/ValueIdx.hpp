#ifndef MINIGCP_VALUE_IDX
#define MINIGCP_VALUE_IDX

#include <cstdint>
#include <span>
#include "MiniGCP/state/Scope.hpp"

namespace cp {

    /**
     * @brief Index of an value.
     * Values are represented with an index instead of a reference because their location
     * depends on where the data is stored, which can change during execution.
     * To access the value, first get a reference to it from a context or a model with 
     * their subscript operator, which adds the current data location to the index.
     * @tparam scope Scope of the value
     * @tparam T Type of the value
     */
    template<Scope scope, typename T>
    class ValueIdx {
    public:
        ValueIdx(uint32_t index) :
            index(index) {}

        T& get(std::byte* data) const {
            return *reinterpret_cast<T*>(data + index);
        }

    private:
        uint32_t index;
    };

}

#endif