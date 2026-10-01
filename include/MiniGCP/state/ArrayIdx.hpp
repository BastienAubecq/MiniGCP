#ifndef MINIGCP_ARRAY_IDX
#define MINIGCP_ARRAY_IDX

#include <cstdint>
#include "MiniGCP/state/Scope.hpp"

namespace cp {

    /**
     * @brief Index of an array.
     * Arrays are represented with an index instead of a pointer because their location
     * depends on where the data is stored, which can change during execution.
     * To access the array, first get a reference to it from a context or a model with 
     * their subscript operator, which adds the current data location to the index.
     * @tparam scope Scope of the array
     * @tparam T Type of elements in the array
     */
    template<Scope scope, typename T>
    class ArrayIdx {
    public:
        ArrayIdx(uint32_t index, uint32_t count) :
            index(index), count(count) {}

        T* get(std::byte* data) const {
            return reinterpret_cast<T*>(data + index);
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