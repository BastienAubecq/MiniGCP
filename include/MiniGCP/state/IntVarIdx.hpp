#ifndef MINIGCP_INT_VAR_IDX
#define MINIGCP_INT_VAR_IDX

#include <cstdint>

namespace cp {

    /**
     * @brief Index of an integer variable in a state.
     * Variables are represented with an index instead of a direct reference because their location
     * depends on where the state is stored, which can change during execution.
     * To access the variable, first get a reference to it from a context or a model with 
     * their subscript operator, which adds the current state location to the index.
     */
    class IntVarIdx {
        friend class IntVarRef;
        friend class AtomicIntVarRef;

    public:
        IntVarIdx(uint32_t index, int32_t offset) :
            index(index), offset(offset) {}

        IntVarIdx& operator+=(int32_t y) {
            offset += y;
            return *this;
        }

        IntVarIdx& operator-=(int32_t y) {
            offset -= y;
            return *this;
        }

    public:
        uint32_t index;
        int32_t offset;
    };


    inline IntVarIdx operator+(IntVarIdx x, int32_t y) {
        return x += y;
    }

    inline IntVarIdx operator+(int32_t x, IntVarIdx y) {
        return y += x;
    }

    inline IntVarIdx operator-(IntVarIdx x, int32_t y) {
        return x -= y;
    }

}

#endif