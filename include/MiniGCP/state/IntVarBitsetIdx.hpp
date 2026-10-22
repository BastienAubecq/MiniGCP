#ifndef MINIGCP_INT_VAR_BITSET_IDX
#define MINIGCP_INT_VAR_BITSET_IDX

#include <cstdint>

namespace cp {

    /**
     * @brief Index of an integer variable represented with a bitset in a state.
     * Variables are represented with an index instead of a direct reference because their location
     * depends on where the state is stored, which can change during execution.
     * To access the variable, first get a reference to it from a context or a model with 
     * their subscript operator, which adds the current state location to the index.
     */
    class IntVarBitsetIdx {
        friend class IntVarBitsetRef;
        friend class AtomicIntVarBitsetRef;

    public:
        IntVarBitsetIdx(uint32_t index, int32_t offset) :
            index(index), offset(offset) {}

        IntVarBitsetIdx& operator+=(int32_t y) {
            offset += y;
            return *this;
        }

        IntVarBitsetIdx& operator-=(int32_t y) {
            offset -= y;
            return *this;
        }

    private:
        uint32_t index;
        int32_t offset;
    };


    inline IntVarBitsetIdx operator+(IntVarBitsetIdx x, int32_t y) {
        return x += y;
    }

    inline IntVarBitsetIdx operator+(int32_t x, IntVarBitsetIdx y) {
        return y += x;
    }

    inline IntVarBitsetIdx operator-(IntVarBitsetIdx x, int32_t y) {
        return x -= y;
    }

}

#endif