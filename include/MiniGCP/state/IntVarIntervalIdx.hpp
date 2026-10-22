#ifndef MINIGCP_INT_VAR_INTERVAL_IDX
#define MINIGCP_INT_VAR_INTERVAL_IDX

#include <cstdint>

namespace cp {

    /**
     * @brief Index of an integer variable represented with an interval in a state.
     * Variables are represented with an index instead of a direct reference because their location
     * depends on where the state is stored, which can change during execution.
     * To access the variable, first get a reference to it from a context or a model with 
     * their subscript operator, which adds the current state location to the index.
     */
    class IntVarIntervalIdx {
        friend class IntVarIntervalRef;
        friend class AtomicIntVarIntervalRef;

    public:
        IntVarIntervalIdx(uint32_t index, int32_t offset) :
            index(index), offset(offset) {}

        IntVarIntervalIdx& operator+=(int32_t y) {
            offset += y;
            return *this;
        }

        IntVarIntervalIdx& operator-=(int32_t y) {
            offset -= y;
            return *this;
        }

    private:
        uint32_t index;
        int32_t offset;
    };


    inline IntVarIntervalIdx operator+(IntVarIntervalIdx x, int32_t y) {
        return x += y;
    }

    inline IntVarIntervalIdx operator+(int32_t x, IntVarIntervalIdx y) {
        return y += x;
    }

    inline IntVarIntervalIdx operator-(IntVarIntervalIdx x, int32_t y) {
        return x -= y;
    }

}

#endif
