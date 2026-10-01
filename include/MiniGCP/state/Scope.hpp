#ifndef MINIGCP_SCOPE
#define MINIGCP_SCOPE

namespace cp {
    
    /**
     * @brief Scope of a memory allocation
     */
    enum class Scope : int {
        /**
         * @brief Global allocation.
         * It can be accessed by all groups of threads.
         */
        global = 0,

        /**
         * @brief Local allocation.
         * Each group of threads accesses its own copy of the allocation.
         */
        local = 1,

        /**
         * @brief State allocation.
         * Each group of threads accesses its own copy of the allocation.
         * The content is restored when backtracking.
         */
        state = 2
    };

}

#endif