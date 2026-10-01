#ifndef MINIGCP_PROBLEM_TYPE
#define MINIGCP_PROBLEM_TYPE

namespace cp {
    
    enum class ProblemType {
        /**
         * @brief Satisfaction problem
         */
        satisfaction,

        /**
         * @brief Minimization problem
         */
        minimum,

        /**
         * @brief Maximization problem
         */
        maximum
    };

}

#endif