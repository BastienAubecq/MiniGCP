#ifndef MINIGCP_CONSTRAINT
#define MINIGCP_CONSTRAINT

#include <concepts>
#include <type_traits>
#include "sycl/sycl.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"

namespace cp {

    template<typename T>
    concept Constraint = sycl::is_device_copyable_v<T> && requires(T constraint, Context ctx) {
        /**
         * @brief Whether the propagator uses a group of threads.
         * Otherwise, each thread in the group will propagate a different instance of the constraint.
         */
        { T::parallel } -> std::convertible_to<bool>;

        /**
         * @brief Propagate the constraint.
         * If a group of thread is used, all threads must return the same value.
         * @param ctx Current context
         * @return Whether any variable's domain was modified
         */
        { constraint.propagate(ctx) } -> std::same_as<DomainChange>;
    };

}

#endif