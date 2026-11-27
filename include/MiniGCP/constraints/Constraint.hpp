#ifndef MINIGCP_CONSTRAINT
#define MINIGCP_CONSTRAINT

#include <concepts>
#include "sycl/sycl.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/EventRegistrar.hpp"

namespace cp {

    template<typename T>
    concept Constraint = sycl::is_device_copyable_v<T> && requires(T const& constraint, Context ctx, EventRegistrar& reg) {
        /**
         * @brief Whether the propagator uses a group of threads.
         * Otherwise, each thread in the group will propagate a different instance of the constraint.
         */
        { T::parallel } -> std::convertible_to<bool>;

        /**
         * @brief Register the constraint to variable domain change events
         * @param reg Event registrar
         */
        { constraint.post(reg) } -> std::same_as<void>;

        /**
         * @brief Propagate the constraint
         * @param ctx Current context
         * @return Whether any variable's domain was modified
         */
        { constraint.propagate(ctx) } -> std::same_as<DomainChange>;
    };

}

#endif