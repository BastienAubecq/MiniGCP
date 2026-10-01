#ifndef MINIGCP_CONSTRAINT_STORE
#define MINIGCP_CONSTRAINT_STORE

#include <cstdint>
#include <tuple>
#include <array>
#include "sycl/sycl.hpp"
#include "MiniGCP/constraints/Constraint.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"

namespace cp {

    /**
     * @brief Device-side constraints
     * @tparam Constraints Types of constraints in the constraint store.
     */
    template<Constraint... Constraints>
    class ConstraintStore {
    public:
        ConstraintStore(sycl::queue& queue, std::span<Constraints const>... constraints) :
            constraints(sycl::malloc_device<Constraints>(constraints.size(), queue)...),
            sizes{static_cast<uint32_t>(constraints.size())...} {
            (queue.memcpy(std::get<Constraints*>(this->constraints), constraints.data(), constraints.size() * sizeof(Constraints)).wait(), ...);
        }

        ConstraintStore(sycl::queue& queue, std::tuple<std::span<Constraints const>...> constraints) :
            ConstraintStore(queue, std::get<std::span<Constraints const>>(constraints)...) {}

        void destroy(sycl::queue& queue) {
            (sycl::free(std::get<Constraints*>(constraints), queue), ...);
        }


        /**
         * @brief Propagate the constraints until a fixpoint or inconsistency is reached
         * @param ctx Current context
         * @return Whether an inconsistency was detected
         */
        bool fixpoint(Context ctx) const {
            DomainChange change;
            do {
                change = propagate(ctx);
                if ((change & DomainChange::inconsistency) == DomainChange::inconsistency) {
                    return true;
                }
            } while (change != DomainChange::none);
            return false;
        }


    private:
        std::tuple<Constraints*...> constraints;
        std::array<uint32_t, sizeof...(Constraints)> sizes;

        /**
         * @brief Propagate all constraints starting from list `i`
         * @param ctx Current context
         * @return Whether any variable's domain was modified
         */
        template<int i = 0>
        DomainChange propagate(Context ctx) const {
            if constexpr (i == sizeof...(Constraints)) {
                return DomainChange::none;
            }
            else {
                DomainChange change = DomainChange::none;
                if constexpr (std::tuple_element_t<i, std::tuple<Constraints...>>::parallel) {
                    // Propagate each constraint in parallel with a group of threads
                    for (uint32_t j = 0; j < std::get<i>(sizes); j++) {
                        change |= std::get<i>(constraints)[j].propagate(ctx);
                        if ((change & DomainChange::inconsistency) == DomainChange::inconsistency) {
                            return DomainChange::inconsistency;
                        }
                    }
                }
                else {
                    // Propagate constraints by batches in parallel with a group of threads
                    for (uint32_t j = ctx.item.get_local_id(0); j < std::get<i>(sizes); j += ctx.item.get_local_range(0)) {
                        change |= std::get<i>(constraints)[j].propagate(ctx);
                    }
                    if (sycl::any_of_group(ctx.item.get_group(), (change & DomainChange::inconsistency) == DomainChange::inconsistency)) {
                        return DomainChange::inconsistency;
                    }
                    if (sycl::any_of_group(ctx.item.get_group(), (change & DomainChange::changed) == DomainChange::changed)) {
                        change |= DomainChange::changed;
                    }
                }
                return change | propagate<i+1>(ctx);
            }
        }
    };

}

#endif