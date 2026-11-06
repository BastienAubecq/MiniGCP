#ifndef MINIGCP_MODEL
#define MINIGCP_MODEL

#include <cstdint>
#include <vector>
#include <span>
#include <tuple>
#include <concepts>
#include <stdexcept>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/constraints/Constraint.hpp"

namespace cp {

    /**
     * @brief Constraint programming model.
     * It contains variables, data allocated by the propagators, and constraints.
     * @tparam Constraints Types of constraints used in the model.
     */
    template<Constraint... Constraints>
    class Model : public Allocator {
    public:
        /**
         * @brief Add a constraint to the model
         * @tparam T Type of the constraint
         * @param constraint The constraint
         */
        template<typename T> requires (std::same_as<T, Constraints> || ...)
        void constraint(T constraint) {
            std::vector<T>& c = std::get<std::vector<T>>(constraints);
            if (c.size() + 1 > UINT32_MAX) {
                throw std::range_error("Allocated size exceeds 32-bit integer range");
            }
            c.push_back(constraint);
        }

        /**
         * @brief Get the constraints
         */
        std::tuple<std::span<const Constraints>...> getConstraints() const {
            if (((std::get<std::vector<Constraints>>(constraints).size() == 0) || ...)) {
                throw std::range_error("Empty constraint list. Remove unused constraint types from the model declaration.");
            }
            return std::make_tuple(std::span(std::get<std::vector<Constraints>>(constraints))...);
        }

    private:
        std::tuple<std::vector<Constraints>...> constraints; // A vector of constraints per constraint type
    };

}

#endif