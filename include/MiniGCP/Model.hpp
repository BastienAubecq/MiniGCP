#ifndef MINIGCP_MODEL
#define MINIGCP_MODEL

#include <array>
#include <cstdint>
#include <vector>
#include <span>
#include <tuple>
#include <concepts>
#include <stdexcept>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/constraints/Constraint.hpp"
#include "MiniGCP/state/EventRegistrar.hpp"

namespace cp {

    namespace detail {
        template<typename T, typename First, typename... Rest>
        consteval uint32_t typeIndexOf() {
            if constexpr (std::same_as<T, First>) return 0;
            else return 1 + typeIndexOf<T, Rest...>();
        }
    }

    /**
     * @brief Layout of per-work-group event tracking buffers in Scope::local
     * @tparam N Number of constraint types in the model
     */
    template<size_t N>
    struct EventStoreLayout {
        uint32_t varEventsOffset = 0;
        uint32_t varEventsWords = 0;
        uint32_t activeOffset = 0;
        uint32_t totalActiveWords = 0;
        std::array<uint32_t, N> typeWordOffsets{};
        std::array<uint32_t, N> typeWords{};
    };

    /**
     * @brief Constraint programming model.
     * It contains variables, data allocated by the propagators, and constraints.
     * @tparam Constraints Types of constraints used in the model.
     */
    template<Constraint... Constraints>
    class Model : public Allocator {
    public:
        /**
         * @brief Add a constraint to the model and register its domain change events
         * @tparam T Type of the constraint
         * @param constraint The constraint
         */
        template<typename T> requires (std::same_as<T, Constraints> || ...)
        void constraint(T constraint) {
            std::vector<T>& c = std::get<std::vector<T>>(constraints);
            if (c.size() + 1 > UINT32_MAX) {
                throw std::range_error("Allocated size exceeds 32-bit integer range");
            }
            uint32_t instIdx = static_cast<uint32_t>(c.size());
            constexpr uint32_t typeIdx = detail::typeIndexOf<T, Constraints...>();
            c.push_back(constraint);
            EventRegistrar reg(*this, subscriptions, typeIdx, instIdx);
            c.back().post(reg);
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

        /**
         * @brief Get all recorded event subscriptions
         */
        std::span<EventSubscription const> getSubscriptions() const {
            return subscriptions;
        }

        /**
         * @brief Allocate per-work-group event buffers (varEvents & active bitsets) in Scope::local
         * and temporary compaction storage. Called before transferring data to the device.
         */
        EventStoreLayout<sizeof...(Constraints)> prepareEventStore() {
            EventStoreLayout<sizeof...(Constraints)> layout;
            std::array<uint32_t, sizeof...(Constraints)> sizes{
                static_cast<uint32_t>(std::get<std::vector<Constraints>>(constraints).size())...
            };

            uint32_t wordOffset = 0;
            for (size_t i = 0; i < sizeof...(Constraints); ++i) {
                layout.typeWordOffsets[i] = wordOffset;
                layout.typeWords[i] = (sizes[i] + 31u) / 32u;
                wordOffset += layout.typeWords[i];
            }
            layout.totalActiveWords = wordOffset;
            layout.varEventsWords = (getNumVars() + 7u) / 8u;

            if (layout.varEventsWords > 0) {
                layout.varEventsOffset = allocate<uint32_t, Scope::local>(layout.varEventsWords);
            }
            if (layout.totalActiveWords > 0) {
                layout.activeOffset = allocate<uint32_t, Scope::local>(layout.totalActiveWords);
                uint32_t* active = getData<uint32_t, Scope::local>(layout.activeOffset);
                // Initialize all valid constraint bits to 1 so reset() schedules initial root propagation
                for (size_t i = 0; i < sizeof...(Constraints); ++i) {
                    uint32_t n = sizes[i];
                    uint32_t fullWords = n / 32u;
                    uint32_t rem = n % 32u;
                    uint32_t base = layout.typeWordOffsets[i];
                    for (uint32_t w = 0; w < fullWords; ++w) {
                        active[base + w] = ~0u;
                    }
                    if (rem > 0) {
                        active[base + fullWords] = (1u << rem) - 1u;
                    }
                }
            }

            // Ensure temporary local memory can hold the compacted active list for any non-parallel constraint type
            uint32_t maxNonParallel = 0;
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                (([&] {
                    if constexpr (!std::tuple_element_t<Is, std::tuple<Constraints...>>::parallel) {
                        if (sizes[Is] > maxNonParallel) maxNonParallel = sizes[Is];
                    }
                }()), ...);
            }(std::make_index_sequence<sizeof...(Constraints)>{});
            if (maxNonParallel > 0) {
                temporary<uint32_t>(maxNonParallel);
            }

            return layout;
        }

    private:
        std::tuple<std::vector<Constraints>...> constraints; // A vector of constraints per constraint type
        std::vector<EventSubscription> subscriptions;
    };

}

#endif