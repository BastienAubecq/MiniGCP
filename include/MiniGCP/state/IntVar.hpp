#ifndef MINIGCP_INT_VAR
#define MINIGCP_INT_VAR

#include <cstdint>
#include <optional>
#include <concepts>
#include "sycl/sycl.hpp"
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"

namespace cp {

    /**
     * @brief Reference to an integer variable
     */
    template<typename T>
    concept IntVarRef = requires(T var, int32_t v) {
        /**
         * @brief Get the minimum value in the domain
         */
        { var.min() } -> std::same_as<int32_t>;

        /**
         * @brief Get the maximum value in the domain
         */
        { var.max() } -> std::same_as<int32_t>;

        /**
         * @brief Get the number of values in the domain
         */
        { var.size() } -> std::same_as<int32_t>;

        /**
         * @brief Check if the domain contains a single value and, if so, return it
         */
        { var.value() } -> std::same_as<std::optional<int32_t>>;

        /**
         * @brief Whether the domain contains a value
         * @param v The value
         */
        { var.contains(v) } -> std::same_as<bool>;

        /**
         * @brief Remove a value from the domain
         * @param v The value
         * @return Whether the domain was modified
         */
        { var.remove(v) } -> std::same_as<DomainChange>;

        /**
         * @brief Fix the domain to a single value
         * @param v The value
         * @return Whether the domain was modified
         */
        { var.fix(v) } -> std::same_as<DomainChange>;

        /**
         * @brief Remove all values below a value from the domain
         * @param v The value
         * @return Whether the domain was modified
         */
        { var.removeBelow(v) } -> std::same_as<DomainChange>;

        /**
         * @brief Remove all values above a value from the domain
         * @param v The value
         * @return Whether the domain was modified
         */
        { var.removeAbove(v) } -> std::same_as<DomainChange>;
    };


    /**
     * @brief Integer variable
     */
    template<typename T>
    concept IntVar = sycl::is_device_copyable_v<T> && requires(T const& var, Context ctx, Allocator& allocator) {
        { var[ctx] } -> IntVarRef;
        { var[allocator] } -> IntVarRef;
    };

    /**
     * @brief Integer variable pointer
     */
    template<typename T>
    concept IntVarPtr = sycl::is_device_copyable_v<T> && requires(T const& ptr, Context ctx, Allocator& allocator, uint32_t i) {
        { auto(ptr[ctx, i]) } -> IntVar;
        { auto(ptr[allocator, i]) } -> IntVar;
    };

    template<IntVarPtr T>
    using IntVarIn = std::remove_reference_t<decltype(std::declval<T>()[std::declval<Context>(), 0])>;

}

#endif