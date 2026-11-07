#ifndef MINIGCP_OBJECTIVE
#define MINIGCP_OBJECTIVE

#include <cstdint>
#include <concepts>
#include "sycl/sycl.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVar.hpp"

namespace cp {

    /**
     * @brief Objective variable
     */
    template<typename T>
    concept Objective = sycl::is_device_copyable_v<T> && requires(T const& x, Context ctx, int32_t v) {
        /**
         * @brief Get the current objective value, assuming the variable is fixed
         * @param ctx Current context
         * @return The value
         */
        { x.value(ctx) } -> std::same_as<int32_t>;

        /**
         * @brief Update the objective, ensuring subsequent solutions will be better
         * @param ctx Current context
         * @param v Lowest objective value found
         * @return Whether the domain was modified
         */
        { x.update(ctx, v) } -> std::same_as<DomainChange>;
    };


    /**
     * @brief Satisfaction problem objective
     */
    class Satisfaction {
    public:
        int32_t value(Context) const {
            return 0;
        }

        DomainChange update(Context, int32_t) const {
            return DomainChange::none;
        }
    };


    /**
     * @brief Minimization problem objective
     * @tparam T Objective variable type
     */
    template<IntVar T>
    class Minimization {
    public:
        Minimization(T x) : x(x) {}

        int32_t value(Context ctx) const {
            return *x[ctx].value();
        }

        DomainChange update(Context ctx, int32_t v) const {
            return x[ctx].removeAbove(v - 1);
        }

    private:
        T x;
    };


    /**
     * @brief Maximization problem objective
     * @tparam T Objective variable type
     */
    template<IntVar T>
    class Maximization {
    public:
        Maximization(T x) : x(x) {}

        int32_t value(Context ctx) const {
            return -*x[ctx].value();
        }

        DomainChange update(Context ctx, int32_t v) const {
            return x[ctx].removeBelow(-v + 1);
        }

    private:
        T x;
    };

}

#endif