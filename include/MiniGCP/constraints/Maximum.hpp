#ifndef MINIGCP_MAXIMUM
#define MINIGCP_MAXIMUM

#include <algorithm>
#include <limits>
#include <utility>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/utils.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Ptr.hpp"

namespace cp {

    /**
     * @brief Constraint : max_i x_i = y.
     * Bound consistency, sequential implementation.
     * @tparam X Type of x
     * @tparam Y Type of y
     */
    template<IntVarPtr X, IntVar Y>
    class MaximumSeq {
    public:
        constexpr static bool parallel = false;

        MaximumSeq(Allocator& allocator, X x, uint32_t n, Y y) :
            x(x), n(n), y(y),
            notFixed(allocator, n),
            nFixed(allocator, 0),
            maxFixed(allocator, std::numeric_limits<int32_t>::min()) {
            int32_t maxMin = std::numeric_limits<int32_t>::min();
            int32_t maxMax = std::numeric_limits<int32_t>::min();
            for (uint32_t i = 0; i < n; i++) {
                notFixed[allocator, i] = i;
                maxMin = std::max(maxMin, x[allocator, i][allocator].min());
                maxMax = std::max(maxMax, x[allocator, i][allocator].max());
            }
            y[allocator].removeBelow(maxMin);
            y[allocator].removeAbove(maxMax);
        }

        DomainChange propagate(Context ctx) const {
            uint32_t nFixed = this->nFixed[ctx];
            int32_t maxFixed = this->maxFixed[ctx];
            uint32_t* notFixed = this->notFixed[ctx];
            DomainChange change = DomainChange::none;
            
            // Update x and get max of min and max values
            int32_t maxMin = maxFixed;
            int32_t maxMax = maxFixed;
            int32_t yMin = y[ctx].min();
            int32_t yMax = y[ctx].max();
            for (uint32_t i = nFixed; i < n; i++) {
                int32_t min = x[ctx, notFixed[i]][ctx].min();
                int32_t max = x[ctx, notFixed[i]][ctx].max();
                change |= x[ctx, notFixed[i]][ctx].removeAbove(yMax);
                maxMin = std::max(maxMin, min);
                maxMax = std::max(maxMax, max);
                if (min == max || max < yMin) { // Don't process again if fixed or outside the range y
                    std::swap(notFixed[nFixed++], notFixed[i]);
                    maxFixed = std::max(maxFixed, max);
                }
            }

            // Update y
            change |= y[ctx].removeBelow(maxMin);
            change |= y[ctx].removeAbove(maxMax);

            // y = x_i if x_i is the only variable inside the range of y
            if (nFixed == n - 1 && maxFixed < yMin) {
                change |= x[ctx, notFixed[n - 1]][ctx].removeBelow(yMin);
            }

            this->nFixed[ctx] = nFixed;
            this->maxFixed[ctx] = maxFixed;
            return change;
        }

    private:
        X x;
        uint32_t n;
        Y y;
        Ptr<uint32_t, Scope::local> notFixed; // Indices of the unfixed variables
        Ref<uint32_t, Scope::state> nFixed; // Number of unfixed variables
        Ref<int32_t, Scope::state> maxFixed; // Max of the unfixed variables
    };


    /**
     * @brief Constraint : max_i x_i = y.
     * Bound consistency, parallel implementation.
     * @tparam X Type of x
     * @tparam Y Type of y
     */
    template<IntVarPtr X, IntVar Y>
    class Maximum {
    public:
        constexpr static bool parallel = true;

        Maximum(Allocator& allocator, X x, uint32_t n, Y y) :
            x(x), n(n), y(y),
            notFixed(allocator, n),
            nFixed(allocator, 0),
            maxFixed(allocator, std::numeric_limits<int32_t>::min()) {
            int32_t maxMin = std::numeric_limits<int32_t>::min();
            int32_t maxMax = std::numeric_limits<int32_t>::min();
            for (uint32_t i = 0; i < n; i++) {
                notFixed[allocator, i] = i;
                maxMin = std::max(maxMin, x[allocator, i][allocator].min());
                maxMax = std::max(maxMax, x[allocator, i][allocator].max());
            }
            y[allocator].removeBelow(maxMin);
            y[allocator].removeAbove(maxMax);
        }

        DomainChange propagate(Context ctx) const {
            uint32_t nFixed = this->nFixed[ctx];
            int32_t maxFixed = this->maxFixed[ctx];
            uint32_t* notFixed = this->notFixed[ctx];
            DomainChange change = DomainChange::none;

            // Update x and get max of min and max values
            int32_t maxMin = maxFixed;
            int32_t maxMax = maxFixed;
            int32_t yMin = y[ctx].min();
            int32_t yMax = y[ctx].max();
            for (uint32_t b = nFixed; b < n; b += ctx.item.get_local_range(0)) {
                uint32_t i = b + ctx.item.get_local_id(0);
                if (i < n) {
                    maxMin = std::max(maxMin, x[ctx, notFixed[i]][ctx].min());
                    maxMax = std::max(maxMax, x[ctx, notFixed[i]][ctx].max());
                    change |= x[ctx, notFixed[i]][ctx].removeAbove(yMax);
                }

                // Swap fixed variables sequentially
                std::optional<int32_t> fixed;
                do {
                    bool broadcast = i >= nFixed && i < n &&
                        (x[ctx, notFixed[i]][ctx].value() || x[ctx, notFixed[i]][ctx].max() < yMin);
                    fixed = groupBroadcastAny(ctx, broadcast ? std::optional(i) : std::nullopt);

                    if (fixed) {
                        if (i == *fixed) {
                            std::swap(notFixed[nFixed], notFixed[*fixed]);
                            maxFixed = std::max(maxFixed, x[ctx, notFixed[nFixed]][ctx].max());
                        }
                        nFixed++;
                        sycl::group_barrier(ctx.item.get_group());
                    }
                } while (fixed);
            }
            maxMin = sycl::reduce_over_group(ctx.item.get_group(), maxMin, sycl::maximum<int32_t>());
            maxMax = sycl::reduce_over_group(ctx.item.get_group(), maxMax, sycl::maximum<int32_t>());

            // Update y
            if (ctx.item.get_local_id(0) == 0) {
                change |= y[ctx].removeBelow(maxMin);
                change |= y[ctx].removeAbove(maxMax);
            }

            // y = x_i if x_i is the only variable inside the range of y
            if (ctx.item.get_local_id(0) == 0 && nFixed == n - 1 && maxFixed < y[ctx].min()) {
                change |= x[ctx, notFixed[n - 1]][ctx].removeBelow(yMin);
            }

            if (ctx.item.get_local_id(0) == 0) {
                this->nFixed[ctx] = nFixed;
                this->maxFixed[ctx] = maxFixed;
            }
            return change;
        }

    private:
        X x;
        uint32_t n;
        [[no_unique_address]] Y y;
        Ptr<uint32_t, Scope::local> notFixed; // Indices of the unfixed variables
        Ref<uint32_t, Scope::state> nFixed; // Number of unfixed variables
        Ref<int32_t, Scope::state> maxFixed; // Max of the unfixed variables
    };


    template<Constraint... Constraints, IntVarPtr X>
    inline IntInterval maximum(Model<Constraints...>& model, X x, uint32_t n) {
        IntInterval y(model);
        model.constraint(Maximum(model, x, n, y));
        return y;
    }

}

#endif