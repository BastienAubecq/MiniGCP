#ifndef MINIGCP_SUM
#define MINIGCP_SUM

#include <utility>
#include <span>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/utils.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Ptr.hpp"

namespace cp {

    /**
     * @brief Constraint : sum_i x_i = y.
     * Bound consistency, sequential implementation.
     * @tparam X Type of x
     * @tparam Y Type of y
     */
    template<IntVarPtr X, IntVar Y>
    class SumSeq {
    public:
        constexpr static bool parallel = false;

        SumSeq(Allocator& allocator, X x, uint32_t n, Y y) :
            x(x), n(n), y(y),
            notFixed(allocator, n),
            nFixed(allocator, 0),
            sumFixed(allocator, 0) {
            int32_t sumMin = 0;
            int32_t sumMax = 0;
            for (uint32_t i = 0; i < n; i++) {
                notFixed[allocator, i] = i;
                sumMin += x[allocator, i][allocator].min();
                sumMax += x[allocator, i][allocator].max();
            }
            y[allocator].removeBelow(sumMin);
            y[allocator].removeAbove(sumMax);
        }

        DomainChange propagate(Context ctx) const {
            uint32_t nFixed = this->nFixed[ctx];
            int32_t sumFixed = this->sumFixed[ctx];
            uint32_t* notFixed = this->notFixed[ctx];
            
            // Get sum of min and max values of x
            int32_t sumMin = sumFixed;
            int32_t sumMax = sumFixed;
            for (uint32_t i = nFixed; i < n; i++) {
                int32_t min = x[ctx, notFixed[i]][ctx].min();
                int32_t max = x[ctx, notFixed[i]][ctx].max();
                sumMin += min;
                sumMax += max;
                if (min == max) {
                    std::swap(notFixed[nFixed++], notFixed[i]);
                    sumFixed += min;
                }
            }

            // Update min and max values
            DomainChange change = DomainChange::none;
            change |= y[ctx].removeBelow(sumMin);
            change |= y[ctx].removeAbove(sumMax);
            sumMin -= y[ctx].max();
            sumMax -= y[ctx].min();
            for (uint32_t i = nFixed; i < n; i++) {
                int32_t min = x[ctx, notFixed[i]][ctx].min();
                int32_t max = x[ctx, notFixed[i]][ctx].max();
                change |= x[ctx, notFixed[i]][ctx].removeBelow(-(sumMax - max));
                change |= x[ctx, notFixed[i]][ctx].removeAbove(-(sumMin - min));
            }

            this->nFixed[ctx] = nFixed;
            this->sumFixed[ctx] = sumFixed;
            return change;
        }

    private:
        X x;
        uint32_t n;
        Y y;
        Ptr<uint32_t, Scope::local> notFixed; // Indices of the unfixed variables
        Ref<uint32_t, Scope::state> nFixed; // Number of unfixed variables
        Ref<int32_t, Scope::state> sumFixed; // Sum of the unfixed variables
    };



    /**
     * @brief Constraint : sum_i x_i = y.
     * Bound consistency, parallel implementation.
     * @tparam X Type of x
     * @tparam Y Type of y
     */
    template<IntVarPtr X, IntVar Y>
    class Sum {
    public:
        constexpr static bool parallel = true;

        Sum(Allocator& allocator, X x, uint32_t n, Y y) :
            x(x), n(n), y(y),
            notFixed(allocator, n),
            nFixed(allocator, 0),
            sumFixed(allocator, 0) {
            int32_t sumMin = 0;
            int32_t sumMax = 0;
            for (uint32_t i = 0; i < n; i++) {
                notFixed[allocator, i] = i;
                sumMin += x[allocator, i][allocator].min();
                sumMax += x[allocator, i][allocator].max();
            }
            y[allocator].removeBelow(sumMin);
            y[allocator].removeAbove(sumMax);
        }

        DomainChange propagate(Context ctx) const {
            uint32_t nFixed = this->nFixed[ctx];
            uint32_t* notFixed = this->notFixed[ctx];
            int32_t& sumFixed = this->sumFixed[ctx];
            
            // Get sum of min and max values of x
            int32_t sumMin = ctx.item.get_local_id(0) == 0 ? sumFixed : 0;
            int32_t sumMax = ctx.item.get_local_id(0) == 0 ? sumFixed : 0;
            for (uint32_t b = nFixed; b < n; b += ctx.item.get_local_range(0)) {
                uint32_t i = b + ctx.item.get_local_id(0);
                if (i < n) {
                    sumMin += x[ctx, notFixed[i]][ctx].min();
                    sumMax += x[ctx, notFixed[i]][ctx].max();
                }

                // Swap fixed variables sequentially
                std::optional<uint32_t> fixed;
                do {
                    std::optional<int32_t> v;
                    std::optional<uint32_t> broadcast;
                    if (i >= nFixed && i < n) {
                        v = x[ctx, notFixed[i]][ctx].value();
                        if (v) broadcast = i;
                    }
                    fixed = groupBroadcastAny(ctx, broadcast);

                    if (fixed) {
                        if (i == *fixed) {
                            std::swap(notFixed[nFixed], notFixed[i]);
                            sumFixed += *v;
                        }
                        nFixed++;
                        sycl::group_barrier(ctx.item.get_group());
                    }
                } while (fixed);
            }

            // Update min and max values
            DomainChange change = DomainChange::none;
            sumMin = sycl::reduce_over_group(ctx.item.get_group(), sumMin, sycl::plus<int32_t>());
            sumMax = sycl::reduce_over_group(ctx.item.get_group(), sumMax, sycl::plus<int32_t>());
            if (ctx.item.get_local_id(0) == 0) {
                change |= y[ctx].removeBelow(sumMin);
                change |= y[ctx].removeAbove(sumMax);
            }
            sycl::group_barrier(ctx.item.get_group());
            sumMin -= y[ctx].max();
            sumMax -= y[ctx].min();
            for (uint32_t i = nFixed + ctx.item.get_local_id(0); i < n; i += ctx.item.get_local_range(0)) {
                int32_t min = x[ctx, notFixed[i]][ctx].min();
                int32_t max = x[ctx, notFixed[i]][ctx].max();
                change |= x[ctx, notFixed[i]][ctx].removeBelow(-(sumMax - max));
                change |= x[ctx, notFixed[i]][ctx].removeAbove(-(sumMin - min));
            }

            if (ctx.item.get_local_id(0) == 0) this->nFixed[ctx] = nFixed;
            return groupChange(ctx.item.get_group(), change);
        }

    private:
        X x;
        uint32_t n;
        [[no_unique_address]] Y y;
        Ptr<uint32_t, Scope::local> notFixed; // Indices of the unfixed variables
        Ref<uint32_t, Scope::state> nFixed; // Number of unfixed variables
        Ref<int32_t, Scope::state> sumFixed; // Sum of the unfixed variables
    };


    /**
     * @brief Add a sum constraint to a model and return its result
     * @param model The model
     * @param x Variables
     * @return Sum of the variables
     */
    template<Constraint... Constraints, IntVarPtr X>
    inline IntInterval sum(Model<Constraints...>& model, X x, uint32_t n) {
        IntInterval y(model);
        model.constraint(Sum(model, x, n, y));
        return y;
    }

}

#endif