#ifndef MINIGCP_SUM
#define MINIGCP_SUM

#include <utility>
#include <span>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/utils.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"
#include "MiniGCP/state/ValueIdx.hpp"

namespace cp {

    /**
     * @brief Constraint : sum_i x_i = y.
     * Bound consistency, sequential implementation.
     */
    class SumSeq {
    public:
        constexpr static bool parallel = false;

        SumSeq(Allocator& allocator, std::span<IntVarIdx const> x, IntVarIdx y) :
            x(allocator.array<Scope::local, IntVarIdx>(x)),
            y(y),
            nFixed(allocator.value<Scope::state, uint32_t>(0)),
            sumFixed(allocator.value<Scope::state, int32_t>(0)) {}

        DomainChange propagate(Context ctx) const {
            uint32_t& nFixed = ctx[this->nFixed];
            int32_t& sumFixed = ctx[this->sumFixed];
            IntVarIdx* x = ctx[this->x];
            
            // Get sum of min and max values of x
            int32_t sumMin = sumFixed;
            int32_t sumMax = sumFixed;
            for (uint32_t i = nFixed; i < this->x.size(); i++) {
                int32_t min = ctx[x[i]].min();
                sumMin += min;
                sumMax += ctx[x[i]].max();
                if (ctx[x[i]].fixed()) {
                    std::swap(x[nFixed++], x[i]);
                    sumFixed += min;
                }
            }

            // Update min and max values
            DomainChange change = DomainChange::none;
            change |= ctx[y].removeBelow(sumMin);
            change |= ctx[y].removeAbove(sumMax);
            sumMin -= ctx[y].max();
            sumMax -= ctx[y].min();
            for (uint32_t i = nFixed; i < this->x.size(); i++) {
                int32_t min = ctx[x[i]].min();
                int32_t max = ctx[x[i]].max();
                change |= ctx[x[i]].removeBelow(-(sumMax - max));
                change |= ctx[x[i]].removeAbove(-(sumMin - min));
            }
            return change;
        }

    private:
        ArrayIdx<Scope::local, IntVarIdx> x;
        IntVarIdx y;
        ValueIdx<Scope::state, uint32_t> nFixed;
        ValueIdx<Scope::state, int32_t> sumFixed;
    };



    /**
     * @brief Constraint : sum_i x_i = y.
     * Bound consistency, parallel implementation.
     */
    class Sum {
    public:
        constexpr static bool parallel = true;

        Sum(Allocator& allocator, std::span<IntVarIdx const> x, IntVarIdx y) :
            x(allocator.array<Scope::local, IntVarIdx>(x)),
            y(y),
            nFixed(allocator.value<Scope::state, uint32_t>(0)),
            sumFixed(allocator.value<Scope::state, int32_t>(0)) {}

        DomainChange propagate(Context ctx) const {
            uint32_t nFixed = ctx[this->nFixed];
            int32_t sumFixed = ctx[this->sumFixed];
            IntVarIdx* x = ctx[this->x];
            
            // Get sum of min and max values of x
            int32_t sumMin = ctx.item.get_local_id(0) == 0 ? sumFixed : 0;
            int32_t sumMax = ctx.item.get_local_id(0) == 0 ? sumFixed : 0;
            for (uint32_t b = nFixed; b < this->x.size(); b += ctx.item.get_local_range(0)) {
                uint32_t i = b + ctx.item.get_local_id(0);
                if (i < this->x.size()) {
                    sumMin += ctx[x[i]].min();
                    sumMax += ctx[x[i]].max();
                }

                // Swap fixed variables sequentially
                std::optional<uint32_t> fixed;
                do {
                    bool broadcast = i >= nFixed && i < this->x.size() && ctx[x[i]].fixed();
                    fixed = groupBroadcastAny(ctx, broadcast ? std::optional(i) : std::nullopt);
                    if (fixed) {
                        if (ctx.item.get_local_id(0) == 0) {
                            std::swap(x[nFixed], x[*fixed]);
                            sumFixed += ctx[x[nFixed]].min();
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
                change |= ctx[y].removeBelow(sumMin);
                change |= ctx[y].removeAbove(sumMax);
            }
            sycl::group_barrier(ctx.item.get_group());
            sumMin -= ctx[y].max();
            sumMax -= ctx[y].min();
            for (uint32_t i = nFixed + ctx.item.get_local_id(0); i < this->x.size(); i += ctx.item.get_local_range(0)) {
                int32_t min = ctx[x[i]].min();
                int32_t max = ctx[x[i]].max();
                change |= ctx[x[i]].removeBelow(-(sumMax - max));
                change |= ctx[x[i]].removeAbove(-(sumMin - min));
            }

            if (ctx.item.get_local_id(0) == 0) {
                ctx[this->nFixed] = nFixed;
                ctx[this->sumFixed] = sumFixed;
            }
            return groupChange(ctx.item.get_group(), change);
        }

    private:
        ArrayIdx<Scope::local, IntVarIdx> x;
        IntVarIdx y;
        ValueIdx<Scope::state, uint32_t> nFixed;
        ValueIdx<Scope::state, int32_t> sumFixed;
    };


    /**
     * @brief Add a sum constraint to a model and return its result
     * @param model The model
     * @param x Variables
     * @return Sum of the variables
     */
    template<Constraint... Constraints>
    inline IntVarIdx sum(Model<Constraints...>& model, std::span<IntVarIdx const> x) {
        int32_t sumMin = 0;
        int32_t sumMax = 0;
        for (uint32_t i = 0; i < x.size(); i++) {
            sumMin += model[x[i]].min();
            sumMax += model[x[i]].max();
        }
        IntVarIdx y = model.variable(sumMin, sumMax);
        model.constraint(Sum(model, x, y));
        return y;
    }

}

#endif