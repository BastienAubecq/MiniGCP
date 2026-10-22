#ifndef MINIGCP_ALL_DIFFERENT_FWC
#define MINIGCP_ALL_DIFFERENT_FWC

#include <utility>
#include <span>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/utils.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVarBitsetIdx.hpp"
#include "MiniGCP/state/ValueIdx.hpp"

namespace cp {

    /**
     * @brief Constraint : x_i != x_j for all i, j.
     * Forward checking filtering, sequential implementation.
     */
    class AllDifferentFWCSeq {
    public:
        constexpr static bool parallel = false;

        AllDifferentFWCSeq(Allocator& allocator, std::span<IntVarBitsetIdx const> variables) :
            variables(allocator.array<Scope::local, IntVarBitsetIdx>(variables)),
            nFixed(allocator.value<Scope::state, uint32_t>(0)) {}

        DomainChange propagate(Context ctx) const {
            uint32_t& nFixed = ctx[this->nFixed];
            IntVarBitsetIdx* vars = ctx[variables];
            DomainChange change = DomainChange::none;
            for (uint32_t i = nFixed; i < variables.size(); i++) {
                if (ctx[vars[i]].fixed()) {
                    int32_t v = ctx[vars[i]].min();
                    std::swap(vars[nFixed++], vars[i]);
                    for (uint32_t j = nFixed; j < variables.size(); j++) {
                        change |= ctx[vars[j]].remove(v);
                    }
                }
            }
            return change;
        }

    private:
        ArrayIdx<Scope::local, IntVarBitsetIdx> variables;
        ValueIdx<Scope::state, uint32_t> nFixed;
    };



    /**
     * @brief Constraint : xi != xj for all i, j.
     * Forward checking filtering, parallel implementation.
     */
    class AllDifferentFWC {
    public:
        constexpr static bool parallel = true;

        AllDifferentFWC(Allocator& allocator, std::span<IntVarBitsetIdx const> variables) :
            variables(allocator.array<Scope::local, IntVarBitsetIdx>(variables)),
            nFixed(allocator.value<Scope::state, uint32_t>(0)) {}

        DomainChange propagate(Context ctx) const {
            uint32_t nFixed = ctx[this->nFixed];
            IntVarBitsetIdx* vars = ctx[variables];
            DomainChange change = DomainChange::none;

            for (uint32_t b = nFixed; b < variables.size(); b += ctx.item.get_local_range(0)) {
                uint32_t i = b + ctx.item.get_local_id(0);
                std::optional<uint32_t> fixed;
                do {
                    // Broadcast any fixed variable
                    bool broadcast = i >= nFixed && i < variables.size() && ctx[vars[i]].fixed();
                    fixed = groupBroadcastAny(ctx, broadcast ? std::optional(i) : std::nullopt);
                    
                    if (fixed) {
                        if (ctx.item.get_local_id(0) == 0) std::swap(vars[nFixed], vars[*fixed]);
                        sycl::group_barrier(ctx.item.get_group());

                        // Remove fixed value from other variables
                        int32_t v = ctx[vars[nFixed++]].min();
                        for (uint32_t j = nFixed + ctx.item.get_local_id(0); j < variables.size(); j += ctx.item.get_local_range(0)) {
                            change |= ctx[vars[j]].remove(v);
                        }
                    }
                } while (fixed);
            }
            
            if (ctx.item.get_local_id(0) == 0) ctx[this->nFixed] = nFixed;
            return groupChange(ctx.item.get_group(), change);
        }

    private:
        ArrayIdx<Scope::local, IntVarBitsetIdx> variables;
        ValueIdx<Scope::state, uint32_t> nFixed;
    };

}

#endif