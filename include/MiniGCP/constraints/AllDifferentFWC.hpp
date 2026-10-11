#ifndef MINIGCP_ALL_DIFFERENT_FWC
#define MINIGCP_ALL_DIFFERENT_FWC

#include <utility>
#include <span>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"
#include "MiniGCP/state/ValueIdx.hpp"

namespace cp {

    /**
     * @brief Constraint : xi != xj for all i, j.
     * Forward checking filtering, sequential implementation.
     */
    class AllDifferentFWCSeq {
    public:
        constexpr static bool parallel = false;

        AllDifferentFWCSeq(Allocator& allocator, std::span<IntVarIdx const> variables) :
            variables(allocator.array<Scope::local, IntVarIdx>(variables)),
            nFixed(allocator.value<Scope::state, uint32_t>(0)) {}

        DomainChange propagate(Context ctx) const {
            uint32_t& fixed = ctx[nFixed];
            IntVarIdx* vars = ctx[variables];
            DomainChange change = DomainChange::none;
            for (uint32_t i = fixed; i < variables.size(); i++) {
                if (ctx[vars[i]].fixed()) {
                    int32_t v = ctx[vars[i]].min();
                    std::swap(vars[fixed++], vars[i]);
                    for (uint32_t j = fixed; j < variables.size(); j++) {
                        change |= ctx[vars[j]].remove(v);
                    }
                }
            }
            return change;
        }

    private:
        ArrayIdx<Scope::local, IntVarIdx> variables;
        ValueIdx<Scope::state, uint32_t> nFixed;
    };



    /**
     * @brief Constraint : xi != xj for all i, j.
     * Forward checking filtering, parallel implementation.
     */
    class AllDifferentFWC {
    public:
        constexpr static bool parallel = true;

        AllDifferentFWC(Allocator& allocator, std::span<IntVarIdx const> variables) :
            variables(allocator.array<Scope::local, IntVarIdx>(variables)),
            nFixed(allocator.value<Scope::state, uint32_t>(0)) {
            allocator.temporary<uint32_t>(1);
        }

        DomainChange propagate(Context ctx) const {
            uint32_t fixed = ctx[nFixed];
            IntVarIdx* vars = ctx[variables];
            DomainChange change = DomainChange::none;
            uint32_t* index = ctx.getTemporary<uint32_t>();

            for (uint32_t b = fixed; b < variables.size(); b += ctx.item.get_local_range(0)) {
                uint32_t i = b + ctx.item.get_local_id(0);
                while (true) {
                    // Broadcast any fixed variable
                    if (ctx.item.get_local_id(0) == 0) *index = -1;
                    sycl::group_barrier(ctx.item.get_group());
                    if (i >= fixed && i < variables.size() && ctx[vars[i]].fixed()) *index = i;
                    sycl::group_barrier(ctx.item.get_group());

                    if (*index == -1) break; // No more fixed variables in this batch
                    else {
                        if (ctx.item.get_local_id(0) == 0) std::swap(vars[fixed], vars[*index]);
                        sycl::group_barrier(ctx.item.get_group());

                        // Remove fixed value from other variables
                        int32_t v = ctx[vars[fixed++]].min();
                        for (uint32_t j = fixed + ctx.item.get_local_id(0); j < variables.size(); j += ctx.item.get_local_range(0)) {
                            change |= ctx[vars[j]].remove(v);
                        }
                    }
                }
            }
            
            if (ctx.item.get_local_id(0) == 0) ctx[nFixed] = fixed;
            return groupChange(ctx.item.get_group(), change);
        }

    private:
        ArrayIdx<Scope::local, IntVarIdx> variables;
        ValueIdx<Scope::state, uint32_t> nFixed;
    };

}

#endif