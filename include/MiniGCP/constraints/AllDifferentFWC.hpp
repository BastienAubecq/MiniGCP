#ifndef MINIGCP_ALL_DIFFERENT_FWC
#define MINIGCP_ALL_DIFFERENT_FWC

#include <utility>
#include <span>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/utils.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Ptr.hpp"

namespace cp {

    /**
     * @brief Constraint : x_i != x_j for all i, j.
     * Forward checking filtering, sequential implementation.
     * @tparam X Type of x
     */
    template<IntVarPtr X>
    class AllDifferentFWCSeq {
    public:
        constexpr static bool parallel = false;

        AllDifferentFWCSeq(Allocator& allocator, X x, uint32_t n) :
            x(x), n(n),
            notFixed(allocator, n),
            nFixed(allocator, 0) {
            for (uint32_t i = 0; i < n; i++) {
                notFixed[allocator, i] = i;
            }
        }

        DomainChange propagate(Context ctx) const {
            uint32_t nFixed = this->nFixed[ctx];
            uint32_t* notFixed = this->notFixed[ctx];
            DomainChange change = DomainChange::none;

            for (uint32_t i = nFixed; i < n; i++) {
                std::optional<int32_t> v = x[ctx, notFixed[i]][ctx].value();
                if (v) {
                    std::swap(notFixed[nFixed++], notFixed[i]);
                    for (uint32_t j = nFixed; j < n; j++) {
                        change |= x[ctx, notFixed[j]][ctx].remove(*v);
                    }
                }
            }

            this->nFixed[ctx] = nFixed;
            return change;
        }

    private:
        X x;
        uint32_t n;
        Ptr<uint32_t, Scope::local> notFixed; // Indices of the unfixed variables
        Ref<uint32_t, Scope::state> nFixed; // Number of unfixed variables
    };



    /**
     * @brief Constraint : xi != xj for all i, j.
     * Forward checking filtering, parallel implementation.
     * @tparam X Type of x
     */
    template<IntVarPtr X>
    class AllDifferentFWC {
    public:
        constexpr static bool parallel = true;

        AllDifferentFWC(Allocator& allocator, X x, uint32_t n) :
            x(x), n(n),
            notFixed(allocator, n),
            nFixed(allocator, 0) {
            for (uint32_t i = 0; i < n; i++) {
                notFixed[allocator, i] = i;
            }
        }

        DomainChange propagate(Context ctx) const {
            uint32_t nFixed = this->nFixed[ctx];
            uint32_t* notFixed = this->notFixed[ctx];
            DomainChange change = DomainChange::none;

            for (uint32_t b = nFixed; b < n; b += ctx.item.get_local_range(0)) {
                uint32_t i = b + ctx.item.get_local_id(0);
                std::optional<uint32_t> fixed;
                do {
                    // Broadcast any fixed variable
                    bool broadcast = i >= nFixed && i < n && x[ctx, notFixed[i]][ctx].value();
                    fixed = groupBroadcastAny(ctx, broadcast ? std::optional(i) : std::nullopt);
                    
                    if (fixed) {
                        if (ctx.item.get_local_id(0) == 0) std::swap(notFixed[nFixed], notFixed[*fixed]);
                        sycl::group_barrier(ctx.item.get_group());

                        // Remove fixed value from other x
                        int32_t v = x[ctx, notFixed[nFixed++]][ctx].min();
                        for (uint32_t j = nFixed + ctx.item.get_local_id(0); j < n; j += ctx.item.get_local_range(0)) {
                            change |= x[ctx, notFixed[j]][ctx].remove(v);
                        }
                        sycl::group_barrier(ctx.item.get_group());
                    }
                } while (fixed);
            }
            
            if (ctx.item.get_local_id(0) == 0) this->nFixed[ctx] = nFixed;
            return groupChange(ctx.item.get_group(), change);
        }

    private:
        X x;
        uint32_t n;
        Ptr<uint32_t, Scope::local> notFixed; // Indices of the unfixed variables
        Ref<uint32_t, Scope::state> nFixed; // Number of unfixed variables
    };

}

#endif