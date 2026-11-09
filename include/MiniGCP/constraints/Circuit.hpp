#ifndef MINIGCP_CIRCUIT
#define MINIGCP_CIRCUIT

#include <utility>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/utils.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Ptr.hpp"

namespace cp {

    /**
     * @brief Constraint : the graph represented by the successor of each node doesn't contain any cycles that don't involve all nodes.
     * Sequential implementation.
     * @tparam Succ Type of the successors
     */
    template<IntVarPtr Succ>
    class CircuitSeq {
    public:
        constexpr static bool parallel = false;

        CircuitSeq(Allocator& allocator, Succ succ, uint32_t n) :
            succ(succ), n(n),
            notFixed(allocator, n),
            pathEnd(allocator, n),
            nFixed(allocator, 0) {
            for (uint32_t i = 0; i < n; i++) {
                notFixed[allocator, i] = i;
                pathEnd[allocator, i] = i;
                succ[allocator, i][allocator].removeBelow(0);
                succ[allocator, i][allocator].removeAbove(n - 1);
                succ[allocator, i][allocator].remove(i);
            }
        }

        DomainChange propagate(Context ctx) const {
            uint32_t nFixed = this->nFixed[ctx];
            uint32_t* notFixed = this->notFixed[ctx];
            uint32_t* pathEnd = this->pathEnd[ctx];
            DomainChange change = DomainChange::none;

            for (uint32_t i = nFixed; i < n; i++) {
                uint32_t u = notFixed[i];
                std::optional<int32_t> v = succ[ctx, u][ctx].value();
                if (v) {
                    std::swap(notFixed[nFixed++], notFixed[i]);
                    if (nFixed < n - 1) { // Prevent cycle
                        uint32_t src = pathEnd[u];
                        uint32_t dst = pathEnd[*v];
                        pathEnd[src] = dst;
                        pathEnd[dst] = src;
                        change |= succ[ctx, dst][ctx].remove(src);
                    }
                }
            }

            this->nFixed[ctx] = nFixed;
            return change;
        }

    private:
        Succ succ;
        uint32_t n;
        Ptr<uint32_t, Scope::local> notFixed; // Indices of the unfixed variables
        Ptr<uint32_t, Scope::state> pathEnd; // Other end of the path that starts at each node
        Ref<uint32_t, Scope::state> nFixed; // Number of unfixed variables
    };



    /**
     * @brief Constraint : the graph represented by the successor of each node doesn't contain any cycles that don't involve all nodes.
     * Parallel implementation.
     * @tparam Succ Type of the successors
     */
    template<IntVarPtr Succ>
    class Circuit {
    public:
        constexpr static bool parallel = true;

        Circuit(Allocator& allocator, Succ succ, uint32_t n) :
            succ(succ), n(n),
            notFixed(allocator, n),
            pathEnd(allocator, n),
            nFixed(allocator, 0) {
            for (uint32_t i = 0; i < n; i++) {
                notFixed[allocator, i] = i;
                pathEnd[allocator, i] = i;
                succ[allocator, i][allocator].removeBelow(0);
                succ[allocator, i][allocator].removeAbove(n - 1);
                succ[allocator, i][allocator].remove(i);
            }
        }

        DomainChange propagate(Context ctx) const {
            uint32_t nFixed = this->nFixed[ctx];
            uint32_t* notFixed = this->notFixed[ctx];
            uint32_t* pathEnd = this->pathEnd[ctx];
            DomainChange change = DomainChange::none;

            for (uint32_t b = nFixed; b < n; b += ctx.item.get_local_range(0)) {
                uint32_t i = b + ctx.item.get_local_id(0);
                std::optional<uint32_t> fixed;
                do {
                    // Broadcast any fixed variable
                    uint32_t u;
                    std::optional<int32_t> v;
                    std::optional<uint32_t> broadcast;
                    if (i >= nFixed && i < n) {
                        u = notFixed[i];
                        v = succ[ctx, u][ctx].value();
                        if (v) broadcast = i;
                    }
                    fixed = groupBroadcastAny(ctx, broadcast);
                    
                    if (fixed) {
                        if (i == *fixed) {
                            u = notFixed[i];
                            std::swap(notFixed[nFixed], notFixed[i]);
                            if (nFixed < n - 2) { // Prevent cycle
                                uint32_t src = pathEnd[u];
                                uint32_t dst = pathEnd[*v];
                                pathEnd[src] = dst;
                                pathEnd[dst] = src;
                                change |= succ[ctx, dst][ctx].remove(src);
                            }
                        }
                        nFixed++;
                        sycl::group_barrier(ctx.item.get_group());
                    }
                } while (fixed);
            }
            
            if (ctx.item.get_local_id(0) == 0) this->nFixed[ctx] = nFixed;
            return groupChange(ctx.item.get_group(), change);
        }

    private:
        Succ succ;
        uint32_t n;
        Ptr<uint32_t, Scope::local> notFixed; // Indices of the unfixed variables
        Ptr<uint32_t, Scope::state> pathEnd; // Other end of the path that starts at each node
        Ref<uint32_t, Scope::state> nFixed; // Number of unfixed variables
    };

}

#endif