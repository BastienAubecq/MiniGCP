#ifndef MINIGCP_CIRCUIT
#define MINIGCP_CIRCUIT

#include <utility>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/utils.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"
#include "MiniGCP/state/ValueIdx.hpp"

namespace cp {

    /**
     * @brief Constraint : the graph represented by the successors doesn't contain any cycles that don't involve all nodes.
     * Sequential implementation.
     */
    class CircuitSeq {
    public:
        constexpr static bool parallel = false;

        CircuitSeq(Allocator& allocator, ArrayIdx<Scope::global, IntVarIdx> succ) :
            succ(succ),
            notFixed(allocator.array<Scope::local, uint32_t>(succ.size())),
            pathEnd(allocator.array<Scope::state, uint32_t>(succ.size())),
            nFixed(allocator.value<Scope::state, uint32_t>(0)) {
            for (uint32_t i = 0; i < succ.size(); i++) {
                allocator[notFixed][i] = i;
                allocator[pathEnd][i] = i;
                allocator[allocator[succ][i]].removeBelow(0);
                allocator[allocator[succ][i]].removeAbove(succ.size() - 1);
                allocator[allocator[succ][i]].remove(i);
            }
        }

        DomainChange propagate(Context ctx) const {
            DomainChange change = DomainChange::none;
            uint32_t* notFixed = ctx[this->notFixed];
            uint32_t* pathEnd = ctx[this->pathEnd];
            for (uint32_t i = ctx[nFixed]; i < succ.size(); i++) {
                uint32_t u = notFixed[i];
                if (ctx[ctx[succ][u]].fixed()) {
                    int32_t v = ctx[ctx[succ][u]].min();
                    std::swap(notFixed[ctx[nFixed]++], notFixed[i]);
                    if (ctx[nFixed] < succ.size() - 1) { // Prevent cycle
                        uint32_t src = pathEnd[u];
                        uint32_t dst = pathEnd[v];
                        pathEnd[src] = dst;
                        pathEnd[dst] = src;
                        change |= ctx[ctx[succ][dst]].remove(src);
                    }
                }
            }
            return change;
        }

    private:
        ArrayIdx<Scope::global, IntVarIdx> succ;
        ArrayIdx<Scope::local, uint32_t> notFixed; // Indices of the unfixed variables
        ArrayIdx<Scope::state, uint32_t> pathEnd; // Other end of the path that starts at each node
        ValueIdx<Scope::state, uint32_t> nFixed;
    };



    /**
     * @brief Constraint : the graph represented by the successors doesn't contain any cycles that don't involve all nodes.
     * Parallel implementation.
     */
    class Circuit {
    public:
        constexpr static bool parallel = true;

        Circuit(Allocator& allocator, ArrayIdx<Scope::global, IntVarIdx> succ) :
            succ(succ),
            notFixed(allocator.array<Scope::local, uint32_t>(succ.size())),
            pathEnd(allocator.array<Scope::state, uint32_t>(succ.size())),
            nFixed(allocator.value<Scope::state, uint32_t>(0)) {
            for (uint32_t i = 0; i < succ.size(); i++) {
                allocator[notFixed][i] = i;
                allocator[pathEnd][i] = i;
                allocator[allocator[succ][i]].removeBelow(0);
                allocator[allocator[succ][i]].removeAbove(succ.size() - 1);
                allocator[allocator[succ][i]].remove(i);
            }
        }

        DomainChange propagate(Context ctx) const {
            uint32_t nFixed = ctx[this->nFixed];
            uint32_t* notFixed = ctx[this->notFixed];
            uint32_t* pathEnd = ctx[this->pathEnd];
            DomainChange change = DomainChange::none;

            for (uint32_t b = nFixed; b < succ.size(); b += ctx.item.get_local_range(0)) {
                uint32_t i = b + ctx.item.get_local_id(0);
                std::optional<uint32_t> fixed;
                do {
                    // Broadcast any fixed variable
                    uint32_t u = notFixed[i];
                    bool broadcast = i >= nFixed && i < succ.size() && ctx[ctx[succ][u]].fixed();
                    fixed = groupBroadcastAny(ctx, broadcast ? std::optional(i) : std::nullopt);
                    
                    if (fixed) {
                        if (ctx.item.get_local_id(0) == 0) {
                            u = notFixed[*fixed];
                            uint32_t v = ctx[ctx[succ][u]].min();
                            std::swap(notFixed[nFixed], notFixed[*fixed]);
                            if (nFixed < succ.size() - 2) { // Prevent cycle
                                uint32_t src = pathEnd[u];
                                uint32_t dst = pathEnd[v];
                                pathEnd[src] = dst;
                                pathEnd[dst] = src;
                                change |= ctx[ctx[succ][dst]].remove(src);
                            }
                        }
                        nFixed++;
                        sycl::group_barrier(ctx.item.get_group());
                    }
                } while (fixed);
            }
            
            if (ctx.item.get_local_id(0) == 0) ctx[this->nFixed] = nFixed;
            return groupChange(ctx.item.get_group(), change);
        }

    private:
        ArrayIdx<Scope::global, IntVarIdx> succ;
        ArrayIdx<Scope::local, uint32_t> notFixed; // Indices of the unfixed variables
        ArrayIdx<Scope::state, uint32_t> pathEnd; // Other end of the path that starts at each node
        ValueIdx<Scope::state, uint32_t> nFixed;
    };

}

#endif