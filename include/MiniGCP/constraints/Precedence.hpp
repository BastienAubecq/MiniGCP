#ifndef MINIGCP_PRECEDENCE
#define MINIGCP_PRECEDENCE

#include <cstdint>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Ptr.hpp"

namespace cp {

    /**
     * @brief Precedence constraint: x + duration <= y
     * @tparam X Type of start variable x
     * @tparam Y Type of start variable y
     */
    template<IntVar X, IntVar Y>
    class Precedence {
    public:
        constexpr static bool parallel = false;

        Precedence(X x, int32_t duration, Y y) : x(x), duration(duration), y(y) {}

        Precedence(Allocator& allocator, X x, int32_t duration, Y y) : x(x), duration(duration), y(y) {
            y[allocator].removeBelow(x[allocator].min() + duration);
            x[allocator].removeAbove(y[allocator].max() - duration);
        }

        DomainChange propagate(Context ctx) const {
            DomainChange change = DomainChange::none;
            change |= y[ctx].removeBelow(x[ctx].min() + duration);
            change |= x[ctx].removeAbove(y[ctx].max() - duration);
            return change;
        }

    private:
        X x;
        int32_t duration;
        Y y;
    };

    /**
     * @brief Vectorized precedence constraint: s[pred[k]] + d[k] <= s[succ[k]] for k in 0..m-1.
     * Propagated in parallel by the work-group threads.
     * @tparam S Type of start variables pointer
     */
    template<IntVarPtr S>
    class PrecedenceVec {
    public:
        constexpr static bool parallel = true;

        PrecedenceVec(
            Allocator& allocator, S s,
            Ptr<uint32_t, Scope::global> pred,
            Ptr<uint32_t, Scope::global> succ,
            Ptr<int32_t, Scope::global> d,
            uint32_t m
        ) : s(s), pred(pred), succ(succ), d(d), m(m) {
            for (uint32_t k = 0; k < m; k++) {
                uint32_t u = pred[allocator, k];
                uint32_t v = succ[allocator, k];
                int32_t dur = d[allocator, k];
                s[allocator, v][allocator].removeBelow(s[allocator, u][allocator].min() + dur);
                s[allocator, u][allocator].removeAbove(s[allocator, v][allocator].max() - dur);
            }
        }

        DomainChange propagate(Context ctx) const {
            DomainChange change = DomainChange::none;
            uint32_t const* predPtr = pred[ctx];
            uint32_t const* succPtr = succ[ctx];
            int32_t const* dPtr = d[ctx];
            for (uint32_t k = ctx.item.get_local_id(0); k < m; k += ctx.item.get_local_range(0)) {
                uint32_t u = predPtr[k];
                uint32_t v = succPtr[k];
                int32_t dur = dPtr[k];
                change |= s[ctx, v][ctx].removeBelow(s[ctx, u][ctx].min() + dur);
                change |= s[ctx, u][ctx].removeAbove(s[ctx, v][ctx].max() - dur);
            }
            return change;
        }

    private:
        S s;
        Ptr<uint32_t, Scope::global> pred;
        Ptr<uint32_t, Scope::global> succ;
        Ptr<int32_t, Scope::global> d;
        uint32_t m;
    };

}

#endif
