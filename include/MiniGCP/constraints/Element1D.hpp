#ifndef MINIGCP_ELEMENT_1D
#define MINIGCP_ELEMENT_1D

#include <algorithm>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"
#include "MiniGCP/state/ValueIdx.hpp"

namespace cp {

    /**
     * @brief Constraint : t[y] = z.
     * Bound consistency on z, domain consistency on y.
     */
    class Element1D {
    public:
        constexpr static bool parallel = false;

        Element1D(Allocator& allocator, ArrayIdx<Scope::global, int32_t> t, IntVarIdx y, IntVarIdx z) :
            t(t), y(y), z(z),
            sorted(allocator.array<Scope::global, uint32_t>(t.size())),
            min(allocator.value<Scope::state, uint32_t>(0)),
            max(allocator.value<Scope::state, uint32_t>(t.size() - 1)) {
            for (uint32_t i = 0; i < t.size(); i++) {
                allocator[sorted][i] = i;
            }
            std::sort(allocator[sorted], allocator[sorted] + t.size(), [&](uint32_t i, uint32_t j) {
                return allocator[t][i] < allocator[t][j];
            });
            allocator[y].removeBelow(0);
            allocator[y].removeAbove(t.size() - 1);
        }

        DomainChange propagate(Context ctx) const {
            int32_t* t = ctx[this->t];
            uint32_t n = this->t.size();
            uint32_t* sorted = ctx[this->sorted];
            DomainChange change = DomainChange::none;

            int32_t zMin = ctx[z].min();
            int32_t i;
            for (i = ctx[min]; i < n && t[sorted[i]] < zMin; i++) {
                change |= ctx[y].remove(sorted[i]);
            }
            while (i < n && !ctx[y].contains(sorted[i])) i++;
            if (i < n) change |= ctx[z].removeBelow(t[sorted[i]]);
            ctx[min] = i;

            int32_t zMax = ctx[z].max();
            for (i = ctx[max]; i >= 0 && t[sorted[i]] > zMax; i--) {
                change |= ctx[y].remove(sorted[i]);
            }
            while (i >= 0 && !ctx[y].contains(sorted[i])) i--;
            if (i >= 0) change |= ctx[z].removeAbove(t[sorted[i]]);
            ctx[max] = i;

            return change;
        }

    private:
        ArrayIdx<Scope::global, int32_t> t;
        IntVarIdx y;
        IntVarIdx z;
        ArrayIdx<Scope::global, uint32_t> sorted; // Indices of the values in t in sorted order
        ValueIdx<Scope::state, uint32_t> min; // Index of the index of the minimum feasible value in t
        ValueIdx<Scope::state, uint32_t> max; // Index of the index of the maximum feasible value in t
    };

}

#endif