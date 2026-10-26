#ifndef MINIGCP_ELEMENT_1D
#define MINIGCP_ELEMENT_1D

#include <algorithm>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVarBitsetIdx.hpp"
#include "MiniGCP/state/IntVarIntervalIdx.hpp"
#include "MiniGCP/state/ValueIdx.hpp"

namespace cp {

    /**
     * @brief Constraint : t[y] = z.
     * Bound consistency on z, domain consistency on y.
     */
    class Element1D {
    public:
        constexpr static bool parallel = false;

        Element1D(Allocator& allocator, ArrayIdx<Scope::global, int32_t> t, IntVarBitsetIdx y, IntVarIntervalIdx z) :
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
            int32_t i = ctx[min];
            int32_t j = ctx[max];

            int32_t zMin = ctx[z].min();
            while (i <= j && t[sorted[i]] < zMin) change |= ctx[y].remove(sorted[i++]);
            while (i <= j && !ctx[y].contains(sorted[i])) i++;
            if (i < n) change |= ctx[z].removeBelow(t[sorted[i]]);
            ctx[min] = i;

            int32_t zMax = ctx[z].max();
            while (j >= i && t[sorted[j]] > zMax) change |= ctx[y].remove(sorted[j--]);
            while (j >= i && !ctx[y].contains(sorted[j])) j--;
            if (j >= 0) change |= ctx[z].removeAbove(t[sorted[j]]);
            ctx[max] = j;

            return change;
        }

    private:
        ArrayIdx<Scope::global, int32_t> t;
        IntVarBitsetIdx y;
        IntVarIntervalIdx z;
        ArrayIdx<Scope::global, uint32_t> sorted; // Indices of the values in t in sorted order
        ValueIdx<Scope::state, uint32_t> min; // Index of the index of the minimum feasible value in t
        ValueIdx<Scope::state, uint32_t> max; // Index of the index of the maximum feasible value in t
    };


    /**
     * @brief Add a 1D element constraint to a model and return its result
     * @param model The model
     * @param t Array of values
     * @param y Index variable
     * @return The result
     */
    template<Constraint... Constraints>
    inline IntVarIntervalIdx element1D(Model<Constraints...>& model, ArrayIdx<Scope::global, int32_t> t, IntVarBitsetIdx y) {
        int32_t min = INT32_MAX;
        int32_t max = INT32_MIN;
        for (uint32_t i = 0; i < t.size(); i++) {
            int32_t v = model[t][i];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        IntVarIntervalIdx z = model.intVarInterval(min, max);
        model.constraint(Element1D(model, t, y, z));
        return z;
    }

}

#endif