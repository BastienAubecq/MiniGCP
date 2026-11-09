#ifndef MINIGCP_ELEMENT_1D
#define MINIGCP_ELEMENT_1D

#include "MiniGCP/Model.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Ptr.hpp"
#include "MiniGCP/state/Ref.hpp"

namespace cp {

    /**
     * @brief Constraint : t[y] = z.
     * Bound consistency on z, domain consistency on y.
     * @tparam Y Type of y
     * @tparam Z Type of z
     */
    template<IntVar Y, IntVar Z>
    class Element1D {
    public:
        constexpr static bool parallel = false;

        Element1D(Allocator& allocator, Ptr<int32_t, Scope::global> t, uint32_t n, Y y, Z z) :
            t(t), n(n), y(y), z(z),
            sorted(allocator, n),
            min(allocator, 0),
            max(allocator, n - 1) {
            for (uint32_t i = 0; i < n; i++) {
                sorted[allocator, i] = i;
            }
            std::sort(sorted[allocator], sorted[allocator] + n, [&](uint32_t i, uint32_t j) {
                return t[allocator, i] < t[allocator, j];
            });
            y[allocator].removeBelow(0);
            y[allocator].removeAbove(n - 1);
            z[allocator].removeBelow(t[allocator, sorted[allocator, 0]]);
            z[allocator].removeAbove(t[allocator, sorted[allocator, n - 1]]);
        }

        DomainChange propagate(Context ctx) const {
            int32_t* t = this->t[ctx];
            uint32_t* sorted = this->sorted[ctx];
            DomainChange change = DomainChange::none;
            int32_t i = min[ctx];
            int32_t j = max[ctx];

            int32_t zMin = z[ctx].min();
            while (i <= j && t[sorted[i]] < zMin) change |= y[ctx].remove(sorted[i++]);
            while (i <= j && !y[ctx].contains(sorted[i])) i++;
            if (i < n) change |= z[ctx].removeBelow(t[sorted[i]]);
            min[ctx] = i;

            int32_t zMax = z[ctx].max();
            while (j >= i && t[sorted[j]] > zMax) change |= y[ctx].remove(sorted[j--]);
            while (j >= i && !y[ctx].contains(sorted[j])) j--;
            if (j >= 0) change |= z[ctx].removeAbove(t[sorted[j]]);
            max[ctx] = j;

            return change;
        }

    private:
        Ptr<int32_t, Scope::global> t;
        uint32_t n;
        Y y;
        Z z;
        Ptr<uint32_t, Scope::global> sorted; // Indices of the values in t in sorted order
        Ref<uint32_t, Scope::state> min; // Index of the index of the minimum feasible value in t
        Ref<uint32_t, Scope::state> max; // Index of the index of the maximum feasible value in t
    };


    /**
     * @brief Add a 1D element constraint to a model and return its result
     * @param model The model
     * @param t Array of values
     * @param y Index variable
     * @return The result
     */
    template<IntVar Y, Constraint... Constraints>
    inline IntInterval element1D(Model<Constraints...>& model, Ptr<int32_t, Scope::global> t, uint32_t n, Y y) {
        IntInterval z(model);
        model.constraint(Element1D(model, t, n, y, z));
        return z;
    }

}

#endif