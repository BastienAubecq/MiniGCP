#ifndef MINIGCP_NOT_EQUAL
#define MINIGCP_NOT_EQUAL

#include "MiniGCP/state/IntVar.hpp"

namespace cp {

    /**
     * @brief Constraint : x != y
     * @tparam X Type of x
     * @tparam Y Type of y
     */
    template<IntVar X, IntVar Y>
    class NotEqual {
    public:
        constexpr static bool parallel = false;

        NotEqual(X x, Y y) : x(x), y(y) {}

        DomainChange propagate(Context ctx) const {
            std::optional<int32_t> v = x[ctx].value(); 
            if (v) return y[ctx].remove(*v);
            v = y[ctx].value();
            if (v) return x[ctx].remove(*v);
            return DomainChange::none;
        }

    private:
        X x;
        Y y;
    };


    inline auto operator!=(IntVar auto x, IntVar auto y) {
        return NotEqual(x, y);
    }

}

#endif