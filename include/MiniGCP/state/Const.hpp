#ifndef MINIGCP_CONST
#define MINIGCP_CONST

#include <cstdint>
#include <optional>
#include <concepts>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"

namespace cp {

    class ConstRef {
    public:
        explicit ConstRef(int32_t c) : c(c) {}

        int32_t min() const {
            return c;
        }

        int32_t max() const {
            return c;
        }

        int32_t size() const {
            return 1;
        }

        std::optional<int32_t> value() const {
            return c;
        }

        bool contains(int32_t v) const {
            return v == c;
        }

        DomainChange remove(int32_t v) const {
            if (v == c) return DomainChange::inconsistency;
            return DomainChange::none;
        }

        DomainChange fix(int32_t v) const {
            if (v == c) return DomainChange::none;
            return DomainChange::inconsistency;
        }

        DomainChange removeBelow(int32_t v) const {
            if (v > c) return DomainChange::inconsistency;
            return DomainChange::none;
        }

        DomainChange removeAbove(int32_t v) const {
            if (v < c) return DomainChange::inconsistency;
            return DomainChange::none;
        }

    private:
        int32_t c;
    };


    /**
     * @brief Constant integer variable
     * @tparam C Constant type.
     */
    template<std::convertible_to<int32_t> C>
    class Const {
    public:
        Const(C c) : c(c) {}

        ConstRef operator[](Context ctx) const {
            return ConstRef(c);
        }

        ConstRef operator[](Allocator& allocator) const {
            return ConstRef(c);
        }
    
    private:
        [[no_unique_address]] C c;
    };


    /**
     * @brief Integer literal
     */
    template<int32_t c>
    class Lit {
    public:
        operator int32_t() const {
            return c;
        }

        Lit<-c> operator-() const {
            return {};
        }
    };

}

#endif