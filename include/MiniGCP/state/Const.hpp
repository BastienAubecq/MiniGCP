#ifndef MINIGCP_CONST
#define MINIGCP_CONST

#include <cstdint>
#include <optional>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"

namespace cp {

    template<int32_t cv>
    class ConstRef {
    public:
        int32_t min() const {
            return cv;
        }

        int32_t max() const {
            return cv;
        }

        int32_t size() const {
            return 1;
        }

        std::optional<int32_t> value() const {
            return cv;
        }

        bool contains(int32_t v) const {
            return v == cv;
        }

        DomainChange remove(int32_t v) const {
            if (v == cv) return DomainChange::inconsistency;
            return DomainChange::none;
        }

        DomainChange fix(int32_t v) const {
            if (v == cv) return DomainChange::none;
            return DomainChange::inconsistency;
        }

        DomainChange removeBelow(int32_t v) const {
            if (v > cv) return DomainChange::inconsistency;
            return DomainChange::none;
        }

        DomainChange removeAbove(int32_t v) const {
            if (v < cv) return DomainChange::inconsistency;
            return DomainChange::none;
        }
    };


    /**
     * @brief Constant integer variable
     * @param cv Constant value
     */
    template<int32_t cv>
    class Const {
    public:
        ConstRef<cv> operator[](Context ctx) const {
            return {};
        }

        ConstRef<cv> operator[](Allocator& allocator) const {
            return {};
        }
    };

}

#endif