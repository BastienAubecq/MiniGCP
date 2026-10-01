#ifndef MINIGCP_DOMAIN_CHANGE
#define MINIGCP_DOMAIN_CHANGE

namespace cp {

    /**
     * @brief Result flags when modifying the domain of a variable
     */
    enum class DomainChange : int {
        none = 0, // The domain wasn't changed
        changed = 1, // The domain was changed
        inconsistency = 2 // An inconsistency was detected
    };

    constexpr DomainChange operator|(DomainChange change1, DomainChange change2) {
        return static_cast<DomainChange>((int)change1 | (int)change2);
    }

    constexpr DomainChange operator&(DomainChange change1, DomainChange change2) {
        return static_cast<DomainChange>((int)change1 & (int)change2);
    }

    inline DomainChange& operator|=(DomainChange& change1, DomainChange change2) {
        change1 = change1 | change2;
        return change1;
    }

    inline DomainChange& operator&=(DomainChange& change1, DomainChange change2) {
        change1 = change1 & change2;
        return change1;
    }

}

#endif