#ifndef MINIGCP_DOMAIN_CHANGE
#define MINIGCP_DOMAIN_CHANGE

#include <cstdint>

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


    /**
     * @brief Fine-grained domain change event flags for constraint wakeup
     */
    enum class DomainEvent : uint8_t {
        none   = 0,
        fix    = 1 << 0, // Domain size became 1
        min    = 1 << 1, // Minimum value increased
        max    = 1 << 2, // Maximum value decreased
        domain = 1 << 3, // Any value was removed from the domain
        bound  = min | max
    };

    constexpr DomainEvent operator|(DomainEvent e1, DomainEvent e2) {
        return static_cast<DomainEvent>(static_cast<uint8_t>(e1) | static_cast<uint8_t>(e2));
    }

    constexpr DomainEvent operator&(DomainEvent e1, DomainEvent e2) {
        return static_cast<DomainEvent>(static_cast<uint8_t>(e1) & static_cast<uint8_t>(e2));
    }

    inline DomainEvent& operator|=(DomainEvent& e1, DomainEvent e2) {
        e1 = e1 | e2;
        return e1;
    }

    inline DomainEvent& operator&=(DomainEvent& e1, DomainEvent e2) {
        e1 = e1 & e2;
        return e1;
    }

    /**
     * @brief Swap min and max event flags (used by negating views)
     */
    constexpr DomainEvent swapMinMax(DomainEvent ev) {
        uint8_t bits = static_cast<uint8_t>(ev);
        uint8_t base = bits & static_cast<uint8_t>(~(static_cast<uint8_t>(DomainEvent::bound)));
        uint8_t newMin = (bits & static_cast<uint8_t>(DomainEvent::max)) ? static_cast<uint8_t>(DomainEvent::min) : 0u;
        uint8_t newMax = (bits & static_cast<uint8_t>(DomainEvent::min)) ? static_cast<uint8_t>(DomainEvent::max) : 0u;
        return static_cast<DomainEvent>(base | newMin | newMax);
    }

}

#endif