#ifndef MINIGCP_EVENT_REGISTRAR
#define MINIGCP_EVENT_REGISTRAR

#include <cstdint>
#include <vector>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/state/DomainChange.hpp"

namespace cp {

    /**
     * @brief Recorded subscription linking a (variable, event) pair to a constraint instance
     */
    struct EventSubscription {
        uint32_t varId;
        uint8_t eventSlot; // 0: fix, 1: min, 2: max, 3: domain
        uint32_t typeIdx;
        uint32_t instIdx;
    };

    /**
     * @brief Host-side registrar passed to Constraint::post() to subscribe to variable domain events
     */
    class EventRegistrar {
    public:
        EventRegistrar(Allocator& allocator, std::vector<EventSubscription>& subscriptions, uint32_t typeIdx, uint32_t instIdx) :
            alloc(allocator), subscriptions(subscriptions), typeIdx(typeIdx), instIdx(instIdx) {}

        Allocator& allocator() const {
            return alloc;
        }

        /**
         * @brief Subscribe the current constraint to one or more events on a concrete variable ID
         */
        void subscribe(uint32_t varId, DomainEvent ev) {
            uint8_t bits = static_cast<uint8_t>(ev);
            for (uint8_t slot = 0; slot < 4; ++slot) {
                if (bits & (1u << slot)) {
                    subscriptions.push_back({ varId, slot, typeIdx, instIdx });
                }
            }
        }

        template<typename V>
        void onFix(V const& x) {
            x.registerEvent(*this, DomainEvent::fix);
        }

        template<typename V>
        void onMinChange(V const& x) {
            x.registerEvent(*this, DomainEvent::min);
        }

        template<typename V>
        void onMaxChange(V const& x) {
            x.registerEvent(*this, DomainEvent::max);
        }

        template<typename V>
        void onBoundChange(V const& x) {
            x.registerEvent(*this, DomainEvent::bound);
        }

        template<typename V>
        void onDomainChange(V const& x) {
            x.registerEvent(*this, DomainEvent::domain);
        }

        template<typename Ptr>
        void onFix(Ptr const& x, uint32_t n) {
            for (uint32_t i = 0; i < n; ++i) {
                x[alloc, i].registerEvent(*this, DomainEvent::fix);
            }
        }

        template<typename Ptr>
        void onMinChange(Ptr const& x, uint32_t n) {
            for (uint32_t i = 0; i < n; ++i) {
                x[alloc, i].registerEvent(*this, DomainEvent::min);
            }
        }

        template<typename Ptr>
        void onMaxChange(Ptr const& x, uint32_t n) {
            for (uint32_t i = 0; i < n; ++i) {
                x[alloc, i].registerEvent(*this, DomainEvent::max);
            }
        }

        template<typename Ptr>
        void onBoundChange(Ptr const& x, uint32_t n) {
            for (uint32_t i = 0; i < n; ++i) {
                x[alloc, i].registerEvent(*this, DomainEvent::bound);
            }
        }

        template<typename Ptr>
        void onDomainChange(Ptr const& x, uint32_t n) {
            for (uint32_t i = 0; i < n; ++i) {
                x[alloc, i].registerEvent(*this, DomainEvent::domain);
            }
        }

    private:
        Allocator& alloc;
        std::vector<EventSubscription>& subscriptions;
        uint32_t typeIdx;
        uint32_t instIdx;
    };

}

#endif
