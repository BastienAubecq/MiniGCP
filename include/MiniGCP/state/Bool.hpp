#ifndef MINIGCP_BOOL
#define MINIGCP_BOOL

#include <cstdint>
#include <optional>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/EventRegistrar.hpp"

namespace cp {

    /**
     * @brief Host-side thread-unsafe boolean variable
     */
    class HostBool {
    public:
        int32_t min() const {
            return !(b & 1);
        }

        int32_t max() const {
            return b > 1;
        }

        int32_t size() const {
            return 1 + (b == 3);
        }

        std::optional<int32_t> value() const {
            if (b == 3) return {};
            return b - 1;
        }

        bool contains(int32_t v) const {
            if (v == 0) return b & 1;
            if (v == 1) return b & 2;
            return false;
        }

        DomainChange fix(int32_t v) {
            if (v < 0 || v > 1) return DomainChange::inconsistency;
            if (b == v + 1) return DomainChange::none;
            if (!(b & v + 1)) return DomainChange::inconsistency;
            b = v + 1;
            return DomainChange::changed;
        }

        DomainChange remove(int32_t v) {
            if (v < 0 || v > 1) return DomainChange::none;
            return fix(!v);
        }

        DomainChange removeBelow(int32_t v) {
            if (v <= 0) return DomainChange::none;
            if (v > 1) return DomainChange::inconsistency;
            return fix(1);
        }

        DomainChange removeAbove(int32_t v) {
            if (v >= 1) return DomainChange::none;
            if (v < 0) return DomainChange::inconsistency;
            return fix(0);
        }

    private:
        uint32_t b = 3;
    };


    /**
     * @brief Device-side storage for a boolean variable
     */
    struct DeviceBool {
        uint32_t b = 3;
    };

    /**
     * @brief Device-side thread-safe reference to a boolean variable
     */
    class DeviceBoolRef {
    public:
        DeviceBoolRef(DeviceBool* ptr, uint32_t varId = 0, uint32_t* varEvents = nullptr) :
            ptr(ptr), varId(varId), varEvents(varEvents) {}

        int32_t min() const {
            return !(ptr->b & 1);
        }

        int32_t max() const {
            return ptr->b > 1;
        }

        int32_t size() const {
            return 1 + (ptr->b == 3);
        }

        std::optional<int32_t> value() const {
            if (ptr->b == 3) return {};
            return ptr->b - 1;
        }

        bool contains(int32_t v) const {
            if (v == 0) return ptr->b & 1;
            if (v == 1) return ptr->b & 2;
            return false;
        }

        DomainChange fix(int32_t v) const {
            if (v < 0 || v > 1) return DomainChange::inconsistency;
            uint32_t mask = static_cast<uint32_t>(v + 1);
            uint32_t p = atomic(ptr->b).fetch_and(mask);
            if (p == mask) return DomainChange::none;
            if (!(p & mask)) return DomainChange::inconsistency;
            DomainEvent ev = DomainEvent::domain | DomainEvent::fix | (v == 1 ? DomainEvent::min : DomainEvent::max);
            recordVarEvent(varEvents, varId, ev);
            return DomainChange::changed;
        }

        DomainChange remove(int32_t v) const {
            if (v < 0 || v > 1) return DomainChange::none;
            return fix(!v);
        }

        DomainChange removeBelow(int32_t v) const {
            if (v <= 0) return DomainChange::none;
            if (v > 1) return DomainChange::inconsistency;
            return fix(1);
        }

        DomainChange removeAbove(int32_t v) const {
            if (v >= 1) return DomainChange::none;
            if (v < 0) return DomainChange::inconsistency;
            return fix(0);
        }

    private:
        using atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>;

        DeviceBool* ptr;
        uint32_t varId;
        uint32_t* varEvents;
    };


    /**
     * @brief Boolean variable
     */
    class Bool {
    public:
        explicit Bool(uint32_t index, uint32_t varId = 0) : index(index), varId(varId) {}
        
        /**
         * @brief Allocate a boolean variable
         * @param allocator CP model allocator
         * @return The variable
         */
        explicit Bool(Allocator& allocator) {
            index = allocator.allocate<HostBool, Scope::state>(1);
            varId = allocator.allocateVar(1);
            *allocator.getData<HostBool, Scope::state>(index) = HostBool();
        }

        DeviceBoolRef operator[](Context ctx) const {
            return DeviceBoolRef(ctx.getData<DeviceBool, Scope::state>(index), varId, ctx.getVarEvents());
        }

        HostBool& operator[](Allocator& allocator) const {
            return *allocator.getData<HostBool, Scope::state>(index);
        }

        void registerEvent(EventRegistrar& reg, DomainEvent ev) const {
            reg.subscribe(varId, ev);
        }

        uint32_t id() const {
            return varId;
        }

    private:
        uint32_t index;
        uint32_t varId;
    };


    /**
     * @brief Pointer to boolean variables
     */
    class BoolPtr {
    public:
        /**
         * @brief Allocate an array of boolean variables
         * @param allocator CP model allocator
         * @param count Number of variables in the array
         */
        BoolPtr(Allocator& allocator, uint32_t count) {
            index = allocator.allocate<HostBool, Scope::state>(count);
            firstVarId = allocator.allocateVar(count);
            HostBool* var = allocator.getData<HostBool, Scope::state>(index);
            for (uint32_t i = 0; i < count; i++) {
                var[i] = HostBool();
            }
        }

        Bool operator[](uint32_t i) const {
            return Bool(index + i * sizeof(HostBool), firstVarId + i);
        }

        Bool operator[](Context, uint32_t i) const {
            return (*this)[i];
        }

        Bool operator[](Allocator&, uint32_t i) const {
            return (*this)[i];
        }

    private:
        uint32_t index;
        uint32_t firstVarId;
    };

}

#endif