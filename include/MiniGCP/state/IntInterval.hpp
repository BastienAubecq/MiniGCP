#ifndef MINIGCP_INT_INTERVAL
#define MINIGCP_INT_INTERVAL

#include <cstdint>
#include <optional>
#include "sycl/sycl.hpp"
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/EventRegistrar.hpp"

namespace cp {

    /**
     * @brief Host-side thread-unsafe integer variable interval
     */
    class HostIntInterval {
    public:
        HostIntInterval(int32_t a, int32_t b) : a(a), b(b) {}

        int32_t min() const {
            return a;
        }

        int32_t max() const {
            return b;
        }

        int32_t size() const {
            return max() - min() + 1;
        }

        std::optional<int32_t> value() const {
            int32_t m = min();
            if (m == max()) return m;
            return {};
        }

        bool contains(int32_t v) const {
            return v >= min() && v <= max();
        }

        DomainChange remove(int32_t v) {
            if (v == min()) return removeBelow(min() + 1);
            if (v == max()) return removeAbove(max() - 1);
            return DomainChange::none;
        }

        DomainChange fix(int32_t v) {
            if (v < min() || v > max()) return DomainChange::inconsistency;
            if (value()) return DomainChange::none;
            a = v;
            b = v;
            return DomainChange::changed;
        }

        DomainChange removeBelow(int32_t v) {
            if (v <= min()) return DomainChange::none;
            if (v > max()) return DomainChange::inconsistency;
            a = v;
            return DomainChange::changed;
        }

        DomainChange removeAbove(int32_t v) {
            if (v >= max()) return DomainChange::none;
            if (v < min()) return DomainChange::inconsistency;
            b = v;
            return DomainChange::changed;
        }

    private:
        int32_t a, b;
    };


    /**
     * @brief Device-side storage for an integer variable interval
     */
    struct DeviceIntInterval {
        int32_t a, b;
    };

    /**
     * @brief Device-side thread-safe reference to an integer variable interval
     */
    class DeviceIntIntervalRef {
    public:
        DeviceIntIntervalRef(DeviceIntInterval* ptr, uint32_t varId = 0, uint32_t* varEvents = nullptr) :
            ptr(ptr), varId(varId), varEvents(varEvents) {}

        int32_t min() const {
            return ptr->a;
        }

        int32_t max() const {
            return ptr->b;
        }

        int32_t size() const {
            return max() - min() + 1;
        }

        std::optional<int32_t> value() const {
            int32_t m = min();
            if (m == max()) return m;
            return {};
        }

        bool contains(int32_t v) const {
            return v >= min() && v <= max();
        }

        DomainChange remove(int32_t v) const {
            if (v == min()) return removeBelow(min() + 1);
            if (v == max()) return removeAbove(max() - 1);
            return DomainChange::none;
        }

        DomainChange fix(int32_t v) const {
            int32_t oldMin = atomic(ptr->a).fetch_max(v, sycl::memory_order::acq_rel);
            int32_t oldMax = atomic(ptr->b).fetch_min(v, sycl::memory_order::acq_rel);
            if (v < oldMin || v > oldMax) return DomainChange::inconsistency;
            if (v > oldMin || v < oldMax) {
                DomainEvent ev = DomainEvent::domain | DomainEvent::fix;
                if (v > oldMin) ev |= DomainEvent::min;
                if (v < oldMax) ev |= DomainEvent::max;
                recordVarEvent(varEvents, varId, ev);
                return DomainChange::changed;
            }
            return DomainChange::none;
        }

        DomainChange removeBelow(int32_t v) const {
            bool changed = atomic(ptr->a).fetch_max(v, sycl::memory_order::acq_rel) < v;
            if (!changed) return DomainChange::none;
            int32_t curMax = atomic(ptr->b).load(sycl::memory_order::acquire);
            if (v > curMax) return DomainChange::inconsistency;
            DomainEvent ev = DomainEvent::domain | DomainEvent::min;
            if (v == curMax) ev |= DomainEvent::fix;
            recordVarEvent(varEvents, varId, ev);
            return DomainChange::changed;
        }

        DomainChange removeAbove(int32_t v) const {
            bool changed = atomic(ptr->b).fetch_min(v, sycl::memory_order::acq_rel) > v;
            if (!changed) return DomainChange::none;
            int32_t curMin = atomic(ptr->a).load(sycl::memory_order::acquire);
            if (curMin > v) return DomainChange::inconsistency;
            DomainEvent ev = DomainEvent::domain | DomainEvent::max;
            if (curMin == v) ev |= DomainEvent::fix;
            recordVarEvent(varEvents, varId, ev);
            return DomainChange::changed;
        }

    private:
        using atomic = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>;

        DeviceIntInterval* ptr;
        uint32_t varId;
        uint32_t* varEvents;
    };


    /**
     * @brief Integer variable represented with an interval
     */
    class IntInterval {
    public:
        explicit IntInterval(uint32_t index, uint32_t varId = 0) : index(index), varId(varId) {}
        
        /**
         * @brief Allocate an integer variable with an interval domain
         * @param allocator CP model allocator
         * @param min Minimum value in the domain
         * @param max Maximum value in the domain
         * @return The variable
         */
        IntInterval(Allocator& allocator, int32_t min, int32_t max) {
            if (max < min) throw std::invalid_argument("max must be greater than or equal to min");
            index = allocator.allocate<HostIntInterval, Scope::state>(1);
            varId = allocator.allocateVar(1);
            *allocator.getData<HostIntInterval, Scope::state>(index) = HostIntInterval(min, max);
        }
        
        /**
         * @brief Allocate an integer variable with an interval domain starting at 0
         * @param allocator CP model allocator
         * @param size Number of values in the domain
         * @return The variable
         */
        IntInterval(Allocator& allocator, int32_t size) :
            IntInterval(allocator, 0, size - 1) {}

        /**
         * @brief Allocate an integer variable with an interval domain spanning all integers
         * @param allocator CP model allocator
         * @return The variable
         */
        explicit IntInterval(Allocator& allocator) :
            IntInterval(allocator, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()) {}

        DeviceIntIntervalRef operator[](Context ctx) const {
            return DeviceIntIntervalRef(ctx.getData<DeviceIntInterval, Scope::state>(index), varId, ctx.getVarEvents());
        }

        HostIntInterval& operator[](Allocator& allocator) const {
            return *allocator.getData<HostIntInterval, Scope::state>(index);
        }

        void registerEvent(EventRegistrar& reg, DomainEvent ev) const {
            reg.subscribe(varId, ev);
        }

        uint32_t id() const {
            return varId;
        }

    public:
        uint32_t index;
        uint32_t varId;
    };


    /**
     * @brief Pointer to integer intervals
     */
    class IntIntervalPtr {
    public:
        /**
         * @brief Allocate an array of integer variables with interval domains
         * @param allocator CP model allocator
         * @param min Minimum value in the domains
         * @param max Maximum value in the domains
         * @param count Number of variables in the array
         */
        IntIntervalPtr(Allocator& allocator, int32_t min, int32_t max, uint32_t count) {
            if (max < min) throw std::invalid_argument("max must be greater than or equal to min");
            index = allocator.allocate<HostIntInterval, Scope::state>(count);
            firstVarId = allocator.allocateVar(count);
            HostIntInterval* var = allocator.getData<HostIntInterval, Scope::state>(index);
            for (uint32_t i = 0; i < count; i++) {
                var[i] = HostIntInterval(min, max);
            }
        }

        /**
         * @brief Allocate an array of integer variables with interval domains starting at 0
         * @param allocator CP model allocator
         * @param size Number of values in the domains
         * @param count Number of variables in the array
         */
        IntIntervalPtr(Allocator& allocator, int32_t size, uint32_t count) :
            IntIntervalPtr(allocator, 0, size - 1, count) {}

        /**
         * @brief Allocate an array of integer variables with interval domains spanning all integers
         * @param allocator CP model allocator
         * @param count Number of variables in the array
         */
        IntIntervalPtr(Allocator& allocator, uint32_t count) :
            IntIntervalPtr(allocator, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), count) {}

        IntInterval operator[](uint32_t i) const {
            return IntInterval(index + i * sizeof(HostIntInterval), firstVarId + i);
        }

        IntInterval operator[](Context, uint32_t i) const {
            return (*this)[i];
        }

        IntInterval operator[](Allocator&, uint32_t i) const {
            return (*this)[i];
        }

    private:
        uint32_t index;
        uint32_t firstVarId;
    };

}

#endif
