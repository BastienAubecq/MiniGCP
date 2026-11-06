#ifndef MINIGCP_INT_INTERVAL
#define MINIGCP_INT_INTERVAL

#include <cstdint>
#include <optional>
#include "sycl/sycl.hpp"
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/Array.hpp"
#include "MiniGCP/state/DomainChange.hpp"

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
     * @brief Device-side thread-safe integer variable interval
     */
    class DeviceIntInterval {
    public:
        DeviceIntInterval(int32_t a, int32_t b) : a(a), b(b) {}

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

        DomainChange fix(int32_t v) {
            int32_t oldMin = atomic(a).fetch_max(v);
            int32_t oldMax = atomic(b).fetch_min(v);
            if (v < oldMin || v > oldMax) return DomainChange::inconsistency;
            if (v > oldMin || v < oldMax) return DomainChange::changed;
            return DomainChange::none;
        }

        DomainChange removeBelow(int32_t v) {
            bool changed = atomic(a).fetch_max(v) < v;
            if (!changed) return DomainChange::none;
            return v <= atomic(b).load() ? DomainChange::changed : DomainChange::inconsistency;    
        }

        DomainChange removeAbove(int32_t v) {
            bool changed = atomic(b).fetch_min(v) > v;
            if (!changed) return DomainChange::none;
            return atomic(a).load() <= v ? DomainChange::changed : DomainChange::inconsistency;    
        }

    private:
        using atomic = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>;

        int32_t a, b;
    };


    class IntIntervalArray;

    /**
     * @brief Integer variable represented with an interval
     */
    class IntInterval {
    public:
        explicit IntInterval(uint32_t index) : index(index) {}
        
        /**
         * @brief Allocate an integer variable with an interval domain
         * @param allocator CP model allocator
         * @param min Minimum value in the domain
         * @param max Maximum value in the domain
         * @return The variable
         */
        IntInterval(Allocator& allocator, int32_t min, int32_t max);
        
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
        IntInterval(Allocator& allocator) :
            IntInterval(allocator, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()) {}

        DeviceIntInterval& operator[](Context ctx) const {
            return *ctx.getData<DeviceIntInterval, Scope::state>(index);
        }

        HostIntInterval& operator[](Allocator& allocator) const {
            return *allocator.getData<HostIntInterval, Scope::state>(index);
        }

    private:
        uint32_t index;
    };


    /**
     * @brief IntInterval array that only stores the index of the first variable 
     */
    class IntIntervalArray {
    public:
        /**
         * @brief Allocate an array of integer variables with interval domains
         * @param allocator CP model allocator
         * @param min Minimum value in the domains
         * @param max Maximum value in the domains
         * @param count Number of variables in the array
         * @return The array
         */
        IntIntervalArray(Allocator& allocator, int32_t min, int32_t max, uint32_t count) : count(count) {
            if (max < min) throw std::invalid_argument("max must be greater than or equal to min");
            index = allocator.allocate<HostIntInterval, Scope::state>(count);
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
         * @return The array
         */
        IntIntervalArray(Allocator& allocator, int32_t size, uint32_t count) :
            IntIntervalArray(allocator, 0, size - 1, count) {}

        /**
         * @brief Allocate an array of integer variables with interval domains spanning all integers
         * @param allocator CP model allocator
         * @param count Number of variables in the array
         * @return The array
         */
        IntIntervalArray(Allocator& allocator, uint32_t count) :
            IntIntervalArray(allocator, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), count) {}

        IntInterval operator[](uint32_t i) const {
            return IntInterval(index + i * sizeof(HostIntInterval));
        }

        IntInterval operator[](Context, uint32_t i) const {
            return (*this)[i];
        }

        IntInterval operator[](Allocator&, uint32_t i) const {
            return (*this)[i];
        }

        uint32_t size() const {
            return count;
        }

    private:
        uint32_t index;
        uint32_t count;
    };

    inline IntInterval::IntInterval(Allocator& allocator, int32_t min, int32_t max) :
        index(IntIntervalArray(allocator, min, max, 1)[0].index) {}

}

#endif
