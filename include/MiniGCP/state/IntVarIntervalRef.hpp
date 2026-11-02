#ifndef MINIGCP_INT_VAR_INTERVAL_REF
#define MINIGCP_INT_VAR_INTERVAL_REF

#include <cstdint>
#include "sycl/sycl.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVarIntervalIdx.hpp"

namespace cp {

    /**
     * @brief Reference to a variable represented with an interval.
     * Instances of this class shouldn't be stored because the location of the variable depends on
     * where the state is stored, which can change during execution.
     * Modification methods aren't thread-safe, use AtomicIntVarIntervalRef instead in device code if the
     * same variable can be modified from multiple threads at the same time.
     */
    class IntVarIntervalRef {
    public:
        /**
         * @brief Get a reference to a variable in a state
         * @param idx Index of the variable
         * @param state The state
         * @return Reference to the variable
         */
        IntVarIntervalRef(IntVarIntervalIdx idx, std::byte* state) :
            ptr(reinterpret_cast<int32_t*>(state + idx.index)), offset(idx.offset) {}


        /**
         * @brief Get the minimum value in the domain
         */
        int32_t min() const {
            return ptr[0] + offset;
        }

        /**
         * @brief Get the maximum value in the domain
         */
        int32_t max() const {
            return ptr[1] + offset;
        }

        /**
         * @brief Get the number of values in the domain
         */
        int32_t size() const {
            return max() - min() + 1;
        }

        /**
         * @brief Check if the domain contains a single value and, if so, return it
         */
        std::optional<int32_t> value() const {
            int32_t m = min();
            if (m == max()) return m;
            return {};
        }


        /**
         * @brief Fix the domain to a single value
         * @param v The value
         * @return Whether the domain was modified
         */
        DomainChange fix(int32_t v) const {
            if (v < min() || v > max()) return DomainChange::inconsistency;
            if (value()) return DomainChange::none;
            ptr[0] = v - offset;
            ptr[1] = v - offset;
            return DomainChange::changed;
        }

        /**
         * @brief Remove all values below a value from the domain
         * @param v The value
         * @return Whether the domain was modified
         */
        DomainChange removeBelow(int32_t v) const {
            if (v <= min()) return DomainChange::none;
            if (v > max()) return DomainChange::inconsistency;
            ptr[0] = v - offset;
            return DomainChange::changed;
        }

        /**
         * @brief Remove all values above a value from the domain
         * @param v The value
         * @return Whether the domain was modified
         */
        DomainChange removeAbove(int32_t v) const {
            if (v >= max()) return DomainChange::none;
            if (v < min()) return DomainChange::inconsistency;
            ptr[1] = v - offset;
            return DomainChange::changed;
        }

    private:
        /**
         * @brief Pointer to the variable :
         * min (32 bits), max (32 bits)
         */
        int32_t* ptr;

        /**
         * @brief Domain offset. It can be used to implement
         * constraints like x = y + c for free if c is a constant.
         */
        int32_t offset;
    };



    /**
     * @brief Same as IntVarIntervalRef but using atomic operations to make modification methods thread-safe
     */
    class AtomicIntVarIntervalRef {
    public:
        /**
         * @brief Get a reference to a variable in a state
         * @param idx Index of the variable
         * @param state The state
         * @return Reference to the variable
         */
        AtomicIntVarIntervalRef(IntVarIntervalIdx idx, std::byte* state) :
            ptr(reinterpret_cast<int32_t*>(state + idx.index)), offset(idx.offset) {}

        
        /**
         * @brief Get the minimum value in the domain
         */
        int32_t min() const {
            return ptr[0] + offset;
        }

        /**
         * @brief Get the maximum value in the domain
         */
        int32_t max() const {
            return ptr[1] + offset;
        }

        /**
         * @brief Get the number of values in the domain
         */
        int32_t size() const {
            return max() - min() + 1;
        }

        /**
         * @brief Check if the domain contains a single value and, if so, return it
         */
        std::optional<int32_t> value() const {
            int32_t m = min();
            if (m == max()) return m;
            return {};
        }


        /**
         * @brief Fix the domain to a single value
         * @param v The value
         * @return Whether the domain was modified
         */
        DomainChange fix(int32_t v) const {
            v -= offset;
            int32_t oldMin = atomic(ptr[0]).fetch_max(v);
            int32_t oldMax = atomic(ptr[1]).fetch_min(v);
            if (v < oldMin || v > oldMax) return DomainChange::inconsistency;
            if (v > oldMin || v < oldMax) return DomainChange::changed;
            return DomainChange::none;
        }

        /**
         * @brief Remove all values below a value from the domain
         * @param v The value
         * @return Whether the domain was modified
         */
        DomainChange removeBelow(int32_t v) const {
            v -= offset;
            bool changed = atomic(ptr[0]).fetch_max(v) < v;
            if (!changed) return DomainChange::none;
            return v <= atomic(ptr[1]).load() ? DomainChange::changed : DomainChange::inconsistency;    
        }

        /**
         * @brief Remove all values above a value from the domain
         * @param v The value
         * @return Whether the domain was modified
         */
        DomainChange removeAbove(int32_t v) const {
            v -= offset;
            bool changed = atomic(ptr[1]).fetch_min(v) > v;
            if (!changed) return DomainChange::none;
            return atomic(ptr[0]).load() <= v ? DomainChange::changed : DomainChange::inconsistency;    
        }

    private:
        using atomic = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>;

        int32_t* ptr;
        int32_t offset;
    };

}

#endif
