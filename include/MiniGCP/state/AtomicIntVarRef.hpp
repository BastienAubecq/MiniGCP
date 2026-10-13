#ifndef MINIGCP_ATOMIC_INT_VAR_REF
#define MINIGCP_ATOMIC_INT_VAR_REF

#include <cstdint>
#include "sycl/sycl.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"

namespace cp {

    /**
     * @brief Same as IntVarRef but using atomic operations to make modification methods thread-safe
     */
    class AtomicIntVarRef {
    public:
        /**
         * @brief Get a reference to a variable in a state
         * @param idx Index of the variable
         * @param state The state
         * @return Reference to the variable
         */
        AtomicIntVarRef(IntVarIdx idx, std::byte* state) :
            ptr(reinterpret_cast<int32_t*>(state + idx.index)), offset(idx.offset) {}


        /**
         * @brief Get the minimum value in the domain
         */
        int32_t min() const {
            return relaxed_atomic(ptr[0]).load() + offset;
        }

        /**
         * @brief Get the maximum value in the domain
         */
        int32_t max() const {
            return relaxed_atomic(ptr[1]).load() + offset;
        }

        /**
         * @brief Get the number of values in the domain
         */
        int32_t size() const {
            return relaxed_atomic(ptr[2]).load();
        }
        
        /**
         * @brief Whether the domain contains a single value
         */
        bool fixed() const {
            return size() == 1;
        }

        /**
         * @brief Whether the domain contains a value
         * @param value The value
         */
        bool contains(int32_t value) const {
            if (value < min() || value > max()) return false;
            value -= offset;
            return consistent_atomic(ptr[3 + value / 32]).load() & ((int32_t)1 << value % 32);
        }

        
        /**
         * @brief Remove a value from the domain
         * @param value The value
         * @return Whether the domain was modified
         */
        DomainChange remove(int32_t value) const {
            if (value < min() || value > max()) return DomainChange::none;
            value -= offset;
            int32_t bit = (int32_t)1 << value % 32;
            int32_t prev = consistent_atomic(ptr[3 + value / 32]).fetch_and(~bit); // Clear bit
            if (!(prev & bit)) return DomainChange::none;
            updateMin();
            updateMax();
            return updateSize(1);
        }

        /**
         * @brief Fix the domain to a single value
         * @param value The value
         * @return Whether the domain was modified
         */
        DomainChange fix(int32_t value) const {
            value -= offset;
            int32_t min = relaxed_atomic(ptr[0]);
            int32_t max = relaxed_atomic(ptr[1]);
            if (value == min && value == max) return DomainChange::none;
            if (value < min || value > max) return DomainChange::inconsistency;
            relaxed_atomic(ptr[0]).fetch_max(value); // Update min
            relaxed_atomic(ptr[1]).fetch_min(value); // Update max

            // Clear and count bits
            int32_t index = 3 + value / 32;
            int32_t bit = (int32_t)1 << value % 32;
            int32_t removed = sycl::popcount(consistent_atomic(ptr[index]).fetch_and(bit) & ~bit);
            for (int i = 3 + min / 32; i < index; i++) {
                removed += sycl::popcount(consistent_atomic(ptr[i]).exchange(0));
            }
            for (int i = index + 1; i <= 3 + max / 32; i++) {
                removed += sycl::popcount(consistent_atomic(ptr[i]).exchange(0));
            }

            return updateSize(removed);
        }
        
        /**
         * @brief Remove all values below a value from the domain
         * @param value The value
         * @return Whether the domain was modified
         */
        DomainChange removeBelow(int32_t value) const {
            value -= offset;
            int32_t min = relaxed_atomic(ptr[0]);
            if (value <= min) return DomainChange::none;
            if (value > relaxed_atomic(ptr[1]).load()) return DomainChange::inconsistency;
            relaxed_atomic(ptr[0]).fetch_max(value); // Update min

            // Clear and count bits
            int32_t index = 3 + value / 32;
            int32_t bits = ((int32_t)1 << value % 32) - 1;
            int32_t removed = sycl::popcount(consistent_atomic(ptr[index]).fetch_and(~bits) & bits);
            for (int i = 3 + min / 32; i < index; i++) {
                removed += sycl::popcount(consistent_atomic(ptr[i]).exchange(0));
            }

            updateMin();
            return updateSize(removed);
        }

        /**
         * @brief Remove all values above a value from the domain
         * @param value The value
         * @return Whether the domain was modified
         */
        DomainChange removeAbove(int32_t value) const {
            value -= offset;
            int32_t max = relaxed_atomic(ptr[1]);
            if (value >= max) return DomainChange::none;
            if (value < relaxed_atomic(ptr[0]).load()) return DomainChange::inconsistency;
            relaxed_atomic(ptr[1]).fetch_min(value); // Update max

            // Clear and count bits
            int32_t index = 3 + value / 32;
            int32_t bits = ~(((int32_t)1 << value % 32) - 1) << 1;
            int32_t removed = sycl::popcount(consistent_atomic(ptr[index]).fetch_and(~bits) & bits);
            for (int i = index + 1; i <= 3 + max / 32; i++) {
                removed += sycl::popcount(consistent_atomic(ptr[i]).exchange(0));
            }

            updateMax();
            return updateSize(removed);
        }


        void print() const {
            printf("%d %d %d %b\n", ptr[0], ptr[1], ptr[2], ptr[3]);
        }


    private:
        using relaxed_atomic = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>;
        using consistent_atomic = sycl::atomic_ref<int32_t, sycl::memory_order::seq_cst, sycl::memory_scope::work_group>;

        int32_t* ptr;
        int32_t offset;

        /**
         * @brief Recompute min after removing some values
         */
        void updateMin() const {
            for (int32_t i = 3 + relaxed_atomic(ptr[0]).load() / 32; i <= 3 + relaxed_atomic(ptr[1]).load() / 32; i++) {
                int32_t word = consistent_atomic(ptr[i]).load();
                if (word != 0) {
                    relaxed_atomic(ptr[0]).fetch_max((i - 3) * 32 + sycl::ctz(word));
                    return;
                }
            }
        }

        /**
         * @brief Recompute max after removing some values
         */
        void updateMax() const {
            for (int32_t i = 3 + relaxed_atomic(ptr[1]).load() / 32; i >= 3 + relaxed_atomic(ptr[0]).load() / 32; i--) {
                int32_t word = consistent_atomic(ptr[i]).load();
                if (word != 0) {
                    relaxed_atomic(ptr[1]).fetch_min((i - 3) * 32 + 31 - sycl::clz(word));
                    return;
                }
            }
        }

        /**
         * @brief Decrease size after removing some values
         * @param removed Number of values removed
         * @return Whether an inconsistency was detected
         */
        DomainChange updateSize(int32_t removed) const {
            if (removed == 0) return DomainChange::none;
            int32_t prev = relaxed_atomic(ptr[2]).fetch_sub(removed);
            return prev == removed ? DomainChange::inconsistency : DomainChange::changed;
        }
    };
}

#endif