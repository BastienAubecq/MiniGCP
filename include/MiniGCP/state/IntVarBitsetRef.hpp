#ifndef MINIGCP_INT_VAR_BITSET_REF
#define MINIGCP_INT_VAR_BITSET_REF

#include <cstdint>
#include <vector>
#include <span>
#include "sycl/sycl.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/IntVarBitsetIdx.hpp"

namespace cp {

    /**
     * @brief Reference to a variable represented with a bitset.
     * Instances of this class shouldn't be stored because the location of the variable depends on
     * where the state is stored, which can change during execution.
     * Modification methods aren't thread-safe, use AtomicIntVarBitsetRef instead in device code if the
     * same variable can be modified from multiple threads at the same time.
     */
    class IntVarBitsetRef {
    public:
        /**
         * @brief Get a reference to a variable in a state
         * @param idx Index of the variable
         * @param state The state
         * @return Reference to the variable
         */
        IntVarBitsetRef(IntVarBitsetIdx idx, std::byte* state) :
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
            return ptr[2];
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
            return ptr[3 + value / 32] & ((int32_t)1 << value % 32);
        }

        
        /**
         * @brief Remove a value from the domain
         * @param value The value
         * @return Whether the domain was modified
         */
        DomainChange remove(int32_t value) const {
            if (!contains(value)) return DomainChange::none;
            value -= offset;
            ptr[3 + value / 32] &= ~((int32_t)1 << value % 32);
            if (value == ptr[0]) updateMin();
            if (value == ptr[1]) updateMax();
            return updateSize(1);
        }

        /**
         * @brief Fix the domain to a single value
         * @param value The value
         * @return Whether the domain was modified
         */
        DomainChange fix(int32_t value) const {
            if (!contains(value)) return DomainChange::inconsistency;
            if (value == min() && value == max()) return DomainChange::none;
            value -= offset;

            // Clear bits
            int32_t index = 3 + value / 32;
            int32_t bit = (int32_t)1 << value % 32;
            ptr[index] = bit;
            for (int i = 3 + ptr[0] / 32; i < index; i++) {
                ptr[i] = 0;
            }
            for (int i = index + 1; i <= 3 + ptr[1] / 32; i++) {
                ptr[i] = 0;
            }

            // Update min, max, size
            ptr[0] = value;
            ptr[1] = value;
            ptr[2] = 1;

            return DomainChange::changed;
        }
        
        /**
         * @brief Remove all values below a value from the domain
         * @param value The value
         * @return Whether the domain was modified
         */
        DomainChange removeBelow(int32_t value) const {
            if (value <= min()) return DomainChange::none;
            if (value > max()) return DomainChange::inconsistency;
            value -= offset;

            // Clear and count bits
            int32_t index = 3 + value / 32;
            int32_t bits = ((int32_t)1 << value % 32) - 1;
            int32_t removed = sycl::popcount(ptr[index] & bits);
            ptr[index] &= ~bits;
            for (int i = 3 + ptr[0] / 32; i < index; i++) {
                removed += sycl::popcount(ptr[i]);
                ptr[i] = 0;
            }

            ptr[0] = value;
            updateMin();
            return updateSize(removed);
        }

        /**
         * @brief Remove all values above a value from the domain
         * @param value The value
         * @return Whether the domain was modified
         */
        DomainChange removeAbove(int32_t value) const {
            if (value >= max()) return DomainChange::none;
            if (value < min()) return DomainChange::inconsistency;
            value -= offset;

            // Clear and count bits
            int32_t index = 3 + value / 32;
            int32_t bits = ~(((int32_t)1 << value % 32) - 1) << 1;
            int32_t removed = sycl::popcount(ptr[index] & bits);
            ptr[index] &= ~bits;
            for (int i = index + 1; i <= 3 + ptr[1] / 32; i++) {
                removed += sycl::popcount(ptr[i]);
                ptr[i] = 0;
            }

            ptr[1] = value;
            updateMax();
            return updateSize(removed);
        }


    private:
        /**
         * @brief Pointer to the variable :
         * min (32 bits), max (32 bits), size (32 bits), bitset (multiple of 32 bits)
         */
        int32_t* ptr;

        /**
         * @brief Domain offset. It can be used to :
         * - Represent variables with domains that don't start at 0.
         * - Implement constraints like x = y + c for free if c is a constant.
         */
        int32_t offset;

        /**
         * @brief Recompute min after removing some values
         */
        void updateMin() const {
            for (int32_t i = 3 + ptr[0] / 32; i <= 3 + ptr[1] / 32; i++) {
                int32_t word = ptr[i];
                if (word != 0) {
                    ptr[0] = (i - 3) * 32 + sycl::ctz(word);
                    return;
                }
            }
        }

        /**
         * @brief Recompute max after removing some values
         */
        void updateMax() const {
            for (int32_t i = 3 + ptr[1] / 32; i >= 3 + ptr[0] / 32; i--) {
                int32_t word = ptr[i];
                if (word != 0) {
                    ptr[1] = (i - 3) * 32 + 31 - sycl::clz(word);
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
            ptr[2] -= removed;
            return ptr[2] == 0 ? DomainChange::inconsistency : DomainChange::changed;
        }
    };



    /**
     * @brief Same as IntVarBitsetRef but using atomic operations to make modification methods thread-safe
     */
    class AtomicIntVarBitsetRef {
    public:
        /**
         * @brief Get a reference to a variable in a state
         * @param idx Index of the variable
         * @param state The state
         * @return Reference to the variable
         */
        AtomicIntVarBitsetRef(IntVarBitsetIdx idx, std::byte* state) :
            ptr(reinterpret_cast<int32_t*>(state + idx.index)), offset(idx.offset) {}


        /**
         * @brief Get the minimum value in the domain
         */
        int32_t min() const {
            return atomic(ptr[0]).load() + offset;
        }

        /**
         * @brief Get the maximum value in the domain
         */
        int32_t max() const {
            return atomic(ptr[1]).load() + offset;
        }

        /**
         * @brief Get the number of values in the domain
         */
        int32_t size() const {
            return atomic(ptr[2]).load();
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
            return atomic(ptr[3 + value / 32]).load() & ((int32_t)1 << value % 32);
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
            int32_t prev = atomic(ptr[3 + value / 32]).fetch_and(~bit); // Clear bit
            if (!(prev & bit)) return DomainChange::none;

            // updateSize must be before updateMin/Max so that if 2 threads update the bitset concurrently,
            // one of them sees both updates in updateMin/Max thanks to the acq_rel atomic operation in updateSize.
            DomainChange change = updateSize(1, true);
            updateMin();
            updateMax();
            return change;
        }

        /**
         * @brief Fix the domain to a single value
         * @param value The value
         * @return Whether the domain was modified
         */
        DomainChange fix(int32_t value) const {
            value -= offset;
            int32_t min = atomic(ptr[0]).load();
            int32_t max = atomic(ptr[1]).load();
            if (value == min && value == max) return DomainChange::none;
            if (value < min || value > max) return DomainChange::inconsistency;
            atomic(ptr[0]).fetch_max(value); // Update min
            atomic(ptr[1]).fetch_min(value); // Update max

            // Clear and count bits
            int32_t index = 3 + value / 32;
            int32_t bit = (int32_t)1 << value % 32;
            int32_t removed = sycl::popcount(atomic(ptr[index]).fetch_and(bit) & ~bit);
            for (int i = 3 + min / 32; i < index; i++) {
                removed += sycl::popcount(atomic(ptr[i]).exchange(0));
            }
            for (int i = index + 1; i <= 3 + max / 32; i++) {
                removed += sycl::popcount(atomic(ptr[i]).exchange(0));
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
            int32_t min = atomic(ptr[0]).load();
            if (value <= min) return DomainChange::none;
            if (value > atomic(ptr[1]).load()) return DomainChange::inconsistency;
            atomic(ptr[0]).fetch_max(value); // Update min

            // Clear and count bits
            int32_t index = 3 + value / 32;
            int32_t bits = ((int32_t)1 << value % 32) - 1;
            int32_t removed = sycl::popcount(atomic(ptr[index]).fetch_and(~bits) & bits);
            for (int i = 3 + min / 32; i < index; i++) {
                removed += sycl::popcount(atomic(ptr[i]).exchange(0));
            }

            DomainChange change = updateSize(removed, true);
            updateMin();
            return change;
        }

        /**
         * @brief Remove all values above a value from the domain
         * @param value The value
         * @return Whether the domain was modified
         */
        DomainChange removeAbove(int32_t value) const {
            value -= offset;
            int32_t max = atomic(ptr[1]).load();
            if (value >= max) return DomainChange::none;
            if (value < atomic(ptr[0]).load()) return DomainChange::inconsistency;
            atomic(ptr[1]).fetch_min(value); // Update max

            // Clear and count bits
            int32_t index = 3 + value / 32;
            int32_t bits = ~(((int32_t)1 << value % 32) - 1) << 1;
            int32_t removed = sycl::popcount(atomic(ptr[index]).fetch_and(~bits) & bits);
            for (int i = index + 1; i <= 3 + max / 32; i++) {
                removed += sycl::popcount(atomic(ptr[i]).exchange(0));
            }

            DomainChange change = updateSize(removed, true);
            updateMax();
            return change;
        }


        /*void print() const {
            printf("%d %d %d %b\n", ptr[0], ptr[1], ptr[2], ptr[3]);
        }*/


    private:
        using atomic = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>;

        int32_t* ptr;
        int32_t offset;

        /**
         * @brief Recompute min after removing some values
         */
        void updateMin() const {
            for (int32_t i = 3 + atomic(ptr[0]).load() / 32; i <= 3 + atomic(ptr[1]).load() / 32; i++) {
                int32_t word = atomic(ptr[i]).load();
                if (word != 0) {
                    atomic(ptr[0]).fetch_max((i - 3) * 32 + sycl::ctz(word));
                    return;
                }
            }
        }

        /**
         * @brief Recompute max after removing some values
         */
        void updateMax() const {
            for (int32_t i = 3 + atomic(ptr[1]).load() / 32; i >= 3 + atomic(ptr[0]).load() / 32; i--) {
                int32_t word = atomic(ptr[i]).load();
                if (word != 0) {
                    atomic(ptr[1]).fetch_min((i - 3) * 32 + 31 - sycl::clz(word));
                    return;
                }
            }
        }

        /**
         * @brief Decrease size after removing some values
         * @param removed Number of values removed
         * @param sync Whether to use an acq_rel atomic operation
         * @return Whether an inconsistency was detected
         */
        DomainChange updateSize(int32_t removed, bool sync = false) const {
            if (removed == 0) return DomainChange::none;
            int32_t prev = atomic(ptr[2]).fetch_sub(removed, sync ? sycl::memory_order::acq_rel : sycl::memory_order::relaxed);
            return prev == removed ? DomainChange::inconsistency : DomainChange::changed;
        }
    };

}

#endif