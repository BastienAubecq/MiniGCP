#ifndef MINIGCP_INT_BITSET
#define MINIGCP_INT_BITSET

#include <cstdint>
#include <limits>
#include <span>
#include <optional>
#include <algorithm>
#include "sycl/sycl.hpp"
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"

namespace cp {

    /**
     * @brief Host-side thread-unsafe reference to an integer variable bitset
     */
    class HostIntBitsetRef {
    public:
        explicit HostIntBitsetRef(int32_t* ptr) : ptr(ptr) {}

        int32_t min() const {
            return ptr[0];
        }

        int32_t max() const {
            return ptr[1];
        }

        int32_t size() const {
            return ptr[2];
        }

        std::optional<int32_t> value() const {
            int32_t m = min();
            if (m == max()) return m;
            return {};
        }

        bool contains(int32_t v) const {
            if (v < min() || v > max()) return false;
            return ptr[3 + v / 32] & ((int32_t)1 << v % 32);
        }

        DomainChange remove(int32_t v) const {
            int32_t min = ptr[0], max = ptr[1];
            if (!contains(v)) return DomainChange::none;
            ptr[3 + v / 32] &= ~((int32_t)1 << v % 32);
            if (v == min) updateMin(min, max);
            if (v == max) updateMax(min, max);
            return updateSize(1);
        }

        DomainChange fix(int32_t v) const {
            int32_t min = ptr[0], max = ptr[1];
            if (!contains(v)) return DomainChange::inconsistency;
            if (v == min && v == max) return DomainChange::none;

            // Clear bits
            int32_t index = 3 + v / 32;
            int32_t bit = (int32_t)1 << v % 32;
            ptr[index] = bit;
            for (int i = 3 + min / 32; i < index; i++) {
                ptr[i] = 0;
            }
            for (int i = index + 1; i <= 3 + max / 32; i++) {
                ptr[i] = 0;
            }

            // Update min, max, size
            ptr[0] = v;
            ptr[1] = v;
            ptr[2] = 1;

            return DomainChange::changed;
        }
        
        DomainChange removeBelow(int32_t v) const {
            int32_t min = ptr[0], max = ptr[1];
            if (v <= min) return DomainChange::none;
            if (v > max) return DomainChange::inconsistency;

            // Clear and count bits
            int32_t index = 3 + v / 32;
            int32_t bits = ((int32_t)1 << v % 32) - 1;
            int32_t removed = sycl::popcount(ptr[index] & bits);
            ptr[index] &= ~bits;
            for (int i = 3 + min / 32; i < index; i++) {
                removed += sycl::popcount(ptr[i]);
                ptr[i] = 0;
            }

            ptr[0] = v;
            updateMin(v, max);
            return updateSize(removed);
        }

        DomainChange removeAbove(int32_t v) const {
            int32_t min = ptr[0], max = ptr[1];
            if (v >= max) return DomainChange::none;
            if (v < min) return DomainChange::inconsistency;

            // Clear and count bits
            int32_t index = 3 + v / 32;
            int32_t bits = ~(((int32_t)1 << v % 32) - 1) << 1;
            int32_t removed = sycl::popcount(ptr[index] & bits);
            ptr[index] &= ~bits;
            for (int i = index + 1; i <= 3 + max / 32; i++) {
                removed += sycl::popcount(ptr[i]);
                ptr[i] = 0;
            }

            ptr[1] = v;
            updateMax(min, v);
            return updateSize(removed);
        }

    private:
        /**
         * @brief Pointer to the variable :
         * min (32 bits), max (32 bits), size (32 bits), bitset (multiple of 32 bits)
         */
        int32_t* ptr;

        /**
         * @brief Recompute min after removing some values
         */
        void updateMin(int32_t min, int32_t max) const {
            for (int32_t i = 3 + min / 32; i <= 3 + max / 32; i++) {
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
        void updateMax(int32_t min, int32_t max) const {
            for (int32_t i = 3 + max / 32; i >= 3 + min / 32; i--) {
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
     * @brief Device-side thread-safe reference to an integer variable bitset
     */
    class DeviceIntBitsetRef {
    public:
        explicit DeviceIntBitsetRef(int32_t* ptr) : ptr(ptr) {}

        int32_t min() const {
            return ptr[0];
        }

        int32_t max() const {
            return ptr[1];
        }

        int32_t size() const {
            return ptr[2];
        }

        std::optional<int32_t> value() const {
            int32_t m = min();
            if (m == max()) return m;
            return {};
        }

        bool contains(int32_t v) const {
            if (v < min() || v > max()) return false;
            return ptr[3 + v / 32] & ((int32_t)1 << v % 32);
        }

        DomainChange remove(int32_t v) const {
            int32_t min = ptr[0], max = ptr[1];
            if (v < min || v > max) return DomainChange::none;
            int32_t bit = (int32_t)1 << v % 32;
            int32_t prev = atomic(ptr[3 + v / 32]).fetch_and(~bit); // Clear bit
            if (!(prev & bit)) return DomainChange::none;

            // updateSize must be before updateMin/Max so that if 2 threads update the bitset concurrently,
            // one of them sees both updates in updateMin/Max thanks to the acq_rel atomic operation in updateSize.
            DomainChange change = updateSize(1, true);
            updateMin(min, max);
            updateMax(min, max);
            return change;
        }

        DomainChange fix(int32_t v) const {
            int32_t min = ptr[0], max = ptr[1];
            if (v == min && v == max) return DomainChange::none;
            if (v < min || v > max) return DomainChange::inconsistency;
            atomic(ptr[0]).fetch_max(v); // Update min
            atomic(ptr[1]).fetch_min(v); // Update max

            // Clear and count bits
            int32_t index = 3 + v / 32;
            int32_t bit = (int32_t)1 << v % 32;
            int32_t removed = sycl::popcount(atomic(ptr[index]).fetch_and(bit) & ~bit);
            for (int i = 3 + min / 32; i < index; i++) {
                removed += sycl::popcount(atomic(ptr[i]).exchange(0));
            }
            for (int i = index + 1; i <= 3 + max / 32; i++) {
                removed += sycl::popcount(atomic(ptr[i]).exchange(0));
            }

            return updateSize(removed);
        }

        DomainChange removeBelow(int32_t v) const {
            int32_t min = ptr[0], max = ptr[1];
            if (v <= min) return DomainChange::none;
            if (v > max) return DomainChange::inconsistency;

            // Clear and count bits
            int32_t index = 3 + v / 32;
            int32_t bits = ((int32_t)1 << v % 32) - 1;
            int32_t removed = sycl::popcount(atomic(ptr[index]).fetch_and(~bits) & bits);
            for (int i = 3 + min / 32; i < index; i++) {
                removed += sycl::popcount(atomic(ptr[i]).exchange(0));
            }

            DomainChange change = updateSize(removed, true);
            updateMin(v, max);
            return change;
        }

        DomainChange removeAbove(int32_t v) const {
            int32_t min = ptr[0], max = ptr[1];
            if (v >= max) return DomainChange::none;
            if (v < min) return DomainChange::inconsistency;
            atomic(ptr[1]).fetch_min(v); // Update max

            // Clear and count bits
            int32_t index = 3 + v / 32;
            int32_t bits = ~(((int32_t)1 << v % 32) - 1) << 1;
            int32_t removed = sycl::popcount(atomic(ptr[index]).fetch_and(~bits) & bits);
            for (int i = index + 1; i <= 3 + max / 32; i++) {
                removed += sycl::popcount(atomic(ptr[i]).exchange(0));
            }

            DomainChange change = updateSize(removed, true);
            updateMax(min, v);
            return change;
        }

        /*void print() const {
            printf("%d %d %d\n", ptr[0], ptr[1], ptr[2]);
            for (uint32_t i = ptr[0]; i <= ptr[1]; i++) {
                if (contains(i)) printf("1");
                else printf("0");
            }
            printf("\n");
        }*/

    private:
        using atomic = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>;

        int32_t* ptr;

        /**
         * @brief Recompute min after removing some values
         */
        void updateMin(int32_t min, int32_t max) const {
            for (int32_t i = 3 + min / 32; i <= 3 + max / 32; i++) {
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
        void updateMax(int32_t min, int32_t max) const {
            for (int32_t i = 3 + max / 32; i >= 3 + min / 32; i--) {
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


    /**
     * @brief Integer variable represented with a bitset
     */
    class IntBitset {
    public:
        explicit IntBitset(uint32_t index) : index(index) {}

        /**
         * @brief Allocate an integer variable with an initially full bitset domain starting at 0
         * @param allocator CP model allocator
         * @param size Size of the domain
         * @return The variable
         */
        IntBitset(Allocator& allocator, int32_t size);

        /**
         * @brief Allocate an integer variable with an arbitrary initial bitset domain starting at 0
         * @param allocator CP model allocator
         * @param values Values in the domain
         * @return The variable
         */
        IntBitset(Allocator& allocator, std::span<const int32_t> values);

        DeviceIntBitsetRef operator[](Context ctx) const {
            return DeviceIntBitsetRef(ctx.getData<int32_t, Scope::state>(index));
        }

        HostIntBitsetRef operator[](Allocator& allocator) const {
            return HostIntBitsetRef(allocator.getData<int32_t, Scope::state>(index));
        }

    private:
        uint32_t index;
    };


    /**
     * @brief Pointer to integer bitsets
     */
    class IntBitsetPtr {
    public:
        /**
         * @brief Allocate an array of integer variables with initially full bitset domains starting at 0
         * @param allocator CP model allocator
         * @param size Size of the domains
         * @param count Number of variables in the array
         */
        IntBitsetPtr(Allocator& allocator, int32_t size, uint32_t count) {
            if (size <= 0) throw std::invalid_argument("size must be strictly positive");
            int32_t nFull = size / 32;
            int32_t remaining = size % 32;
            uint32_t nWords = 3 + nFull + (remaining > 0);
            stride = nWords * sizeof(int32_t);
            index = allocator.allocate<int32_t, Scope::state>(nWords * count);
            int32_t* var = allocator.getData<int32_t, Scope::state>(index);

            // Add min, max, size
            var[0] = 0;
            var[1] = size - 1;
            var[2] = size;

            // Add bits
            for (int32_t i = 0; i < nFull; i++) {
                var[3 + i] = -1;
            }
            if (remaining > 0) {
                var[3 + nFull] = ((int32_t)1 << remaining) - 1;
            }

            // Copies
            for (uint32_t i = 1; i < count; i++) {
                std::copy(var, var + nWords, var + nWords * i);
            }
        }

        /**
         * @brief Allocate an array of integer variables with arbitrary initial bitset domains starting at 0
         * @param allocator CP model allocator
         * @param values Values in the domains
         * @param count Number of variables in the array
         */
        IntBitsetPtr(Allocator& allocator, std::span<const int32_t> values, uint32_t count) {
            int32_t max = std::numeric_limits<int32_t>::min();
            for (int32_t v : values) {
                if (v < 0) throw std::invalid_argument("values can't be negative");
                if (v > max) max = v;
            }
            int32_t nWords = max / 32 + 4;
            stride = nWords * sizeof(int32_t);
            index = allocator.allocate<int32_t, Scope::state>(nWords * count);
            int32_t* var = allocator.getData<int32_t, Scope::state>(index);

            // Add min, max, size
            var[0] = 0;
            var[1] = max;
            var[2] = values.size();

            // Add empty bits
            for (int32_t i = 3; i < nWords; i++) {
                var[i] = 0;
            }

            // Set bits corresponding to values
            for (int32_t v : values) {
                var[3 + v / 32] |= (int32_t)1 << v % 32;
            }
        }

        IntBitset operator[](uint32_t i) const {
            return IntBitset(index + i * stride);
        }

        IntBitset operator[](Context, uint32_t i) const {
            return (*this)[i];
        }

        IntBitset operator[](Allocator&, uint32_t i) const {
            return (*this)[i];
        }

    private:
        uint32_t index;
        uint32_t stride;
    };

    inline IntBitset::IntBitset(Allocator& allocator, int32_t size) :
        index(IntBitsetPtr(allocator, size, 1)[0].index) {}

    inline IntBitset::IntBitset(Allocator& allocator, std::span<const int32_t> values) :
        index(IntBitsetPtr(allocator, values, 1)[0].index) {}

}

#endif