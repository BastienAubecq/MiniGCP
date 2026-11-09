#ifndef MINIGCP_BOOL
#define MINIGCP_BOOL

#include <cstdint>
#include <optional>
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"

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
     * @brief Device-side thread-safe boolean variable
     */
    class DeviceBool {
    public:
        int32_t min() const {
            return !(b & 1);
        }

        int32_t max() const {
            return b & 2;
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
            uint32_t p = atomic(b).fetch_and(v + 1);
            if (p == v + 1) return DomainChange::none;
            if (!(p & v + 1)) return DomainChange::inconsistency;
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
        using atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>;

        uint32_t b = 3;
    };


    /**
     * @brief Boolean variable
     */
    class Bool {
    public:
        explicit Bool(uint32_t index) : index(index) {}
        
        /**
         * @brief Allocate an boolean variable
         * @param allocator CP model allocator
         * @return The variable
         */
        explicit Bool(Allocator& allocator) {
            index = allocator.allocate<HostBool, Scope::state>(1);
            *allocator.getData<HostBool, Scope::state>(index) = HostBool();
        }

        DeviceBool& operator[](Context ctx) const {
            return *ctx.getData<DeviceBool, Scope::state>(index);
        }

        HostBool& operator[](Allocator& allocator) const {
            return *allocator.getData<HostBool, Scope::state>(index);
        }

    private:
        uint32_t index;
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
            HostBool* var = allocator.getData<HostBool, Scope::state>(index);
            for (uint32_t i = 0; i < count; i++) {
                var[i] = HostBool();
            }
        }

        Bool operator[](uint32_t i) const {
            return Bool(index + i * sizeof(HostBool));
        }

        Bool operator[](Context, uint32_t i) const {
            return (*this)[i];
        }

        Bool operator[](Allocator&, uint32_t i) const {
            return (*this)[i];
        }

    private:
        uint32_t index;
    };

}

#endif