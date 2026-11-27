#ifndef MINIGCP_CONTEXT
#define MINIGCP_CONTEXT

#include <array>
#include <cstdint>
#include "sycl/sycl.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/Scope.hpp"

namespace cp {

    /**
     * @brief Record a domain change event for a variable into the work-group's dirty event buffer
     */
    inline void recordVarEvent(uint32_t* varEvents, uint32_t varId, DomainEvent ev) {
        if (varEvents == nullptr || ev == DomainEvent::none) return;
        uint32_t wordIdx = varId >> 3;
        uint32_t shift = (varId & 7u) << 2;
        uint32_t mask = static_cast<uint32_t>(ev) << shift;
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>(varEvents[wordIdx]).fetch_or(mask);
    }

    /**
     * @brief Information about the current thread and where its data is located.
     * Data blocks are represented with an index instead of a direct reference because
     * their location depends on where the data is stored, which can change during execution.
     * To access the data, first get a reference to it by subscripting it with a context,
     * which adds the current data location to the index.
     */
    class Context {
    public:
        /**
         * @brief Information about the current thread
         */
        sycl::nd_item<1> item;

        /**
         * @brief Error stream
         */
        sycl::stream error;

        Context(sycl::nd_item<1> item, sycl::stream error, std::array<std::byte*, 3> data, std::byte* temporary, uint32_t* varEvents = nullptr)
            : item(item), error(error), data(data), temporary(temporary), varEvents(varEvents) {}

        /**
         * @brief Get a reference to a temporary array in this context
         */
        template<typename T>
        T* getTemporary() const {
            return reinterpret_cast<T*>(temporary);
        }

        /**
         * @brief Get the per-work-group dirty variable event buffer (4 bits per variable)
         */
        uint32_t* getVarEvents() const {
            return varEvents;
        }

        /**
         * @brief Get the data for a scope
         * @tparam scope The scope
         */
        template<Scope scope>
        std::byte* getData() const {
            return std::get<(int)scope>(data);
        }

        /**
         * @brief Get a data block
         * @tparam T Type of elements in the block
         * @tparam scope Scope of the block
         * @param index Index of the block
         * @return Pointer to the block
         */
        template<typename T, Scope scope>
        T* getData(uint32_t index) const {
            return reinterpret_cast<T*>(getData<scope>() + index);
        }

    private:
        std::array<std::byte*, 3> data;
        std::byte* temporary;
        uint32_t* varEvents;
    };

}

#endif