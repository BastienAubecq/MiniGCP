#ifndef MINIGCP_ALLOCATOR
#define MINIGCP_ALLOCATOR

#include <cstdint>
#include <vector>
#include <span>
#include <array>
#include <stdexcept>
#include "MiniGCP/state/Scope.hpp"

namespace cp {

    /**
     * @brief Constraint programming model allocator.
     * It contains variables and data allocated by the propagators.
     */
    class Allocator {
    public:
        /**
         * @brief Allocate a temporary array.
         * All temporary arrays overlap and are stored in local memory.
         * They should be reset before each use and after calling functions that also use them.
         * Get a pointer to the temporary array with the `Context.getTemporary()` method.
         * @tparam T Type of elements in the array
         * @param size Number of elements in the array
         */
        template<typename T>
        void temporary(uint32_t size) {
            size_t bytes = size * sizeof(T);
            if (bytes + 4 > UINT32_MAX) {
                throw std::range_error("Allocated size exceeds 32-bit integer range");
            } 
            temporarySize = std::max(temporarySize, static_cast<uint32_t>(bytes));
        }

        /**
         * @brief Allocate a data block
         * @tparam T Type of elements in the block
         * @tparam scope Scope of the block
         * @param count Number of elements in the block
         * @param alignment Alignment of the block
         * @return Index of the block
         */
        template<typename T, Scope scope>
        uint32_t allocate(uint32_t count, size_t alignment = alignof(T)) {
            std::vector<std::byte>& d = std::get<(int)scope>(data);
            uint32_t index = d.size();
            size_t blockSize = count * sizeof(T);
            size_t mod = d.size() % alignment;
            if (mod > 0) { // Add padding to align new block
                blockSize += alignment - mod;
                index += alignment - mod;
            }
            if (d.size() + blockSize > UINT32_MAX) {
                throw std::range_error("Allocated size exceeds 32-bit integer range");
            }
            d.resize(d.size() + blockSize);
            return index;
        }

        /**
         * @brief Get the data for a scope
         * @tparam scope The scope
         */
        template<Scope scope>
        std::byte* getData() {
            return std::get<(int)scope>(data).data();
        }

        /**
         * @brief Get a data block
         * @tparam T Type of elements in the block
         * @tparam scope Scope of the block
         * @param index Index of the block
         * @return Pointer to the block
         */
        template<typename T, Scope scope>
        T* getData(uint32_t index) {
            return reinterpret_cast<T*>(getData<scope>() + index);
        }

        std::array<std::span<std::byte const>, 3> getData() const {
            return { data[0], data[1], data[2] };
        }

        uint32_t getTemporarySize() const {
            return temporarySize;
        }

    private:
        std::array<std::vector<std::byte>, 3> data; // A vector of bytes per scope
        uint32_t temporarySize = 0;
    };

};

#endif