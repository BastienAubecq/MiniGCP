#ifndef MINIGCP_MODEL
#define MINIGCP_MODEL

#include <cstdint>
#include <vector>
#include <span>
#include <array>
#include <tuple>
#include <algorithm>
#include <concepts>
#include <stdexcept>
#include "MiniGCP/constraints/Constraint.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntVarBitsetIdx.hpp"
#include "MiniGCP/state/IntVarBitsetRef.hpp"
#include "MiniGCP/state/IntVarIntervalIdx.hpp"
#include "MiniGCP/state/IntVarIntervalRef.hpp"
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/ValueIdx.hpp"

namespace cp {

    /**
     * @brief Constraint programming model allocator.
     * It contains variables and data allocated by the propagators.
     */
    class Allocator {
    public:
        /**
         * @brief Allocate an array
         * @tparam scope Scope of the array
         * @tparam T Type of elements in the array
         * @param size Number of elements in the array
         * @param alignment Alignment of the array
         * @return Index of the array
         */
        template<Scope scope, typename T>
        ArrayIdx<scope, T> array(uint32_t size, size_t alignment = alignof(T)) {
            return { allocate<scope, T>(size, alignment), size };
        }

        /**
         * @brief Allocate an array
         * @tparam scope Scope of the array
         * @tparam T Type of elements in the array
         * @param values Initial values in the array
         * @param alignment Alignment of the array
         * @return Index of the array
         */
        template<Scope scope, typename T>
        ArrayIdx<scope, T> array(std::span<T const> values, size_t alignment = alignof(T)) {
            uint32_t size = static_cast<uint32_t>(values.size());
            uint32_t index = allocate<scope, T>(size, alignment);
            std::copy(values.begin(), values.end(), getData<scope, T>(index));
            return { index, size };
        }

        /**
         * @brief Allocate a value
         * @tparam scope Scope of the value
         * @tparam T Type of the value
         * @param values The value
         * @param alignment Alignment of the value
         * @return Index of the array
         */
        template<Scope scope, typename T>
        ValueIdx<scope, T> value(T value, size_t alignment = alignof(T)) {
            uint32_t index = allocate<scope, T>(1, alignment);
            *getData<scope, T>(index) = value;
            return { index };
        }

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
            if (bytes > UINT32_MAX) {
                throw std::range_error("Allocated size exceeds 32-bit integer range");
            } 
            temporarySize = std::max(temporarySize, static_cast<uint32_t>(bytes));
        }


        /**
         * @brief Allocate an integer variable with an initially full bitset domain starting at 0
         * @param size Size of the domain
         * @return Index of the variable
         */
        IntVarBitsetIdx intVarBitset(int32_t size) {
            if (size <= 0) throw std::invalid_argument("size must be strictly positive");
            int32_t nFull = size / 32;
            int32_t remaining = size % 32;
            uint32_t index = allocate<Scope::state, int32_t>(3 + nFull + (remaining > 0));
            int32_t* var = getData<Scope::state, int32_t>(index);

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

            return IntVarBitsetIdx(index, 0);
        }

        /**
         * @brief Allocate an integer variable with an initially full bitset domain
         * @param min Minimum value in the domain
         * @param max Maximum value in the domain
         * @return Index of the variable
         */
        IntVarBitsetIdx intVarBitset(int32_t min, int32_t max) {
            return intVarBitset(max - min + 1) + min;
        }

        /**
         * @brief Allocate an integer variable with an arbitrary initial bitset domain
         * @param values Values in the domain
         * @return Index of the variable
         */
        IntVarBitsetIdx intVarBitset(std::span<const int32_t> values) {
            // Find min and max
            int32_t min = INT32_MAX;
            int32_t max = INT32_MIN;
            for (int32_t v : values) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
            int32_t nWords = (max - min) / 32 + 1;
            uint32_t index = allocate<Scope::state, int32_t>(3 + nWords);
            int32_t* var = getData<Scope::state, int32_t>(index);

            // Add min, max, size
            var[0] = 0;
            var[1] = max - min;
            var[2] = values.size();

            // Add empty bits
            for (int32_t i = 0; i < nWords; i++) {
                var[3 + i] = 0;
            }

            // Set bits corresponding to values
            for (int32_t v : values) {
                v -= min;
                var[3 + v / 32] |= (int32_t)1 << v % 32;
            }

            return IntVarBitsetIdx(index, min);
        }

        /**
         * @brief Allocate an integer variable with an interval
         * @param min Minimum value in the domain
         * @param max Maximum value in the domain
         * @return Index of the variable
         */
        IntVarIntervalIdx intVarInterval(int32_t min, int32_t max) {
            if (max < min) throw std::invalid_argument("max must be greater than or equal to min");
            uint32_t index = allocate<Scope::state, int32_t>(2);
            int32_t* var = getData<Scope::state, int32_t>(index);
            var[0] = min;
            var[1] = max;
            return IntVarIntervalIdx(index, 0);
        }

        /**
         * @brief Allocate an integer variable with an interval domain starting at 0
         * @param size Size of the domain
         * @return Index of the variable
         */
        IntVarIntervalIdx intVarInterval(int32_t size) {
            if (size <= 0) throw std::invalid_argument("size must be strictly positive");
            return intVarInterval(0, size - 1);
        }


        /**
         * @brief Get a reference to a variable in the model
         * @param idx Index of the variable
         * @return Reference to the variable.
         * The reference is only valid until the next allocation.
         */
        IntVarBitsetRef operator[](IntVarBitsetIdx idx) {
            return IntVarBitsetRef(idx, std::get<(int)Scope::state>(data).data());
        }

        /**
         * @brief Get a reference to a variable in the model
         * @param idx Index of the variable
         * @return Reference to the variable.
         * The reference is only valid until the next allocation.
         */
        IntVarIntervalRef operator[](IntVarIntervalIdx idx) {
            return IntVarIntervalRef(idx, std::get<(int)Scope::state>(data).data());
        }

        /**
         * @brief Get a reference to an array in this context
         * @param idx Index of the array
         * @return Reference to the array
         * The reference is only valid until the next allocation.
         */
        template<Scope scope, typename T>
        T* operator[](ArrayIdx<scope, T> idx) {
            return idx.get(std::get<(int)scope>(data).data());
        }

        /**
         * @brief Get a reference to a value in this context
         * @param idx Index of the value
         * @return Reference to the value
         * The reference is only valid until the next allocation.
         */
        template<Scope scope, typename T>
        T& operator[](ValueIdx<scope, T> idx) {
            return idx.get(std::get<(int)scope>(data).data());
        }


        /**
         * @brief Get allocated data
         */
        std::array<std::span<std::byte const>, 3> getData() const {
            return { data[0], data[1], data[2] };
        }

        uint32_t getTemporarySize() const {
            return temporarySize;
        }

    private:
        std::array<std::vector<std::byte>, 3> data; // A vector of bytes per scope
        uint32_t temporarySize = 0;

        /**
         * @brief Allocate an array
         * @tparam scope Scope of the array
         * @tparam T Type of elements in the array
         * @param size Number of elements in the array
         * @param alignment Alignment of the array
         * @return Index of the array
         */
        template<Scope scope, typename T>
        uint32_t allocate(uint32_t size, size_t alignment = alignof(T)) {
            std::vector<std::byte>& d = std::get<(int)scope>(data);
            uint32_t index = d.size();
            size_t arraySize = size * sizeof(T);
            size_t mod = d.size() % alignment;
            if (mod > 0) { // Add padding to align new array
                arraySize += alignment - mod;
                index += alignment - mod;
            }
            if (d.size() + arraySize > UINT32_MAX) {
                throw std::range_error("Allocated size exceeds 32-bit integer range");
            }
            d.resize(d.size() + arraySize);
            return index;
        }

        /**
         * @brief Get a pointer to an array
         * @tparam scope Scope of the array
         * @tparam T Type of elements in the array
         * @param index Index of the array
         * @return Pointer to the array
         */
        template<Scope scope, typename T>
        T* getData(uint32_t index) {
            return reinterpret_cast<T*>(std::get<(int)scope>(data).data() + index);
        }
    };


    
    enum class ProblemType {
        satisfaction,
        minimization,
        maximization
    };



    /**
     * @brief Constraint programming model.
     * It contains variables, data allocated by the propagators, and constraints.
     * @tparam Constraints Types of constraints used in the model.
     */
    template<Constraint... Constraints>
    class Model : public Allocator {
    public:
        ProblemType type;
        IntVarIntervalIdx objective = {0, 0};
        ArrayIdx<Scope::global, IntVarBitsetIdx> output = {0, 0};

        Model(ProblemType type) : type(type) {}

        /**
         * @brief Add a constraint to the model
         * @tparam T Type of the constraint
         * @param constraint The constraint
         */
        template<typename T> requires (std::same_as<T, Constraints> || ...)
        void constraint(T constraint) {
            std::vector<T>& c = std::get<std::vector<T>>(constraints);
            if (c.size() + 1 > UINT32_MAX) {
                throw std::range_error("Allocated size exceeds 32-bit integer range");
            }
            c.push_back(constraint);
        }

        /**
         * @brief Get the constraints
         */
        std::tuple<std::span<const Constraints>...> getConstraints() const {
            if (((std::get<std::vector<Constraints>>(constraints).size() == 0) || ...)) {
                throw std::range_error("Empty constraint list. Remove unused constraint types from the model declaration.");
            }
            return std::make_tuple(std::span(std::get<std::vector<Constraints>>(constraints))...);
        }

    private:
        std::tuple<std::vector<Constraints>...> constraints; // A vector of constraints per constraint type
    };

}

#endif