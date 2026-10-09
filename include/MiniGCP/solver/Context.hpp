#ifndef MINIGCP_CONTEXT
#define MINIGCP_CONTEXT

#include <cstdint>
#include <array>
#include "sycl/sycl.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"
#include "MiniGCP/state/AtomicIntVarRef.hpp"
#include "MiniGCP/state/ArrayIdx.hpp"
#include "MiniGCP/state/ValueIdx.hpp"

namespace cp {

    /**
     * @brief Information about the current thread and where its data is located
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

        Context(sycl::nd_item<1> item, sycl::stream error, std::array<std::byte*, 3> data, std::byte* temporary)
            : item(item), error(error), data(data), temporary(temporary) {}

        /**
         * @brief Get a reference to a variable in this context
         * @param idx Index of the variable
         * @return Reference to the variable
         */
        AtomicIntVarRef operator[](IntVarIdx idx) const {
            return AtomicIntVarRef(idx, getData<Scope::state>());
        }

        /**
         * @brief Get a reference to an array in this context
         * @param idx Index of the array
         * @return Reference to the array
         */
        template<Scope scope, typename T>
        T* operator[](ArrayIdx<scope, T> idx) const {
            return idx.get(getData<scope>());
        }

        /**
         * @brief Get a reference to a value in this context
         * @param idx Index of the value
         * @return Reference to the value
         */
        template<Scope scope, typename T>
        T& operator[](ValueIdx<scope, T> idx) const {
            return idx.get(getData<scope>());
        }

        /**
         * @brief Get a reference to a temporary array in this context
         */
        template<typename T>
        T* getTemporary() const {
            return reinterpret_cast<T*>(temporary);
        }

        /**
         * @brief Get the data for a scope
         * @tparam scope The scope
         */
        template<Scope scope>
        std::byte* getData() const {
            return std::get<(int)scope>(data);
        }

    private:
        std::array<std::byte*, 3> data;
        std::byte* temporary;
    };

}

#endif