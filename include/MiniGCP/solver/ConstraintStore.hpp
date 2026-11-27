#ifndef MINIGCP_CONSTRAINT_STORE
#define MINIGCP_CONSTRAINT_STORE

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <tuple>
#include <vector>
#include "sycl/sycl.hpp"
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/Constraint.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "MiniGCP/state/EventRegistrar.hpp"

namespace cp {

    /**
     * @brief Device-side constraints with event-driven wakeup
     * @tparam Constraints Types of constraints in the constraint store.
     */
    template<Constraint... Constraints>
    class ConstraintStore {
    public:
        ConstraintStore(
            sycl::queue& queue,
            std::tuple<std::span<Constraints const>...> constraintSpans,
            EventStoreLayout<sizeof...(Constraints)> layout,
            uint32_t numVars,
            std::span<EventSubscription const> subscriptions
        ) :
            constraints(sycl::malloc_device<Constraints>(std::get<std::span<Constraints const>>(constraintSpans).size(), queue)...),
            sizes{static_cast<uint32_t>(std::get<std::span<Constraints const>>(constraintSpans).size())...},
            varEventsWords(layout.varEventsWords),
            activeOffset(layout.activeOffset),
            totalActiveWords(layout.totalActiveWords),
            typeWordOffsets(layout.typeWordOffsets),
            typeWords(layout.typeWords) {
            (queue.memcpy(
                std::get<Constraints*>(this->constraints),
                std::get<std::span<Constraints const>>(constraintSpans).data(),
                std::get<std::span<Constraints const>>(constraintSpans).size() * sizeof(Constraints)
            ), ...);

            // Build deduplicated CSR subscription table indexed by (varId * 4 + eventSlot)
            uint32_t numKeys = numVars * 4u;
            std::vector<std::vector<uint32_t>> adj(numKeys);
            for (EventSubscription const& sub : subscriptions) {
                uint32_t key = (sub.varId << 2) + sub.eventSlot;
                uint32_t bitPos = layout.typeWordOffsets[sub.typeIdx] * 32u + sub.instIdx;
                if (key < numKeys) {
                    adj[key].push_back(bitPos);
                }
            }

            std::vector<uint32_t> hostOffsets(numKeys + 1u, 0u);
            uint32_t totalEdges = 0;
            for (uint32_t k = 0; k < numKeys; ++k) {
                auto& list = adj[k];
                std::sort(list.begin(), list.end());
                list.erase(std::unique(list.begin(), list.end()), list.end());
                hostOffsets[k] = totalEdges;
                totalEdges += static_cast<uint32_t>(list.size());
            }
            hostOffsets[numKeys] = totalEdges;

            std::vector<uint32_t> hostSubscribers;
            hostSubscribers.reserve(totalEdges);
            for (uint32_t k = 0; k < numKeys; ++k) {
                hostSubscribers.insert(hostSubscribers.end(), adj[k].begin(), adj[k].end());
            }

            csrOffsets = sycl::malloc_device<uint32_t>(hostOffsets.size(), queue);
            queue.memcpy(csrOffsets, hostOffsets.data(), hostOffsets.size() * sizeof(uint32_t));
            if (totalEdges > 0) {
                csrSubscribers = sycl::malloc_device<uint32_t>(totalEdges, queue);
                queue.memcpy(csrSubscribers, hostSubscribers.data(), totalEdges * sizeof(uint32_t));
            }
            queue.wait();
        }

        void destroy(sycl::queue& queue) const {
            (sycl::free(std::get<Constraints*>(constraints), queue), ...);
            if (csrOffsets != nullptr) sycl::free(csrOffsets, queue);
            if (csrSubscribers != nullptr) sycl::free(csrSubscribers, queue);
        }

        void invalidate() {
            ((std::get<Constraints*>(constraints) = nullptr), ...);
            csrOffsets = nullptr;
            csrSubscribers = nullptr;
        }


        /**
         * @brief Propagate active constraints until a fixpoint or inconsistency is reached
         * @param ctx Current context
         * @return Whether an inconsistency was detected
         */
        bool fixpoint(Context ctx) const {
            auto group = ctx.item.get_group();
            dispatchEvents(ctx);

            while (hasActive(ctx)) {
                DomainChange change = propagate(ctx);
                if (sycl::any_of_group(group, (change & DomainChange::inconsistency) == DomainChange::inconsistency)) {
                    clearPending(ctx);
                    return true;
                }
                dispatchEvents(ctx);
            }
            sycl::group_barrier(group);
            return false;
        }


    private:
        std::tuple<Constraints*...> constraints;
        std::array<uint32_t, sizeof...(Constraints)> sizes;
        uint32_t varEventsWords;
        uint32_t activeOffset;
        uint32_t totalActiveWords;
        std::array<uint32_t, sizeof...(Constraints)> typeWordOffsets;
        std::array<uint32_t, sizeof...(Constraints)> typeWords;
        uint32_t* csrOffsets = nullptr;
        uint32_t* csrSubscribers = nullptr;

        /**
         * @brief Translate dirty variable event flags into active constraint bits
         */
        void dispatchEvents(Context ctx) const {
            if (varEventsWords == 0) return;
            auto group = ctx.item.get_group();
            uint32_t tid = ctx.item.get_local_id(0);
            uint32_t groupSize = ctx.item.get_local_range(0);
            uint32_t* varEvents = ctx.getVarEvents();
            uint32_t* active = ctx.getData<uint32_t, Scope::local>(activeOffset);

            sycl::group_barrier(group);
            for (uint32_t w = tid; w < varEventsWords; w += groupSize) {
                uint32_t word = varEvents[w];
                if (word == 0) continue;
                varEvents[w] = 0;
                while (word != 0) {
                    uint32_t bit = sycl::ctz(word);
                    uint32_t subVar = bit >> 2;
                    uint32_t varId = (w << 3) + subVar;
                    uint32_t evMask = (word >> (subVar << 2)) & 0xFu;
                    word &= ~(0xFu << (subVar << 2));
                    while (evMask != 0) {
                        uint32_t slot = sycl::ctz(evMask);
                        evMask &= evMask - 1u;
                        uint32_t key = (varId << 2) + slot;
                        uint32_t begin = csrOffsets[key];
                        uint32_t end = csrOffsets[key + 1u];
                        for (uint32_t e = begin; e < end; ++e) {
                            uint32_t bitPos = csrSubscribers[e];
                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group>(active[bitPos >> 5])
                                .fetch_or(1u << (bitPos & 31u));
                        }
                    }
                }
            }
            sycl::group_barrier(group);
        }

        /**
         * @brief Check whether any constraint is currently scheduled
         */
        bool hasActive(Context ctx) const {
            auto group = ctx.item.get_group();
            uint32_t tid = ctx.item.get_local_id(0);
            uint32_t groupSize = ctx.item.get_local_range(0);
            uint32_t* active = ctx.getData<uint32_t, Scope::local>(activeOffset);
            bool any = false;
            for (uint32_t w = tid; w < totalActiveWords; w += groupSize) {
                if (active[w] != 0) {
                    any = true;
                    break;
                }
            }
            return sycl::any_of_group(group, any);
        }

        /**
         * @brief Clear all pending variable events and active constraint bits (called on failure)
         */
        void clearPending(Context ctx) const {
            auto group = ctx.item.get_group();
            uint32_t tid = ctx.item.get_local_id(0);
            uint32_t groupSize = ctx.item.get_local_range(0);
            uint32_t* varEvents = ctx.getVarEvents();
            uint32_t* active = ctx.getData<uint32_t, Scope::local>(activeOffset);
            if (varEvents != nullptr) {
                for (uint32_t w = tid; w < varEventsWords; w += groupSize) {
                    varEvents[w] = 0;
                }
            }
            for (uint32_t w = tid; w < totalActiveWords; w += groupSize) {
                active[w] = 0;
            }
            sycl::group_barrier(group);
        }

        /**
         * @brief Propagate all active constraints starting from list `i`
         * @param ctx Current context
         * @return Whether any variable's domain was modified or an inconsistency occurred
         */
        template<int i = 0>
        DomainChange propagate(Context ctx) const {
            if constexpr (i == sizeof...(Constraints)) {
                return DomainChange::none;
            }
            else {
                auto group = ctx.item.get_group();
                uint32_t tid = ctx.item.get_local_id(0);
                uint32_t groupSize = ctx.item.get_local_range(0);
                uint32_t* active = ctx.getData<uint32_t, Scope::local>(activeOffset);
                uint32_t wordOff = typeWordOffsets[i];
                uint32_t nWords = typeWords[i];
                DomainChange change = DomainChange::none;

                if constexpr (std::tuple_element_t<i, std::tuple<Constraints...>>::parallel) {
                    // Propagate each active constraint in parallel with the whole work-group
                    for (uint32_t w = 0; w < nWords; ++w) {
                        uint32_t word = active[wordOff + w];
                        if (word == 0) continue;
                        sycl::group_barrier(group);
                        if (tid == 0) active[wordOff + w] = 0;
                        sycl::group_barrier(group);
                        while (word != 0) {
                            uint32_t bit = sycl::ctz(word);
                            word &= word - 1u;
                            uint32_t j = (w << 5) + bit;
                            auto constraint = std::get<i>(constraints)[j];
                            change |= constraint.propagate(ctx);
                        }
                    }
                }
                else {
                    // Compact active constraint indices into temporary local memory, then propagate in parallel
                    uint32_t* activeList = ctx.getTemporary<uint32_t>();
                    uint32_t nActive = 0;
                    for (uint32_t base = 0; base < nWords; base += groupSize) {
                        uint32_t w = base + tid;
                        uint32_t word = (w < nWords) ? active[wordOff + w] : 0u;
                        if (w < nWords && word != 0) active[wordOff + w] = 0;
                        uint32_t cnt = sycl::popcount(word);
                        uint32_t pos = sycl::inclusive_scan_over_group(group, cnt, sycl::plus<uint32_t>());
                        uint32_t dst = nActive + pos - cnt;
                        while (word != 0) {
                            uint32_t bit = sycl::ctz(word);
                            word &= word - 1u;
                            activeList[dst++] = (w << 5) + bit;
                        }
                        nActive += sycl::group_broadcast(group, pos, groupSize - 1u);
                    }
                    sycl::group_barrier(group);

                    for (uint32_t k = tid; k < nActive; k += groupSize) {
                        uint32_t j = activeList[k];
                        auto constraint = std::get<i>(constraints)[j];
                        change |= constraint.propagate(ctx);
                    }
                    sycl::group_barrier(group);
                }

                if (sycl::any_of_group(group, (change & DomainChange::inconsistency) == DomainChange::inconsistency)) {
                    return DomainChange::inconsistency;
                }
                if (sycl::any_of_group(group, (change & DomainChange::changed) == DomainChange::changed)) {
                    change |= DomainChange::changed;
                }

                return change | propagate<i+1>(ctx);
            }
        }
    };

}

#endif