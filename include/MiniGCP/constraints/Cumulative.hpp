#ifndef MINIGCP_CUMULATIVE
#define MINIGCP_CUMULATIVE

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include "sycl/sycl.hpp"
#include "MiniGCP/Allocator.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntVar.hpp"
#include "MiniGCP/state/Ptr.hpp"

namespace cp {

    /**
     * @brief Cumulative constraint using Time-Table filtering (Gay, Hartert, Schaus, CP 2015)
     * and the event-point Profile structure (Gingras & Quimper, IJCAI 2016).
     * Sequential implementation.
     *
     * Enforces that at any time t: sum_{i : S_i <= t < S_i + p_i} h_i <= capacity.
     * @tparam S Type of the start time variables pointer
     */
    template<IntVarPtr S>
    class CumulativeSeq {
    public:
        constexpr static bool parallel = false;

        CumulativeSeq(
            Allocator& allocator, S s,
            Ptr<int32_t, Scope::global> p,
            Ptr<int32_t, Scope::global> h,
            uint32_t n, int32_t capacity
        ) : s(s), p(p), h(h), n(n), capacity(capacity),
            maxEvents(std::bit_ceil(std::max<uint32_t>(2, 2 * n))) {
            allocator.temporary<int32_t>(4 * maxEvents + 4);
        }

        DomainChange propagate(Context ctx) const {
            int32_t* tmp = ctx.getTemporary<int32_t>();
            uint32_t const offDelta  = maxEvents;
            uint32_t const offProfT  = 2 * maxEvents;
            uint32_t const offProfH  = 3 * maxEvents;

            int32_t const* pPtr = p[ctx];
            int32_t const* hPtr = h[ctx];

            // 1. Collect mandatory parts [lst_i, ect_i[ where lst_i < ect_i
            uint32_t nEvents = 0;
            for (uint32_t i = 0; i < n; i++) {
                int32_t dur = pPtr[i];
                int32_t height = hPtr[i];
                if (dur <= 0 || height <= 0) continue;
                if (height > capacity) return DomainChange::inconsistency;

                auto& var = s[ctx, i][ctx];
                int32_t est = var.min();
                int32_t lst = var.max();
                int32_t ect = est + dur;
                if (lst < ect) {
                    tmp[nEvents]            = lst;
                    tmp[offDelta + nEvents] = height;
                    nEvents++;
                    tmp[nEvents]            = ect;
                    tmp[offDelta + nEvents] = -height;
                    nEvents++;
                }
            }

            if (nEvents == 0) {
                return DomainChange::none;
            }

            // 2. Sort event points by time (insertion sort on local array)
            for (uint32_t i = 1; i < nEvents; i++) {
                int32_t tVal = tmp[i];
                int32_t dVal = tmp[offDelta + i];
                int32_t j = static_cast<int32_t>(i) - 1;
                while (j >= 0 && tmp[j] > tVal) {
                    tmp[j + 1]            = tmp[j];
                    tmp[offDelta + j + 1] = tmp[offDelta + j];
                    j--;
                }
                tmp[j + 1]            = tVal;
                tmp[offDelta + j + 1] = dVal;
            }

            // 3. Build Quimper's Profile via prefix sum and compaction + Overload check
            uint32_t m = 0;
            int32_t runningHeight = 0;
            for (uint32_t e = 0; e < nEvents; e++) {
                runningHeight += tmp[offDelta + e];
                if (e + 1 == nEvents || tmp[e] != tmp[e + 1]) {
                    if (runningHeight > capacity) {
                        return DomainChange::inconsistency;
                    }
                    tmp[offProfT + m] = tmp[e];
                    tmp[offProfH + m] = runningHeight;
                    m++;
                }
            }

            uint32_t numRects = m - 1;
            if (numRects == 0) {
                return DomainChange::none;
            }

            // 4. Filter unfixed tasks against the profile (Gay et al., Algorithm 2)
            DomainChange change = DomainChange::none;
            for (uint32_t i = 0; i < n; i++) {
                int32_t dur = pPtr[i];
                int32_t height = hPtr[i];
                if (dur <= 0 || height <= 0) continue;

                auto& var = s[ctx, i][ctx];
                int32_t est = var.min();
                int32_t lst = var.max();
                if (est == lst) continue;

                int32_t ect = est + dur;
                int32_t lct = lst + dur;
                bool hasMand = (lst < ect);
                int32_t maxAllowed = capacity - height;

                // Sweep left-to-right to filter lower bound (est)
                if (est < tmp[offProfT + numRects]) {
                    uint32_t lo = 0, hi = numRects - 1;
                    while (lo < hi) {
                        uint32_t mid = (lo + hi) >> 1;
                        if (tmp[offProfT + mid + 1] <= est) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    uint32_t j1 = lo;
                    int32_t sStar = est;
                    while (j1 < numRects && sStar <= lst && tmp[offProfT + j1] < sStar + dur) {
                        int32_t hOther = (hasMand && tmp[offProfT + j1] >= lst && tmp[offProfT + j1 + 1] <= ect)
                                         ? (tmp[offProfH + j1] - height)
                                         : tmp[offProfH + j1];
                        if (hOther > maxAllowed) {
                            sStar = tmp[offProfT + j1 + 1];
                        }
                        j1++;
                    }
                    if (sStar > est) {
                        change |= var.removeBelow(sStar);
                        if ((change & DomainChange::inconsistency) == DomainChange::inconsistency) {
                            return DomainChange::inconsistency;
                        }
                    }
                }

                // Sweep right-to-left to filter upper bound (lct)
                if (lct > tmp[offProfT]) {
                    uint32_t rlo = 0, rhi = numRects - 1;
                    while (rlo < rhi) {
                        uint32_t mid = (rlo + rhi + 1) >> 1;
                        if (tmp[offProfT + mid] < lct) {
                            rlo = mid;
                        } else {
                            rhi = mid - 1;
                        }
                    }
                    int32_t j2 = static_cast<int32_t>(rlo);
                    int32_t eStar = lct;
                    while (j2 >= 0 && eStar >= ect && tmp[offProfT + j2 + 1] > eStar - dur) {
                        int32_t hOther = (hasMand && tmp[offProfT + j2] >= lst && tmp[offProfT + j2 + 1] <= ect)
                                         ? (tmp[offProfH + j2] - height)
                                         : tmp[offProfH + j2];
                        if (hOther > maxAllowed) {
                            eStar = tmp[offProfT + j2];
                        }
                        j2--;
                    }
                    if (eStar < lct) {
                        change |= var.removeAbove(eStar - dur);
                        if ((change & DomainChange::inconsistency) == DomainChange::inconsistency) {
                            return DomainChange::inconsistency;
                        }
                    }
                }
            }

            return change;
        }

    private:
        S s;
        Ptr<int32_t, Scope::global> p;
        Ptr<int32_t, Scope::global> h;
        uint32_t n;
        int32_t capacity;
        uint32_t maxEvents;
    };


    /**
     * @brief Work-group parallel Cumulative constraint using Time-Table filtering
     * (Gay, Hartert, Schaus, CP 2015) and the event-point Profile structure
     * (Gingras & Quimper, IJCAI 2016).
     *
     * Parallelizes:
     * 1. Mandatory part event collection via work-group prefix scan,
     * 2. Time-point sorting via work-group cooperative Bitonic Sort in local memory,
     * 3. Quimper's Profile construction via parallel inclusive prefix sum & compaction,
     * 4. Task lower/upper bound filtering in parallel across work-group threads.
     *
     * @tparam S Type of the start time variables pointer
     */
    template<IntVarPtr S>
    class Cumulative {
    public:
        constexpr static bool parallel = true;

        Cumulative(
            Allocator& allocator, S s,
            Ptr<int32_t, Scope::global> p,
            Ptr<int32_t, Scope::global> h,
            uint32_t n, int32_t capacity
        ) : s(s), p(p), h(h), n(n), capacity(capacity),
            maxEvents(std::bit_ceil(std::max<uint32_t>(2, 2 * n))) {
            allocator.temporary<int32_t>(4 * maxEvents + 4);
        }

        DomainChange propagate(Context ctx) const {
            auto group = ctx.item.get_group();
            uint32_t tid = ctx.item.get_local_id(0);
            uint32_t groupSize = ctx.item.get_local_range(0);

            int32_t* tmp = ctx.getTemporary<int32_t>();
            uint32_t const offDelta = maxEvents;
            uint32_t const offProfT = 2 * maxEvents;
            uint32_t const offProfH = 3 * maxEvents;

            int32_t const* pPtr = p[ctx];
            int32_t const* hPtr = h[ctx];

            // 1. Deterministic parallel collection of mandatory parts [lst_i, ect_i[
            uint32_t nEvents = 0;
            bool heightOverload = false;
            for (uint32_t base = 0; base < n; base += groupSize) {
                uint32_t i = base + tid;
                int32_t dur = 0;
                int32_t height = 0;
                int32_t lst = 0;
                int32_t ect = 0;
                bool hasMand = false;
                if (i < n) {
                    dur = pPtr[i];
                    height = hPtr[i];
                    if (dur > 0 && height > 0) {
                        if (height > capacity) {
                            heightOverload = true;
                        } else {
                            auto& var = s[ctx, i][ctx];
                            int32_t est = var.min();
                            lst = var.max();
                            ect = est + dur;
                            hasMand = (lst < ect);
                        }
                    }
                }
                uint32_t flag = hasMand ? 2u : 0u;
                uint32_t pos = sycl::inclusive_scan_over_group(group, flag, sycl::plus<uint32_t>());
                if (hasMand) {
                    uint32_t idx = nEvents + pos - 2;
                    tmp[idx]                = lst;
                    tmp[offDelta + idx]     = height;
                    tmp[idx + 1]            = ect;
                    tmp[offDelta + idx + 1] = -height;
                }
                uint32_t chunkEvents = sycl::group_broadcast(group, pos, groupSize - 1);
                nEvents += chunkEvents;
            }

            if (sycl::any_of_group(group, heightOverload)) {
                sycl::group_barrier(group);
                return DomainChange::inconsistency;
            }
            if (nEvents == 0) {
                sycl::group_barrier(group);
                return DomainChange::none;
            }

            // 2. Parallel Bitonic Sort of time points in local memory
            uint32_t pow2 = 1;
            while (pow2 < nEvents) pow2 <<= 1;

            for (uint32_t e = nEvents + tid; e < pow2; e += groupSize) {
                tmp[e]            = std::numeric_limits<int32_t>::max();
                tmp[offDelta + e] = 0;
            }
            sycl::group_barrier(group);

            for (uint32_t k = 2; k <= pow2; k <<= 1) {
                for (uint32_t j = k >> 1; j > 0; j >>= 1) {
                    for (uint32_t i = tid; i < pow2; i += groupSize) {
                        uint32_t ixj = i ^ j;
                        if (ixj > i) {
                            bool ascending = ((i & k) == 0);
                            int32_t ti = tmp[i];
                            int32_t tixj = tmp[ixj];
                            if ((ascending && ti > tixj) || (!ascending && ti < tixj)) {
                                tmp[i]               = tixj;
                                tmp[ixj]             = ti;
                                int32_t di           = tmp[offDelta + i];
                                tmp[offDelta + i]    = tmp[offDelta + ixj];
                                tmp[offDelta + ixj]  = di;
                            }
                        }
                    }
                    sycl::group_barrier(group);
                }
            }

            // 3a. Parallel inclusive prefix sum over evtDelta to compute cumulative height
            int32_t runningSum = 0;
            for (uint32_t base = 0; base < nEvents; base += groupSize) {
                uint32_t e = base + tid;
                int32_t val = (e < nEvents) ? tmp[offDelta + e] : 0;
                int32_t scanVal = sycl::inclusive_scan_over_group(group, val, sycl::plus<int32_t>());
                if (e < nEvents) {
                    tmp[offDelta + e] = runningSum + scanVal;
                }
                int32_t chunkTotal = sycl::group_broadcast(group, scanVal, groupSize - 1);
                runningSum += chunkTotal;
            }
            sycl::group_barrier(group);

            // 3b. Parallel compaction of distinct time points into Quimper's Profile + Overload check
            uint32_t m = 0;
            bool profileOverload = false;
            for (uint32_t base = 0; base < nEvents; base += groupSize) {
                uint32_t e = base + tid;
                bool isLast = (e < nEvents) && (e + 1 == nEvents || tmp[e] != tmp[e + 1]);
                uint32_t flag = isLast ? 1u : 0u;
                uint32_t pos = sycl::inclusive_scan_over_group(group, flag, sycl::plus<uint32_t>());
                if (isLast) {
                    uint32_t dst = m + pos - 1;
                    int32_t hVal = tmp[offDelta + e];
                    tmp[offProfT + dst] = tmp[e];
                    tmp[offProfH + dst] = hVal;
                    if (hVal > capacity) {
                        profileOverload = true;
                    }
                }
                uint32_t chunkCount = sycl::group_broadcast(group, pos, groupSize - 1);
                m += chunkCount;
            }

            if (sycl::any_of_group(group, profileOverload)) {
                sycl::group_barrier(group);
                return DomainChange::inconsistency;
            }
            sycl::group_barrier(group);

            uint32_t numRects = m - 1;
            DomainChange change = DomainChange::none;

            // 4. Parallel Time-Table filtering across tasks (Gay et al., Algorithm 2)
            if (numRects > 0) {
                for (uint32_t i = tid; i < n; i += groupSize) {
                    int32_t dur = pPtr[i];
                    int32_t height = hPtr[i];
                    if (dur <= 0 || height <= 0) continue;

                    auto& var = s[ctx, i][ctx];
                    int32_t est = var.min();
                    int32_t lst = var.max();
                    if (est == lst) continue;

                    int32_t ect = est + dur;
                    int32_t lct = lst + dur;
                    bool hasMand = (lst < ect);
                    int32_t maxAllowed = capacity - height;

                    // Sweep left-to-right to filter lower bound (est)
                    if (est < tmp[offProfT + numRects]) {
                        uint32_t lo = 0, hi = numRects - 1;
                        while (lo < hi) {
                            uint32_t mid = (lo + hi) >> 1;
                            if (tmp[offProfT + mid + 1] <= est) {
                                lo = mid + 1;
                            } else {
                                hi = mid;
                            }
                        }
                        uint32_t j1 = lo;
                        int32_t sStar = est;
                        while (j1 < numRects && sStar <= lst && tmp[offProfT + j1] < sStar + dur) {
                            int32_t hOther = (hasMand && tmp[offProfT + j1] >= lst && tmp[offProfT + j1 + 1] <= ect)
                                             ? (tmp[offProfH + j1] - height)
                                             : tmp[offProfH + j1];
                            if (hOther > maxAllowed) {
                                sStar = tmp[offProfT + j1 + 1];
                            }
                            j1++;
                        }
                        if (sStar > est) {
                            change |= var.removeBelow(sStar);
                            if ((change & DomainChange::inconsistency) == DomainChange::inconsistency) {
                                break;
                            }
                        }
                    }

                    // Sweep right-to-left to filter upper bound (lct)
                    if (lct > tmp[offProfT]) {
                        uint32_t rlo = 0, rhi = numRects - 1;
                        while (rlo < rhi) {
                            uint32_t mid = (rlo + rhi + 1) >> 1;
                            if (tmp[offProfT + mid] < lct) {
                                rlo = mid;
                            } else {
                                rhi = mid - 1;
                            }
                        }
                        int32_t j2 = static_cast<int32_t>(rlo);
                        int32_t eStar = lct;
                        while (j2 >= 0 && eStar >= ect && tmp[offProfT + j2 + 1] > eStar - dur) {
                            int32_t hOther = (hasMand && tmp[offProfT + j2] >= lst && tmp[offProfT + j2 + 1] <= ect)
                                             ? (tmp[offProfH + j2] - height)
                                             : tmp[offProfH + j2];
                            if (hOther > maxAllowed) {
                                eStar = tmp[offProfT + j2];
                            }
                            j2--;
                        }
                        if (eStar < lct) {
                            change |= var.removeAbove(eStar - dur);
                            if ((change & DomainChange::inconsistency) == DomainChange::inconsistency) {
                                break;
                            }
                        }
                    }
                }
            }

            sycl::group_barrier(group);
            return change;
        }

    private:
        S s;
        Ptr<int32_t, Scope::global> p;
        Ptr<int32_t, Scope::global> h;
        uint32_t n;
        int32_t capacity;
        uint32_t maxEvents;
    };

}

#endif
