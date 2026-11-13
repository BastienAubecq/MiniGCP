#include <span>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>
#include "MiniGCP.hpp"

using namespace std;
using namespace cp;


/**
 * @brief Resource-Constrained Project Scheduling Problem (RCPSP) instance
 */
class RCPSPInstance {
public:
    uint32_t n; // Total number of tasks (including source 0 and sink n - 1)
    uint32_t nResources;
    vector<int32_t> capacities;
    vector<int32_t> durations;
    vector<vector<int32_t>> demands; // demands[r][i] = requirement of task i on resource r
    vector<uint32_t> pred;
    vector<uint32_t> succ;
    vector<int32_t> predDur;

    /**
     * @brief Generate a random reproducible RCPSP instance
     * @param nActivities Number of non-dummy activities
     * @param nResources Number of renewable resources
     * @param seed Random seed
     */
    RCPSPInstance(uint32_t nActivities, uint32_t nResources = 2, uint32_t seed = 0)
        : n(nActivities + 2), nResources(nResources),
          capacities(nResources, 6), durations(n, 0),
          demands(nResources, vector<int32_t>(n, 0)) {
        mt19937 gen(seed);
        uniform_int_distribution<int32_t> durDist(2, 6);
        uniform_int_distribution<int32_t> reqDist(1, 4);

        for (uint32_t i = 1; i <= nActivities; i++) {
            durations[i] = durDist(gen);
            for (uint32_t r = 0; r < nResources; r++) {
                demands[r][i] = reqDist(gen);
            }
        }

        // Source 0 precedes all activities; all activities precede sink n - 1
        for (uint32_t i = 1; i <= nActivities; i++) {
            addPrecedence(0, i);
            for (uint32_t j = i + 1; j <= nActivities; j++) {
                if (j - i <= 3 && (gen() % 3 == 0)) {
                    addPrecedence(i, j);
                }
            }
            addPrecedence(i, n - 1);
        }
    }

    /**
     * @brief Load an RCPSP instance in Patterson (.rcp) format
     * @param path File path
     */
    explicit RCPSPInstance(char const* path) {
        ifstream input(path);
        input.exceptions(std::ios::failbit);
        input >> n >> nResources;
        capacities.resize(nResources);
        for (uint32_t r = 0; r < nResources; r++) {
            input >> capacities[r];
        }

        durations.resize(n);
        demands.assign(nResources, vector<int32_t>(n, 0));

        vector<bool> hasSuccessor(n, false);
        for (uint32_t i = 0; i < n; i++) {
            input >> durations[i];
            for (uint32_t r = 0; r < nResources; r++) {
                input >> demands[r][i];
            }
            uint32_t nSucc = 0;
            input >> nSucc;
            for (uint32_t k = 0; k < nSucc; k++) {
                uint32_t sIdx;
                input >> sIdx;
                // Standard .rcp files use 1-based task indices (1..n)
                uint32_t v = (sIdx >= 1 && sIdx <= n) ? (sIdx - 1) : sIdx;
                addPrecedence(i, v);
                hasSuccessor[i] = true;
            }
        }

        // Ensure every task i < n - 1 precedes the sink task n - 1
        for (uint32_t i = 0; i + 1 < n; i++) {
            if (!hasSuccessor[i]) {
                addPrecedence(i, n - 1);
            }
        }
    }

    void addPrecedence(uint32_t u, uint32_t v) {
        pred.push_back(u);
        succ.push_back(v);
        predDur.push_back(durations[u]);
    }
};


/**
 * @brief Build a solver for an RCPSP instance
 * @param instance The RCPSP instance
 * @param params Device parameters
 * @return The solver
 */
static auto getRCPSPSolver(RCPSPInstance const& instance, IsDeviceParams auto params) {
    Model<
        PrecedenceVec<IntIntervalPtr>,
        Cumulative<IntIntervalPtr>
    > model;

    uint32_t n = instance.n;
    int32_t horizon = std::accumulate(instance.durations.begin(), instance.durations.end(), 0);

    IntIntervalPtr start(model, 0, horizon, n);
    start[0][model].fix(0);

    model.constraint(PrecedenceVec(
        model, start,
        Ptr(model, span<uint32_t const>(instance.pred)),
        Ptr(model, span<uint32_t const>(instance.succ)),
        Ptr(model, span<int32_t const>(instance.predDur)),
        instance.pred.size()
    ));

    Ptr<int32_t, Scope::global> durationsPtr(model, span<int32_t const>(instance.durations));
    for (uint32_t r = 0; r < instance.nResources; r++) {
        Ptr<int32_t, Scope::global> demandsPtr(model, span<int32_t const>(instance.demands[r]));
        model.constraint(Cumulative(model, start, durationsPtr, demandsPtr, n, instance.capacities[r]));
    }

    EstFirst strategy(start, n);
    return DFSSolver(model, Minimization(start[n - 1]), start, n, strategy, params);
}


#ifndef NO_MAIN
int main(int argc, char const* const* argv) {
    if (argc != 2) {
        cerr << "Arguments: input path" << endl;
        return 1;
    }

    RCPSPInstance instance(argv[1]);
    DeviceParams<64> params(sycl::gpu_selector_v, 1);
    auto solver = getRCPSPSolver(instance, params);
    solver.solve(noLimit);

    cout << solver.getStatistics() << endl;
    for (int32_t s : solver.getSolution()) {
        cout << s << " ";
    }
    cout << endl;
}
#endif
