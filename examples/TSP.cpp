#include <span>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <random>
#include "MiniGCP.hpp"

using namespace std;
using namespace cp;


/**
 * @brief Traveling salesman problem instance
 */
class TSPInstance {
public:
    uint32_t n;
    vector<int32_t> distances;

    TSPInstance(uint32_t n, uint32_t size) : n(n), distances(n * n) {
        mt19937 generator(0);
        uniform_int_distribution<uint32_t> coordinate(0, size);
        vector<pair<uint32_t, uint32_t>> positions;
        positions.reserve(n);
        for (uint32_t i = 0; i < n; i++) {
            positions.emplace_back(coordinate(generator), coordinate(generator));
        }

        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < n; j++) {
                double dx = static_cast<double>(positions[i].first) - positions[j].first;
                double dy = static_cast<double>(positions[i].second) - positions[j].second;
                distances[i * n + j] = static_cast<int32_t>(lround(hypot(dx, dy)));
            }
        }
    }

    explicit TSPInstance(char const* path) {
        ifstream input(path);
        input.exceptions(std::ios::failbit);
        input >> n;
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < n; j++) {
                int32_t d;
                input >> d;
                distances.push_back(d);
            }
        }
    }

    int32_t distance(uint32_t i, uint32_t j) const {
        return distances[i * n + j];
    }
};


/**
 * @brief Get a solver for a TSP instance
 * @param instance The instance
 * @param params Device parameters
 * @return The solver
 */
static auto getTSPSolver(TSPInstance const& instance, IsDeviceParams auto params) {
    Model<
        AllDifferentFWC<IntBitsetPtr>,
        Circuit<IntBitsetPtr>,
        Element1D<IntBitset, IntInterval>,
        Sum<IntIntervalPtr, IntInterval>
    > model;

    uint32_t n = instance.n;
    IntBitsetPtr succ(model, n, n);
    model.constraint(AllDifferentFWC(model, succ, n));
    model.constraint(Circuit(model, succ, n));
    IntIntervalPtr cost(model, n);
    for (uint32_t i = 0; i < n; i++) {
        span<int32_t const> distances(instance.distances.data() + i * n, n);
        model.constraint(Element1D(model, Ptr(model, distances), n, succ[i], cost[i]));
    }
    IntInterval objective = sum(model, cost, n);

    FirstFail strategy(succ, n);
    return DFSSolver(model, Minimization(objective), succ, n, strategy, params);
}


#ifndef NO_MAIN
int main(int argc, char const* const* argv) {
    if (argc != 2) {
        cerr << "Arguments: input path" << endl;
        return 1;
    }

    TSPInstance instance(argv[1]);
    DeviceParams<1> params(sycl::cpu_selector_v, 1);
    auto solver = getTSPSolver(instance, params);
    solver.solve(noLimit);

    cout << solver.getStatistics() << endl;
    for (int32_t s : solver.getSolution()) {
        cout << s << " ";
    }
    cout << endl;
}
#endif