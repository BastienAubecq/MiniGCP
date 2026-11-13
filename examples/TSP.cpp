#include <span>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include "MiniGCP.hpp"

using namespace std;
using namespace cp;


/**
 * @brief Traveling salesman problem
 * https://en.wikipedia.org/wiki/Travelling_salesman_problem
 */
class TSP {
public:
    uint32_t n;

    /**
     * @brief Generate a random TSP instance in a 2D grid
     * @param n Number of cities
     * @param size Size of the grid
     */
    TSP(uint32_t n, uint32_t size) : n(n) {
        mt19937 generator(0);
        uniform_int_distribution<uint32_t> coordinate(0, size);
        vector<pair<uint32_t, uint32_t>> positions;
        for (uint32_t i = 0; i < n; i++) {
            positions.emplace_back(coordinate(generator), coordinate(generator));
        }

        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < n; j++) {
                double dx = static_cast<double>(positions[i].first) - positions[j].first;
                double dy = static_cast<double>(positions[i].second) - positions[j].second;
                distances.push_back(static_cast<int32_t>(lround(hypot(dx, dy))));
            }
        }
    }

    /**
     * @brief Read a TSP instance from a file
     * @param path Path to the file
     */
    explicit TSP(char const* path) {
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


    /**
     * @brief Get a solver for the TSP instance
     * @param params Device parameters
     * @return The solver
     */
    auto solver(IsDeviceParams auto params) const {
        Model<
            AllDifferentFWC<IntBitsetPtr>,
            Circuit<IntBitsetPtr>,
            Element1D<IntBitset, IntInterval>,
            Sum<IntIntervalPtr, IntInterval>
        > model;

        IntBitsetPtr succ(model, n, n);
        model.constraint(AllDifferentFWC(model, succ, n));
        model.constraint(Circuit(model, succ, n));
        IntIntervalPtr cost(model, n);
        for (uint32_t i = 0; i < n; i++) {
            span<int32_t const> distancesSpan(distances.data() + i * n, n);
            model.constraint(Element1D(model, Ptr(model, distancesSpan), n, succ[i], cost[i]));
        }
        IntInterval objective = sum(model, cost, n);

        FirstFail strategy(succ, n);
        return DFSSolver(model, Minimization(objective), succ, n, strategy, params);
    }


    int32_t distance(uint32_t i, uint32_t j) const {
        return distances[i * n + j];
    }

private:
    vector<int32_t> distances;
};


#ifndef NO_MAIN
int main(int argc, char const* const* argv) {
    if (argc != 2) {
        cerr << "Arguments: input path" << endl;
        return 1;
    }

    TSP tsp(argv[1]);
    DeviceParams<128> params(sycl::gpu_selector_v, 1);
    auto solver = tsp.solver(params);
    solver.solve(noLimit);

    cout << solver.getStatistics() << endl;
    for (int32_t s : solver.getSolution()) {
        cout << s << " ";
    }
    cout << endl;
}
#endif