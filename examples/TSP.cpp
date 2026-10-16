#include <span>
#include <iostream>
#include <fstream>
#include <chrono>
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

    explicit TSPInstance(char const* path) {
        ifstream input(path);
        input >> n;
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < n; j++) {
                int32_t d;
                input >> d;
                distances.push_back(d);
            }
        }
    }

    int32_t distance(uint32_t i, uint32_t j) {
        return distances[i * n + j];
    }
};


/**
 * @brief Solve a TSP instance
 * @param instance The instance
 * @param params Device parameters
 * @param solution Successor of each node in the best solution
 * @return Search statistics
 */
static SearchStatistics solve(TSPInstance const& instance, DeviceParams params, vector<int32_t>& solution) {
    Model<AllDifferentFWC, Circuit, Element1D, Sum> model(ProblemType::minimization);
    uint32_t n = instance.n;

    vector<IntVarIdx> succ;
    for (uint32_t i = 0; i < n; i++) {
        succ.push_back(model.variable(n));
    }
    auto succArray = model.array<Scope::global, IntVarIdx>(succ);
    model.constraint(AllDifferentFWC(model, succ));
    model.constraint(Circuit(model, succArray));
    model.output = succArray;

    vector<IntVarIdx> cost;
    for (uint32_t i = 0; i < n; i++) {
        cost.push_back(element1D(model, model.array<Scope::global>(span(instance.distances.data() + i * n, n)), succ[i]));
    }
    model.objective = sum(model, cost);

    FirstFail strategy(model.output);
    auto solver = DFSSolver(model, strategy, params);
    solver.solve(noLimit);

    for (uint32_t i = 0; i < n; i++) {
        solution.push_back(solver.getSolution()[i]);
    }
    return solver.getStatistics();
}


int main(int argc, char const* const* argv) {
    if (argc != 2) {
        cerr << "Arguments: input path" << endl;
        return 1;
    }

    TSPInstance instance(argv[1]);
    DeviceParams params(sycl::cpu_selector_v, 1, 1);
    vector<int32_t> solution;
    SearchStatistics stats = solve(instance, params, solution);
    cout << stats << endl;
    for (int32_t s : solution) {
        cout << s << " ";
    }
    cout << endl;
}