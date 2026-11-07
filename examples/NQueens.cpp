#include <iostream>
#include <vector>
#include <string>
#include "MiniGCP.hpp"

using namespace std;
using namespace cp;


/**
 * @brief Get a solver for an NQueens problem instance
 * @param n Number of queens
 * @param params Device parameters
 * @return The solver
 */
static auto getNQueensSolver(uint32_t n, IsDeviceParams auto params) {
    Model<
        AllDifferentFWC<IntBitsetArray>,
        AllDifferentFWC<AddN<IntBitsetArray>>,
        AllDifferentFWC<AddN<Neg1<IntBitsetArray>>>
    > model;

    IntBitsetArray variables(model, n, n);
    vector<int32_t> indicesVec;
    for (uint32_t i = 0; i < n; i++) {
        indicesVec.push_back(i);
    }
    Array<int32_t> indices(model, indicesVec);
    model.constraint(AllDifferentFWC(model, variables));
    model.constraint(AllDifferentFWC(model, variables + indices));
    model.constraint(AllDifferentFWC(model, -variables + indices));

    FirstFail strategy(variables);
    return DFSSolver(model, Satisfaction(), variables, strategy, params);
}


/**
 * @brief Get a solver for an NQueens problem instance.
 * The problem is decomposed into binary constraints.
 * @param n Number of queens
 * @param params Device parameters
 * @return The solver
 */
static auto getNQueensBinarySolver(uint32_t n, IsDeviceParams auto params) {
    Model<
        NotEqual<IntBitset, IntBitset>,
        NotEqual<Add<IntBitset>, Add<IntBitset>>
    > model;

    IntBitsetArray variables(model, n, n);
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < i; j++) {
            model.constraint(variables[i] != variables[j]);
            model.constraint(variables[i] + i != variables[j] + j);
            model.constraint(variables[i] - i != variables[j] - j);
        }
    }

    FirstFail strategy(variables);
    return DFSSolver(model, Satisfaction(), variables, strategy, params);
}


#ifndef NO_MAIN
int main(int argc, char const* const* argv) {
    if (argc < 2 || argc > 3) {
        cerr << "Arguments: number of queens, maximum number of solutions (optional)" << endl;
        return 1;
    }

    uint32_t n = std::stoi(argv[1]);
    uint32_t limit = argc == 3 ? std::stoi(argv[2]) : UINT32_MAX;
    DeviceParams<1> params(sycl::cpu_selector_v, 1);
    auto solver = getNQueensSolver(n, params);
    solver.solve([=](SearchStatistics stats) { return stats.nSolutions >= limit; });

    cout << solver.getStatistics() << endl;
    for (int32_t s : solver.getSolution()) {
        cout << s << " ";
    }
    cout << endl;
}
#endif