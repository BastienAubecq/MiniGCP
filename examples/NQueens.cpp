#include <iostream>
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
        AllDifferentFWC<IntBitsetPtr>,
        AllDifferentFWC<AddVec<IntBitsetPtr, Range>>,
        AllDifferentFWC<AddVec<NegVec<IntBitsetPtr>, Range>>
    > model;

    IntBitsetPtr variables(model, n, n);
    model.constraint(AllDifferentFWC(model, variables, n));
    model.constraint(AllDifferentFWC(model, variables + Range(), n));
    model.constraint(AllDifferentFWC(model, -variables + Range(), n));

    FirstFail strategy(variables, n);
    return DFSSolver(model, Satisfaction(), variables, n, strategy, params);
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
        NotEqual<Add<IntBitset, int32_t>, Add<IntBitset, int32_t>>
    > model;

    IntBitsetPtr variables(model, n, n);
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < i; j++) {
            model.constraint(variables[i] != variables[j]);
            model.constraint(variables[i] + i != variables[j] + j);
            model.constraint(variables[i] - i != variables[j] - j);
        }
    }

    FirstFail strategy(variables, n);
    return DFSSolver(model, Satisfaction(), variables, n, strategy, params);
}


/**
 * @brief Get a solver for an NQueens problem instance.
 * The problem is decomposed into vectors of binary constraints.
 * @param n Number of queens
 * @param params Device parameters
 * @return The solver
 */
static auto getNQueensVectorSolver(uint32_t n, IsDeviceParams auto params) {
    Model<
        NotEqualVec<Same<IntBitset>, IntBitsetPtr>,
        NotEqualVec<Same<Add<IntBitset, int32_t>>, AddVec<IntBitsetPtr, Range>>,
        NotEqualVec<Same<Add<Neg<IntBitset>, int32_t>>, AddVec<NegVec<IntBitsetPtr>, Range>>
    > model;

    IntBitsetPtr variables(model, n, n);
    for (int32_t i = 1; i < n; i++) {
        model.constraint(NotEqualVec(Same(variables[i]), variables, i));
        model.constraint(NotEqualVec(Same(variables[i] + i), variables + Range(), i));
        model.constraint(NotEqualVec(Same(-variables[i] + i), -variables + Range(), i));
    }

    FirstFail strategy(variables, n);
    return DFSSolver(model, Satisfaction(), variables, n, strategy, params);
}


#ifndef NO_MAIN
int main(int argc, char const* const* argv) {
    if (argc < 2 || argc > 3) {
        cerr << "Arguments: number of queens, maximum number of solutions (optional)" << endl;
        return 1;
    }

    uint32_t n = std::stoi(argv[1]);
    uint32_t limit = argc == 3 ? std::stoi(argv[2]) : UINT32_MAX;
    DeviceParams<64> params(sycl::gpu_selector_v, 1);
    auto solver = getNQueensVectorSolver(n, params);
    solver.solve([=](SearchStatistics stats) { return stats.nSolutions >= limit; });

    cout << solver.getStatistics() << endl;
    for (int32_t s : solver.getSolution()) {
        cout << s << " ";
    }
    cout << endl;
}
#endif