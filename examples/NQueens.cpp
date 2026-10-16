#include <iostream>
#include <vector>
#include <string>
#include "MiniGCP.hpp"

using namespace std;
using namespace cp;


/**
 * @brief Solve the NQueens problem
 * @param n Number of queens
 * @param limit Maximum number of solutions
 * @param solution Row of each queen in the first solution
 * @return Search statistics
 */
static SearchStatistics solve(uint32_t n, uint32_t limit, DeviceParams params, vector<int32_t>& solution) {
    Model<AllDifferentFWC> model(ProblemType::satisfaction);
    vector<IntVarIdx> variables;
    vector<IntVarIdx> diag1;
    vector<IntVarIdx> diag2;
    for (int i = 0; i < n; i++) {
        IntVarIdx q = model.variable(n);
        variables.push_back(q);
        diag1.push_back(q + i);
        diag2.push_back(q - i);
    }
    model.constraint(AllDifferentFWC(model, variables));
    model.constraint(AllDifferentFWC(model, diag1));
    model.constraint(AllDifferentFWC(model, diag2));
    model.output = model.array<Scope::global, IntVarIdx>(variables);

    FirstFail strategy(model.output);
    auto solver = DFSSolver(model, strategy, params);
    solver.solve([=](SearchStatistics stats) { return stats.nSolutions >= limit; });

    for (uint32_t i = 0; i < n; i++) {
        solution.push_back(solver.getSolution()[i]);
    }
    return solver.getStatistics();
}


int main(int argc, char const* const* argv) {
    if (argc < 2 || argc > 3) {
        cerr << "Arguments: number of queens, maximum number of solutions (optional)" << endl;
        return 1;
    }

    uint32_t n = std::stoi(argv[1]);
    uint32_t limit = argc == 3 ? std::stoi(argv[2]) : UINT32_MAX;
    DeviceParams params(sycl::cpu_selector_v, 1, 1);
    vector<int32_t> solution;
    SearchStatistics stats = solve(n, limit, params, solution);
    cout << stats << endl;
    for (int32_t s : solution) {
        cout << s << " ";
    }
    cout << endl;
}