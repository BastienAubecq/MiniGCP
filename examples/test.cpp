#include <iostream>
#include <vector>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/NotEqual.hpp"
#include "MiniGCP/search/FirstFail.hpp"
#include "MiniGCP/solver/EPSSolver.hpp"

using namespace std;
using namespace cp;

int main() {
    int n = 12;

    Model<NotEqual> model(ProblemType::satisfaction);
    vector<IntVarIdx> variables;
    for (int i = 0; i < n; i++) {
        variables.push_back(model.variable(n));
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            model.constraint(variables[i] != variables[j]);
            model.constraint(variables[i] != variables[j] + i - j);
            model.constraint(variables[i] != variables[j] + j - i);
        }
    }
    auto variablesArray = model.array<Scope::global, IntVarIdx>(variables);
    model.output = variablesArray;

    FirstFail strategy(variablesArray);
    DeviceParams params(sycl::gpu_selector_v, 128, 14);
    auto solver = EPSSolver(model, strategy, params, 10);
    solver.solve([](SearchStatistics stats) { return stats.nSolutions >= 1000; });

    cout << solver.getStatistics() << endl;
    solver.solve(noLimit);

    cout << solver.getStatistics() << endl;
    span<int32_t const> solution = solver.getSolution();
    for (int i = 0; i < n; i++) {
        cout << solution[i] << " ";
    }
    cout << endl;
}