#include <goal_control.hpp>
#include "ml_static_solver.hpp"

namespace ml {

StaticSolver::StaticSolver(ParameterList const& p)
    : params(p) {
}

StaticSolver::~StaticSolver() {
}

void StaticSolver::solve() {
  goal::print("solving");
}

} // end namespace ml
