#include <goal_control.hpp>
#include "ml_static_solver.hpp"

namespace ml {

Solver::Solver() {
}

Solver::~Solver() {
}

Solver* create_solver(ParameterList const& p) {
  GOAL_ALWAYS_ASSERT(p.isType<std::string>("solver type"));
  auto type = p.get<std::string>("solver type");
  Solver* solver = 0;
  if (type == "static")
    solver = new StaticSolver(p);
  else
    goal::fail("unknown solver type");
  return solver;
}

void destroy_solver(Solver* s) {
  delete s;
}

} // end namespace ml
