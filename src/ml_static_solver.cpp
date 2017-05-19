#include <goal_assembly.hpp>
#include <goal_control.hpp>
#include <goal_discretization.hpp>
#include <goal_output.hpp>
#include <goal_sol_info.hpp>

#include "ml_mechanics.hpp"
#include "ml_static_solver.hpp"

namespace ml {

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<std::string>("solver type", "");
  p.sublist("discretization");
  p.sublist("physics");
  p.sublist("output");
  return p;
}

static void validate_params(ParameterList const& p) {
  GOAL_ALWAYS_ASSERT(p.isSublist("discretization"));
  GOAL_ALWAYS_ASSERT(p.isSublist("physics"));
  GOAL_ALWAYS_ASSERT(p.isSublist("output"));
  p.validateParameters(get_valid_params(), 0);
}

StaticSolver::StaticSolver(ParameterList const& p)
    : params(p),
      disc(0),
      mech(0),
      info(0),
      out(0) {
  validate_params(params);
  auto dp = params.sublist("discretization");
  auto pp = params.sublist("physics");
  auto op = params.sublist("output");
  disc = goal::create_disc(dp);
  out = goal::create_output(op, disc);
  mech = ml::create_mech(pp, disc);
}

StaticSolver::~StaticSolver() {
  ml::destroy_mech(mech);
  goal::destroy_output(out);
  goal::destroy_disc(disc);
}

void StaticSolver::solve_primal() {
  goal::print("*** primal problem");
  mech->build_coarse_indexer();
  mech->build_primal_model();
  auto indexer = mech->get_indexer();
  info = goal::create_sol_info(indexer, 0);
  goal::compute_primal_jacobian(mech, info, disc, 0, 0);
  goal::destroy_sol_info(info);
  mech->destroy_model();
  mech->destroy_indexer();
}

void StaticSolver::solve() {
  goal::print("solving");
  solve_primal();
}

} // end namespace ml
