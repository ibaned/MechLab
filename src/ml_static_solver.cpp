#include <goal_assembly.hpp>
#include <goal_control.hpp>
#include <goal_dbcs.hpp>
#include <goal_discretization.hpp>
#include <goal_linear_solvers.hpp>
#include <goal_indexer.hpp>
#include <goal_output.hpp>
#include <goal_sol_info.hpp>

#include "ml_mechanics.hpp"
#include "ml_static_solver.hpp"

namespace ml {

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<std::string>("solver type", "");
  p.sublist("discretization");
  p.sublist("mechanics");
  p.sublist("output");
  p.sublist("linear algebra");
  return p;
}

static void validate_params(ParameterList const& p) {
  GOAL_ALWAYS_ASSERT(p.isSublist("discretization"));
  GOAL_ALWAYS_ASSERT(p.isSublist("mechanics"));
  GOAL_ALWAYS_ASSERT(p.isSublist("output"));
  GOAL_ALWAYS_ASSERT(p.isSublist("linear algebra"));
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
  auto mp = params.sublist("mechanics");
  auto op = params.sublist("output");
  disc = goal::create_disc(dp);
  out = goal::create_output(op, disc);
  mech = ml::create_mech(mp, disc);
  auto model = mp.get<std::string>("model");
  is_linear = (model == "elastic");
}

StaticSolver::~StaticSolver() {
  ml::destroy_mech(mech);
  goal::destroy_output(out);
  goal::destroy_disc(disc);
}

void StaticSolver::solve_linear_primal() {
  auto indexer = mech->get_indexer();
  goal::compute_primal_jacobian(mech, info, disc, 0, 0);
  auto R = info->owned->R;
  auto dRdu = info->owned->dRdu;
  auto du = info->owned->du;
  du->putScalar(0.0);
  R->scale(-1.0);
  auto lp = params.sublist("linear algebra");
  goal::solve_linear_system(lp, dRdu, du, R, indexer);
  indexer->add_to_fields(mech->get_u(), du);
  goal::compute_primal_residual(mech, info, disc, 0, 0);
}

void StaticSolver::solve_nonlinear_primal() {
}

void StaticSolver::solve_primal() {
  goal::print("*** primal problem");

  // build + assemble the primal data
  mech->build_coarse_indexer();
  mech->build_primal_model();
  goal::set_dbc_values(mech, 0.0);
  info = goal::create_sol_info(mech->get_indexer(), 0);

  // solve the linear algebra problem
  if (is_linear) solve_linear_primal();
  else solve_nonlinear_primal();

  // finalize the primal data
  goal::destroy_sol_info(info);
  mech->destroy_model();
  mech->destroy_indexer();
}

void StaticSolver::solve() {
  goal::print("solving");
  solve_primal();
  out->write(0);
}

} // end namespace ml
