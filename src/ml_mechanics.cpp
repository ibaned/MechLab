#include <goal_control.hpp>
#include <goal_discretization.hpp>
#include <goal_field.hpp>
#include <goal_states.hpp>
#include "ml_mechanics.hpp"

namespace ml {

static ParameterList get_valid_params(goal::Discretization* d) {
  ParameterList p;
  p.set<int>("p order", 0);
  p.set<int>("q degree", 0);
  p.set<std::string>("model", "");
  p.sublist("dirichlet bcs");
  p.sublist("neumann bcs");
  for (int i = 0; i < d->get_num_elem_sets(); ++i)
    p.sublist(d->get_elem_set_name(i));
  return p;
}

static void validate_params(
    ParameterList const& p, goal::Discretization* d) {
  GOAL_ALWAYS_ASSERT(p.isType<int>("p order"));
  GOAL_ALWAYS_ASSERT(p.isType<int>("q degree"));
  GOAL_ALWAYS_ASSERT(p.isType<std::string>("model"));
  GOAL_ALWAYS_ASSERT(p.isSublist("dirichlet bcs"));
  for (int i = 0; i < d->get_num_elem_sets(); ++i)
    GOAL_ALWAYS_ASSERT(p.isSublist(d->get_elem_set_name(i)));
  p.validateParameters(get_valid_params(d), 0);
}

Mechanics::Mechanics(ParameterList const& p, goal::Discretization* d)
    : goal::Physics(d),
      params(p),
      is_primal(false),
      is_dual(false),
      is_error(false),
      states(0) {
  validate_params(p, d);
  p_order = params.get<int>("p order");
  q_degree = params.get<int>("q degree");
  model = params.get<std::string>("model");
  build_fields();
  build_states();
}

Mechanics::~Mechanics() {
  goal::destroy_states(states);
  for (size_t i = 0; i < u.size(); ++i)
    goal::destroy_field(u[i]);
  for (size_t i = 0; i < z.size(); ++i)
    goal::destroy_field(z[i]);
  for (size_t i = 0; i < e.size(); ++i)
    goal::destroy_field(e[i]);
  for (size_t i = 0; i < u_fine.size(); ++i)
    goal::destroy_field(u_fine[i]);
  for (size_t i = 0; i < z_fine.size(); ++i)
    goal::destroy_field(z_fine[i]);
}

ParameterList const& Mechanics::get_dbc_params() {
  return params.sublist("dirichlet bcs");
}

void Mechanics::set_primal() {
  is_primal = true;
  is_dual = false;
  is_error = false;
}

void Mechanics::set_dual() {
  is_primal = false;
  is_dual = true;
  is_error = false;
}

void Mechanics::set_error() {
  is_primal = false;
  is_dual = false;
  is_error = true;
}

void Mechanics::build_fields() {
  auto d = disc->get_num_dims();
  auto p = p_order;
  auto q = q_degree;
  auto t = goal::LAGRANGE;
  if (d > 0) u.push_back(goal::create_field({disc, "ux", p, q, t}));
  if (d > 1) u.push_back(goal::create_field({disc, "uy", p, q, t}));
  if (d > 2) u.push_back(goal::create_field({disc, "uz", p, q, t}));
  for (size_t i = 0; i < u.size(); ++i)
    u[i]->set_associated_dof_idx(i);
}

void Mechanics::build_states() {
  small_strain = false;
  states = goal::create_states(disc, q_degree);
  if (model == "elastic") {
    states->add("cauchy", 2);
    small_strain = true;
  } else if (model == "J2") {
    states->add("eqps", 0, true);
    states->add("Fp", 2, true, true);
    states->add("cauchy", 2);
  } else
    goal::fail("unkown material model %s", model.c_str());
}

template <typename T>
static void write_graph(goal::FieldManager fm, const char* n) {
  fm->writeGraphvizFile<T>(n, true, true);
}

void Mechanics::build_primal_volumetric(FieldManager fm) {
  set_primal();
  register_volumetric<Residual>(fm);
  register_volumetric<Jacobian>(fm);
  write_graph<Jacobian>(fm, "p_volumetric.dot");
}

void Mechanics::build_primal_neumann(FieldManager fm) {
  set_primal();
  register_neumann<Residual>(fm);
  register_neumann<Jacobian>(fm);
  write_graph<Jacobian>(fm, "p_neumann.dot");
}

void Mechanics::build_dual_volumetric(FieldManager fm) {
  set_dual();
  register_volumetric<Jacobian>(fm);
  write_graph<Jacobian>(fm, "d_volumetric.dot");
}

void Mechanics::build_dual_neumann(FieldManager fm) {
  set_dual();
  register_neumann<Jacobian>(fm);
  write_graph<Jacobian>(fm, "d_neumann.dot");
}

void Mechanics::build_error_volumetric(FieldManager fm) {
  set_error();
  register_volumetric<Residual>(fm);
  write_graph<Residual>(fm, "e_volumetric.dot");
}

void Mechanics::build_error_neumann(FieldManager fm) {
  set_error();
  register_neumann<Residual>(fm);
  write_graph<Residual>(fm, "e_neumann.dot");
}

Mechanics* create_mech(ParameterList const& p, goal::Discretization* d) {
  return new Mechanics(p, d);
}

void destroy_mech(Mechanics* p) {
  delete p;
}

} // end namespace ml
