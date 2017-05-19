#include <goal_control.hpp>
#include <goal_field.hpp>
#include <goal_states.hpp>
#include <goal_traits.hpp>
#include <goal_workset.hpp>
#include <Teuchos_ParameterList.hpp>

#include "ml_ev_elastic.hpp"

namespace ml {

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<double>("E", 0.0);
  p.set<double>("nu", 0.0);
  p.set<double>("alpha", 0.0);
  return p;
}

template <typename EVALT, typename TRAITS>
Elastic<EVALT, TRAITS>::Elastic(
    std::vector<goal::Field*> const& u,
    goal::States* s,
    ParameterList const& mp,
    int type)
    : states(s),
      cauchy("cauchy", u[0]->ip2_dl(type)) {

  num_dims = u[0]->get_num_dims();
  num_ips = u[0]->get_num_ips(type);
  GOAL_DEBUG_ASSERT(num_dims == (int)u.size());

  mp.validateParameters(get_valid_params(), 0);
  E = mp.get<double>("E");
  nu = mp.get<double>("nu");

  grad_u.resize(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    auto n = u[i]->g_name();
    auto dl = u[i]->g_dl(type);
    grad_u[i] = PHX::MDField<const ScalarT, Ent, IP, Dim>(n, dl);
    this->addDependentField(grad_u[i]);
  }

  this->addEvaluatedField(cauchy);
  this->setName("Elastic");
}

PHX_POST_REGISTRATION_SETUP(Elastic, data, fm) {
  for (int i = 0; i < num_dims; ++i)
    this->utils.setFieldData(grad_u[i], fm);
  this->utils.setFieldData(cauchy, fm);
  (void)data;
}

PHX_EVALUATE_FIELDS(Elastic, workset) {
  using Tensor = minitensor::Tensor<ScalarT>;
  Tensor eps(num_dims);
  Tensor sigma(num_dims);
  Tensor I(minitensor::eye<ScalarT>(num_dims));

  double mu = E / (2.0 * (1.0 + nu));
  double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

  for (int elem = 0; elem < workset.size; ++elem) {
    auto e = workset.entities[elem];
    for (int ip = 0; ip < num_dims; ++ip) {

      for (int i = 0; i < num_dims; ++i)
      for (int j = 0; j < num_dims; ++j)
        eps(i, j) = 0.5*(grad_u[i](elem, ip, j) + grad_u[j](elem, ip, i));

      sigma = 2.0*mu*eps + lambda*minitensor::trace(eps)*I;
      for (int i = 0; i < num_dims; ++i)
      for (int j = 0; j < num_dims; ++j)
        cauchy(elem, ip, i, j) = sigma(i, j);

      states->set_tensor("cauchy", e, ip, sigma);
    }
  }
}

template class Elastic<goal::Traits::Residual, goal::Traits>;
template class Elastic<goal::Traits::Jacobian, goal::Traits>;

} // end namespace ml
