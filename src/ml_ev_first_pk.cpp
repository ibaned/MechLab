#include <goal_control.hpp>
#include <goal_field.hpp>
#include <goal_traits.hpp>
#include <goal_workset.hpp>
#include <MiniTensor.h>

#include "ml_ev_first_pk.hpp"

namespace ml {

template <typename EVALT, typename TRAITS>
FirstPK<EVALT, TRAITS>::FirstPK(
    std::vector<goal::Field*> const& u,
    goal::Field* p,
    bool small,
    int type)
    : def_grad("F", u[0]->ip2_dl(type)),
      det_def_grad("J", u[0]->ip0_dl(type)),
      cauchy("cauchy", u[0]->ip2_dl(type)),
      first_pk("first_pk", u[0]->ip2_dl(type)) {

  num_ips = u[0]->get_num_ips(type);
  num_dims = u[0]->get_num_dims();
  GOAL_DEBUG_ASSERT(num_dims == (int)u.size());

  small_strain = small;
  have_pressure = (p) ? true : false;

  if (have_pressure) {
    auto n = p->name();
    auto dl = p->ip_dl(type);
    pressure = PHX::MDField<const ScalarT, Ent, IP>(n, dl);
    this->addDependentField(pressure);
  }

  this->addDependentField(def_grad);
  this->addDependentField(det_def_grad);
  this->addDependentField(cauchy);
  this->addEvaluatedField(first_pk);
  this->setName("First PK");
}

PHX_POST_REGISTRATION_SETUP(FirstPK, data, fm) {
  if (have_pressure)
    this->utils.setFieldData(pressure, fm);
  this->utils.setFieldData(def_grad, fm);
  this->utils.setFieldData(det_def_grad, fm);
  this->utils.setFieldData(cauchy, fm);
  this->utils.setFieldData(first_pk, fm);
  (void)data;
}

PHX_EVALUATE_FIELDS(FirstPK, workset) {

  using Tensor = minitensor::Tensor<ScalarT>;

  // populate the first PK tensor with the Cauchy stress.
  for (int elem = 0; elem < workset.size; ++elem)
  for (int ip = 0; ip < num_ips; ++ip)
  for (int i = 0; i < num_ips; ++i)
  for (int j = 0; j < num_ips; ++j)
    first_pk(elem, ip, i, j) = cauchy(elem, ip, i, j);

  // substitute pressure if mixed formulation
  if (have_pressure) {
    for (int elem = 0; elem < workset.size; ++elem) {
      for (int ip = 0; ip < num_ips; ++ip) {
        ScalarT pbar = 0.0;
        for (int i = 0; i < num_dims; ++i)
          pbar += cauchy(elem, ip, i, i);
        pbar /= num_dims;
        for (int i = 0; i < num_dims; ++i)
          first_pk(elem, ip, i, i) += pressure(elem, ip) - pbar;
      }
    }
  }

  // pull back to the reference configuration if finite deformation.
  if (! small_strain) {

    ScalarT J;
    Tensor F(num_dims);
    Tensor Finv(num_dims);
    Tensor sigma(num_dims);
    Tensor P(num_dims);
    Tensor I(minitensor::eye<ScalarT>(num_dims));

    for (int elem = 0; elem < workset.size; ++elem) {
      for (int ip = 0; ip < num_ips; ++ip) {

        J = det_def_grad(elem, ip);
        for (int i = 0; i < num_dims; ++i) {
          for (int j = 0; j < num_dims; ++j) {
            F(i, j) = def_grad(elem, ip, i, j);
            sigma(i, j) = first_pk(elem, ip, i, j);
          }
        }

        Finv = minitensor::inverse(F);
        P = J * sigma * minitensor::transpose(Finv);
        for (int i = 0; i < num_dims; ++i)
        for (int j = 0; j < num_dims; ++j)
          first_pk(elem, ip, i, j) = P(i, j);

      }
    }
  }

}

template class FirstPK<goal::Traits::Residual, goal::Traits>;
template class FirstPK<goal::Traits::Jacobian, goal::Traits>;

} // end namespace ml
