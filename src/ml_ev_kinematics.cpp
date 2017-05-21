#include <goal_control.hpp>
#include <goal_field.hpp>
#include <goal_traits.hpp>
#include <goal_workset.hpp>
#include <MiniTensor.h>

#include "ml_ev_kinematics.hpp"

namespace ml {

template <typename EVALT, typename TRAITS>
Kinematics<EVALT, TRAITS>::Kinematics(
    std::vector<goal::Field*> const& u, int type)
    : def_grad("F", u[0]->ip2_dl(type)),
      det_def_grad("J", u[0]->ip0_dl(type)) {

  num_dims = u[0]->get_num_dims();
  num_ips = u[0]->get_num_ips(type);
  GOAL_DEBUG_ASSERT(num_dims == (int)u.size());

  grad_u.resize(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    auto n = u[i]->g_name();
    auto dl = u[i]->g_ip_dl(type);
    grad_u[i] = PHX::MDField<const ScalarT, Ent, IP, Dim>(n, dl);
    this->addDependentField(grad_u[i]);
  }

  this->addEvaluatedField(def_grad);
  this->addEvaluatedField(det_def_grad);
  this->setName("Kinematics");
}

PHX_POST_REGISTRATION_SETUP(Kinematics, data, fm) {
  for (int i = 0; i < num_dims; ++i)
    this->utils.setFieldData(grad_u[i], fm);
  this->utils.setFieldData(def_grad, fm);
  this->utils.setFieldData(det_def_grad, fm);
  (void)data;
}

PHX_EVALUATE_FIELDS(Kinematics, workset) {
  minitensor::Tensor<ScalarT> F(num_dims);

  for (int elem = 0; elem < workset.size; ++elem) {
    for (int ip = 0; ip < num_ips; ++ip) {

      for (int i = 0; i < num_dims; ++i)
      for (int j = 0; j < num_dims; ++j)
        def_grad(elem, ip, i, j) = grad_u[i](elem, ip, j);

      for (int i = 0; i < num_dims; ++i)
        def_grad(elem, ip, i, i) += 1.0;

      for (int i = 0; i < num_dims; ++i)
      for (int j = 0; j < num_dims; ++j)
        F(i, j) = def_grad(elem, ip, i, j);

      det_def_grad(elem, ip) = minitensor::det(F);
    }
  }
}

template class Kinematics<goal::Traits::Residual, goal::Traits>;
template class Kinematics<goal::Traits::Jacobian, goal::Traits>;

} // end namespace ml
