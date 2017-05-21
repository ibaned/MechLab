#include <goal_control.hpp>
#include <goal_field.hpp>
#include <goal_traits.hpp>
#include <goal_workset.hpp>

#include "ml_ev_momentum_resid.hpp"

namespace ml {

template <typename EVALT, typename TRAITS>
MomentumResid<EVALT, TRAITS>::MomentumResid(
    std::vector<goal::Field*> const& u,
    int type)
    : wdv(u[0]->wdv_name(), u[0]->ip0_dl(type)),
      stress("first_pk", u[0]->ip2_dl(type)) {

  num_nodes = u[0]->get_num_nodes(type);
  num_ips = u[0]->get_num_ips(type);
  num_dims = u[0]->get_num_dims();
  GOAL_DEBUG_ASSERT(num_dims == (int)u.size());

  grad_w.resize(num_dims);
  resid.resize(num_dims);

  for (int i = 0; i < num_dims; ++i) {
    auto wn = u[i]->g_basis_name();
    auto wdl = u[i]->w_dl(type);
    auto rn = u[i]->resid_name();
    auto rdl = u[i]->dl(type);
    grad_w[i] = PHX::MDField<const double, Ent, Node, IP, Dim>(wn, wdl);
    resid[i] = PHX::MDField<ScalarT, Ent, Node>(rn, rdl);
    this->addDependentField(grad_w[i]);
    this->addEvaluatedField(resid[i]);
  }

  this->addDependentField(wdv);
  this->addDependentField(stress);
  this->setName("Momentum Resid");
}

PHX_POST_REGISTRATION_SETUP(MomentumResid, data, fm) {
  for (int i = 0; i < num_dims; ++i) {
    this->utils.setFieldData(grad_w[i], fm);
    this->utils.setFieldData(resid[i], fm);
  }
  this->utils.setFieldData(wdv, fm);
  this->utils.setFieldData(stress, fm);
  (void)data;
}

PHX_EVALUATE_FIELDS(MomentumResid, workset) {
  for (int elem = 0; elem < workset.size; ++elem) {

    for (int node = 0; node < num_nodes; ++node)
    for (int dim = 0; dim < num_dims; ++dim)
      resid[dim](elem, node) = 0.0;

    for (int ip = 0; ip < num_ips; ++ip)
    for (int node = 0; node < num_nodes; ++node)
    for (int i = 0; i < num_dims; ++i)
    for (int j = 0; j < num_dims; ++j)
      resid[i](elem, node) +=
        stress(elem, ip, i, j) *
        grad_w[i](elem, node, ip, j) *
        wdv(elem, ip);
  }
}

template class MomentumResid<goal::Traits::Residual, goal::Traits>;
template class MomentumResid<goal::Traits::Jacobian, goal::Traits>;

} // namespace ml
