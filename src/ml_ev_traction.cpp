#include <apf.h>
#include <goal_control.hpp>
#include <goal_field.hpp>
#include <goal_indexer.hpp>
#include <goal_sol_info.hpp>
#include <goal_traits.hpp>
#include <goal_workset.hpp>
#include <Phalanx_DataLayout_MDALayout.hpp>

#include "ml_ev_traction.hpp"

namespace ml {

using Teuchos::rcp;

template <typename EVALT, typename TRAITS>
Traction<EVALT, TRAITS>::Traction(
    std::vector<goal::Field*> const& u,
    Teuchos::Array<std::string> const& array,
    goal::Indexer* i,
    int type)
    : disp(u),
      bc(array),
      indexer(i),
      info(0),
      wdv(u[0]->wdv_name(), u[0]->ip0_dl(type)) {

  num_nodes = u[0]->get_num_nodes(type);
  num_ips = u[0]->get_num_ips(type);
  num_dims = u[0]->get_num_dims();

  GOAL_DEBUG_ASSERT(num_dims == (int)u.size());
  GOAL_DEBUG_ASSERT(bc.size() == (num_dims + 1));

  w.resize(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    auto n = u[i]->basis_name();
    auto dl = u[i]->w_dl(type);
    w[i] = PHX::MDField<const double, Ent, Node, IP>(n, dl);
    this->addDependentField(w[i]);
  }

  auto name = "Traction: " + bc[0];
  PHX::Tag<ScalarT> op(name, rcp(new PHX::MDALayout<Dummy>(0)));

  this->addDependentField(wdv);
  this->addEvaluatedField(op);
  this->setName(name);
}

PHX_POST_REGISTRATION_SETUP(Traction, data, fm) {
  this->utils.setFieldData(wdv, fm);
  for (int i = 0; i < num_dims; ++i)
    this->utils.setFieldData(w[i], fm);
  (void)data;
}

PHX_PRE_EVALUATE_FIELDS(Traction, i) {
  info = i;
  GOAL_DEBUG_ASSERT(Teuchos::nonnull(info->ghost->R));
}

PHX_EVALUATE_FIELDS(Traction, workset) {
  apf::Vector3 x(0, 0, 0);
  apf::Vector3 xi(0, 0, 0);
  apf::Vector3 traction(0, 0, 0);
  std::vector<goal::LO> lids;

  auto t = workset.t_now;
  auto q_degree = disp[0]->get_q_degree();
  auto mesh = indexer->get_apf_mesh();
  auto R = info->ghost->R;

  for (int side = 0; side < workset.size; ++side) {
    auto s = workset.entities[side];
    auto me = apf::createMeshElement(mesh, s);
    for (int ip = 0; ip < num_ips; ++ip) {
      apf::getIntPoint(me, q_degree, ip, xi);
      apf::mapLocalToGlobal(me, xi, x);
      for (int i = 0; i < num_dims; ++i)
        traction[i] = goal::eval(bc[i+1], x[0], x[1], x[2], t);
      for (int node = 0; node < num_nodes; ++node) {
        for (int dim = 0; dim < num_dims; ++dim) {
          goal::LO row = indexer->get_ghost_lid(dim, s, node);
          double val =
            -w[dim](side, node, ip) *
            traction[dim] *
            wdv(side, ip);
          R->sumIntoLocalValue(row, val);
        }
      }
    }
    apf::destroyMeshElement(me);
  }
}

PHX_POST_EVALUATE_FIELDS(Traction, i) {
  (void)i;
}

template class Traction<goal::Traits::Residual, goal::Traits>;
template class Traction<goal::Traits::Jacobian, goal::Traits>;

} // end namespace ml
