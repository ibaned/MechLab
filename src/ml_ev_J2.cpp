#include <goal_control.hpp>
#include <goal_field.hpp>
#include <goal_states.hpp>
#include <goal_traits.hpp>
#include <goal_workset.hpp>
#include <Teuchos_ParameterList.hpp>

#include "ml_ev_J2.hpp"

namespace ml {

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<double>("E", 0.0);
  p.set<double>("nu", 0.0);
  p.set<double>("K", 0.0);
  p.set<double>("Y", 0.0);
  p.set<double>("alpha", 0.0);
  return p;
}

template <typename EVALT, typename TRAITS>
J2<EVALT, TRAITS>::J2(
    std::vector<goal::Field*> const& u,
    goal::States* s,
    ParameterList const& mp,
    int type)
    : states(s),
      def_grad("F", u[0]->ip2_dl(type)),
      det_def_grad("J", u[0]->ip0_dl(type)),
      cauchy("cauchy", u[0]->ip2_dl(type)) {

  num_dims = u[0]->get_num_dims();
  num_ips = u[0]->get_num_ips(type);
  GOAL_DEBUG_ASSERT(num_dims = (int)u.size());

  mp.validateParameters(get_valid_params(), 0);
  E = mp.get<double>("E");
  nu = mp.get<double>("nu");
  K = mp.get<double>("K");
  Y = mp.get<double>("Y");

  this->addDependentField(def_grad);
  this->addDependentField(det_def_grad);
  this->addEvaluatedField(cauchy);
  this->setName("J2");
}

PHX_POST_REGISTRATION_SETUP(J2, data, fm) {
  this->utils.setFieldData(def_grad, fm);
  this->utils.setFieldData(det_def_grad, fm);
  this->utils.setFieldData(cauchy, fm);
  (void)data;
}

PHX_EVALUATE_FIELDS(J2, workset) {

  using Tensor = minitensor::Tensor<ScalarT>;

  // material variables
  double kappa = E / (3.0 * (1.0 - 2.0 * nu));
  double mu = E / (2.0 * (1.0 + nu));
  double sq23 = std::sqrt(2.0 / 3.0);

  // quantities at previous time
  ScalarT eqps;
  Tensor Fp(num_dims);
  Tensor Fpinv(num_dims);
  Tensor Cpinv(num_dims);

  // quantities at current time
  ScalarT J;
  ScalarT Jm23;
  ScalarT dgam;
  Tensor F(num_dims);
  Tensor Fpn(num_dims);
  Tensor N(num_dims);
  Tensor sigma(num_dims);
  Tensor I(minitensor::eye<ScalarT>(num_dims));

  // trial state quantities
  ScalarT f;
  ScalarT mubar;
  Tensor be(num_dims);
  Tensor s(num_dims);

  for (int elem = 0; elem < workset.size; ++elem) {

    apf::MeshEntity* e = workset.entities[elem];

    for (int ip = 0; ip < num_ips; ++ip) {

      // deformation gradient quantities
      for (int i = 0; i < num_dims; ++i)
      for (int j = 0; j < num_dims; ++j)
        F(i, j) = def_grad(elem, ip, i, j);
      J = det_def_grad(elem, ip);
      Jm23 = std::pow(J, -2.0 / 3.0);

      // get the plastic deformation grad quantities
      states->get_tensor("Fp_old", e, ip, Fp);
      Fpinv = minitensor::inverse(Fp);

      // compute the trial state
      Cpinv = Fpinv * minitensor::transpose(Fpinv);
      be = Jm23 * F * Cpinv * minitensor::transpose(F);
      s = mu * minitensor::dev(be);
      mubar = minitensor::trace(be) * mu / num_dims;

      // check the yield condition
      ScalarT smag = minitensor::norm(s);
      states->get_scalar("eqps_old", e, ip, eqps);
      f = smag - sq23 * (Y + K * eqps);

      // plastic increment: return mapping algorithm
      if (f > 1.0e-12) {

        int iter = 0;
        bool converged = false;
        dgam = 0.0;
        ScalarT H = 0.0;
        ScalarT dH = 0.0;
        ScalarT alpha = 0.0;
        ScalarT res = 0.0;

        ScalarT X = 0.0;
        ScalarT R = f;
        ScalarT dRdX = -2.0 * mubar * (1.0 + H / (3.0 * mubar));

        while ((!converged) && (iter < 30)) {
          iter++;
          X = X - R / dRdX;
          alpha = eqps + sq23 * X;
          H = K * alpha;
          dH = K;
          R = smag - (2.0 * mubar * X + sq23 * (Y + H));
          dRdX = -2.0 * mubar * (1.0 + dH / (3.0 * mubar));
          res = std::abs(R);
          if ( (res < 1.0e-11) || (res / Y < 1.0e-11) || (res / f < 1.0e-11))
            converged = true;
          if (iter == 30)
            goal::fail("J2: return mapping failed to converge");
        }

        // updates
        dgam = X;
        N = (1.0 / smag) * s;
        s -= 2.0 * mubar * dgam * N;
        states->set_scalar("eqps", e, ip, alpha);

        // get Fpn
        Fpn = minitensor::exp(dgam * N) * Fp;
        states->set_tensor("Fp", e, ip, Fpn);
      }

      else
        states->set_scalar("eqps", e, ip, eqps);

      // compute stress
      ScalarT p = 0.5 * kappa * (J - 1.0 / J);
      sigma = I * p + s / J;
      states->set_tensor("cauchy", e, ip, sigma);
      for (int i = 0; i < num_dims; ++i)
      for (int j = 0; j < num_dims; ++j)
        cauchy(elem, ip, i, j) = sigma(i, j);

    }
  }
}

template class J2<goal::Traits::Residual, goal::Traits>;
template class J2<goal::Traits::Jacobian, goal::Traits>;

} // end namespace ml
