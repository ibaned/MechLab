#ifndef ml_ev_kinematics_hpp
#define ml_ev_kinematics_hpp

#include <Phalanx_Evaluator_Macros.hpp>
#include <goal_dimension.hpp>

namespace ml {

PHX_EVALUATOR_CLASS(Kinematics)

  public:

    /// @brief Construct the kinematics evaluator.
    /// @param u The displacement fields.
    /// @param t The entity type to operate on.
    Kinematics(std::vector<goal::Field*> const& u, int t);

  private:

    using Ent = goal::Ent;
    using IP = goal::IP;
    using Dim = goal::Dim;

    int num_ips;
    int num_dims;

    // input
    std::vector<PHX::MDField<const ScalarT, Ent, IP, Dim> > grad_u;

    // output
    PHX::MDField<ScalarT, Ent, IP, Dim, Dim> def_grad;
    PHX::MDField<ScalarT, Ent, IP> det_def_grad;

PHX_EVALUATOR_CLASS_END

} // end namespace ml

#endif
