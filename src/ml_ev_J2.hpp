#ifndef ml_ev_J2_hpp
#define ml_ev_J2_hpp

/// @file ml_ev_J2.hpp

#include <Phalanx_Evaluator_Macros.hpp>
#include <goal_dimension.hpp>

/// @cond
namespace Teuchos {
class ParameterList;
}

namespace goal {
class Field;
class States;
}
/// @endcond

namespace ml {

using Teuchos::ParameterList;

PHX_EVALUATOR_CLASS(J2)

  public:

    /// @brief Construct the J2 stress evaluator.
    /// @param u The displacement fields.
    /// @param s The stat
    /// @param mp The parameter list of material properties.
    /// @param type the entity type to operate on.
    J2(
        std::vector<goal::Field*> const& u,
        goal::States* s,
        ParameterList const& mp,
        int type);

  private:

    using Ent = goal::Ent;
    using Dim = goal::Dim;
    using IP = goal::IP;

    int num_ips;
    int num_dims;

    double E;
    double nu;
    double K;
    double Y;
    goal::States* states;

    // input
    PHX::MDField<const ScalarT, Ent, IP, Dim, Dim> def_grad;
    PHX::MDField<const ScalarT, Ent, IP> det_def_grad;

    // output
    PHX::MDField<ScalarT, Ent, IP, Dim, Dim> cauchy;

PHX_EVALUATOR_CLASS_END

} // end namespace ml

#endif
