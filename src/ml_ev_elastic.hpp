#ifndef ml_ev_elastic_hpp
#define ml_ev_elastic_hpp

/// @file ml_ev_elastic.hpp

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

PHX_EVALUATOR_CLASS(Elastic)

  public:

    /// @brief Construct the elastic stress evaluator.
    /// @param u The displacement fields.
    /// @param s The state fields structure.
    /// @param mp A parameter list of material properties.
    /// @param type The entity type to operate on.
    Elastic(
        std::vector<goal::Field*> const& u,
        goal::States* s,
        ParameterList const& mp,
        int type);

  private:

    using Ent = goal::Ent;
    using IP = goal::IP;
    using Dim = goal::Dim;

    int num_ips;
    int num_dims;
    
    double E;
    double nu;
    goal::States* states;

    // input
    std::vector<PHX::MDField<const ScalarT, Ent, IP, Dim> > grad_u;

    // output
    PHX::MDField<ScalarT, Ent, IP, Dim, Dim> cauchy;

PHX_EVALUATOR_CLASS_END

} // end namespace ml

#endif
