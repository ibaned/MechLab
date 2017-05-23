#ifndef ml_ev_traction_hpp
#define ml_ev_traction_hpp

/// @file ml_ev_traction.hpp

#include <Phalanx_Evaluator_Macros.hpp>
#include <goal_dimension.hpp>

/// @cond
namespace goal {
class Indexer;
class Discretization;
class SolutionInfo;
}
/// @endcond

namespace ml {

PHX_EVALUATOR_CLASS_PP(Traction)

  public:

    /// @brief Construct the tractions evaluator.
    /// @brief The traction parameter list.
    /// @param u The displacement fields.
    /// @param bc The boundary condition array.
    /// @param i The linear algebra indexer.
    /// @param type The entity to operate on.
    Traction(
        std::vector<goal::Field*> const& u,
        Teuchos::Array<std::string> const& bc,
        goal::Indexer* i,
        int type);

  private:

    using Node = goal::Node;
    using Ent = goal::Ent;
    using Dim = goal::Dim;
    using IP = goal::IP;

    goal::Indexer* indexer;
    goal::Discretization* disc;
    goal::SolutionInfo* info;

    // input
    std::vector<PHX::MDField<const ScalarT, Ent, Node, IP> > w;

PHX_EVALUATOR_CLASS_END

} // end namespace ml

#endif
