#ifndef ml_static_solver_hpp
#define ml_static_solver_hpp

/// @file ml_static_solver.hpp

#include <Teuchos_ParameterList.hpp>
#include "ml_solver.hpp"

/// @cond
namespace goal {
class Discretization;
class SolInfo;
class Output;
}
/// @endcond

namespace ml {

using Teuchos::ParameterList;

/// @cond
class Mechanics;
/// @endcond

/// @brief An interface to solve static problems.
class StaticSolver : public Solver {

  public:

    /// @brief Construct the static solver.
    /// @param p The full parameter list describing this solver.
    StaticSolver(ParameterList const& p);

    /// @brief Destroy the static solver.
    ~StaticSolver();

    /// @brief Run the solver
    void solve();

  private:

    void solve_primal();
    void solve_dual();
    void estimate_error();
    void adapt_mesh();

    ParameterList params;
    goal::Discretization* disc;
    ml::Mechanics* mech;
    goal::SolInfo* info;
    goal::Output* output;
};

} // end namespace ml

#endif
