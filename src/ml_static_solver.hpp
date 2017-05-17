#ifndef ml_static_solver_hpp
#define ml_static_solver_hpp

/// @file ml_static_solver.hpp

#include <Teuchos_ParameterList.hpp>
#include "ml_solver.hpp"

namespace ml {

using Teuchos::ParameterList;

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

    ParameterList params;
};

} // end namespace ml

#endif
