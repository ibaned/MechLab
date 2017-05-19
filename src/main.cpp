#include <goal_control.hpp>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include "ml_solver.hpp"

int main(int argc, char** argv) {
  goal::initialize();
  GOAL_ALWAYS_ASSERT(argc == 2);
  int status = EXIT_SUCCESS;
  try {
    const char* in = argv[1];
    goal::print("reading input file: %s", in);
    Teuchos::ParameterList p;
    auto pp = Teuchos::Ptr<Teuchos::ParameterList>(&p);
    Teuchos::updateParametersFromYamlFile(in, pp);
    auto solver = ml::create_solver(p);
    solver->solve();
    ml::destroy_solver(solver);
  } catch (std::exception const& ex) {
    goal::print("caught exception:");
    goal::print("%s", ex.what());
    status = EXIT_FAILURE;
  } catch (...) {
    goal::print("caught unknown exception");
    status = EXIT_FAILURE;
  }
  goal::finalize();
  return status;
}
