if(NOT BUILD_TESTING)
  return()
endif()

set(MPIEXE mpirun)
set(MPIFLAGS -np)

include(meshgen.cmake)
