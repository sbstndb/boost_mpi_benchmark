cmake_minimum_required(VERSION 3.10)
project(mpi_benchmark)

# Trouver les dépendances
find_package(MPI REQUIRED)
find_package(Boost REQUIRED COMPONENTS mpi serialization
)
add_executable(mpi_benchmark mpi_benchmark.cpp)
target_include_directories(mpi_benchmark PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_benchmark PUBLIC ${MPI_CXX_LIBRARIES} Boost::mpi Boost::serialization)

