cmake_minimum_required(VERSION 3.14)
project(mpi_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Trouver les dépendances
find_package(MPI REQUIRED)
find_package(Boost REQUIRED COMPONENTS mpi serialization)

# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Créer l'exécutable
add_executable(mpi_benchmark src/mpi_benchmark.cpp)
target_include_directories(mpi_benchmark PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_benchmark PUBLIC
    ${MPI_CXX_LIBRARIES}
    Boost::mpi
    Boost::serialization
    benchmark::benchmark
)
