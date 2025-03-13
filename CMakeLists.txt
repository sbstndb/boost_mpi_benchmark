cmake_minimum_required(VERSION 3.10)
project(mpi_benchmark)

# Trouver les dépendances
find_package(MPI REQUIRED)
find_package(Boost REQUIRED COMPONENTS mpi serialization)
add_executable(mpi_benchmark mpi_benchmark.cpp)
target_include_directories(mpi_benchmark PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_benchmark PUBLIC ${MPI_CXX_LIBRARIES} Boost::mpi Boost::serialization)

add_executable(mpi_benchmark2 mpi_benchmark2.cpp)
target_include_directories(mpi_benchmark2 PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_benchmark2 PUBLIC ${MPI_CXX_LIBRARIES} Boost::mpi Boost::serialization)


add_executable(mpi_benchmark3 mpi_benchmark3.cpp)
target_include_directories(mpi_benchmark3 PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_benchmark3 PUBLIC ${MPI_CXX_LIBRARIES} Boost::mpi Boost::serialization)

add_executable(mpi_benchmark4 mpi_benchmark4.cpp)
target_include_directories(mpi_benchmark4 PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_benchmark4 PUBLIC ${MPI_CXX_LIBRARIES} Boost::mpi Boost::serialization)

add_executable(mpi_benchmark5 mpi_benchmark5.cpp)
target_include_directories(mpi_benchmark5 PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_benchmark4 PUBLIC ${MPI_CXX_LIBRARIES} Boost::mpi Boost::serialization)

add_executable(mpi_benchmark6 mpi_benchmark6.cpp)
target_include_directories(mpi_benchmark6 PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_benchmark4 PUBLIC ${MPI_CXX_LIBRARIES} Boost::mpi Boost::serialization)

add_executable(mpi_benchmark7 mpi_benchmark7.cpp)
target_include_directories(mpi_benchmark7 PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(mpi_benchmark7 PUBLIC ${MPI_CXX_LIBRARIES} Boost::mpi Boost::serialization)



