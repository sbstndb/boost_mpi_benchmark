#ifndef BENCHMARKS_HPP
#define BENCHMARKS_HPP

// Benchmark avec MPI Send/Recv brut
void benchmark_raw_mpi_vector(int rank, int size, int num_iterations);

// Benchmark avec MPI Bcast
void benchmark_bcast_mpi_vector(int rank, int size, int num_iterations);

// Benchmark avec MPI Pack/Unpack
void benchmark_pack_mpi_vector(int rank, int size, int num_iterations);

// Benchmark avec MPI Datatype
void benchmark_datatype_mpi_vector(int rank, int size, int num_iterations);

// Benchmark avec RDMA (MPI RMA)
void benchmark_rdma_mpi_vector(int rank, int size, int num_iterations);

// Benchmark avec Boost MPI
void benchmark_boost_mpi_vector(int rank, int size, int num_iterations);

// Benchmark avec Boost MPI Packed
void benchmark_boost_packed_mpi_vector(int rank, int size, int num_iterations);

#endif // BENCHMARKS_HPP
