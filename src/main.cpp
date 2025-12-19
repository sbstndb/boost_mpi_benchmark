#include <iostream>
#include <mpi.h>
#include "types.hpp"
#include "benchmarks.hpp"

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = -1;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start, end;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_raw_mpi_vector(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Raw MPI VectorOfVectors: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_bcast_mpi_vector(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Bcast MPI VectorOfVectors: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_pack_mpi_vector(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Pack MPI VectorOfVectors: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_datatype_mpi_vector(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Datatype MPI VectorOfVectors: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_rdma_mpi_vector(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "RDMA MPI VectorOfVectors: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_boost_mpi_vector(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Boost MPI VectorOfVectors: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_boost_packed_mpi_vector(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Boost Packed MPI VectorOfVectors: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Finalize();
    return 0;
}
