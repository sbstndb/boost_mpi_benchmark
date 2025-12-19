#include "types.hpp"
#include <mpi.h>
#include <vector>

void benchmark_bcast_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec;
    if (rank == 0) {
        int outer_size = vec.data.size();
        std::vector<int> inner_sizes(outer_size);
        for (int j = 0; j < outer_size; j++) {
            inner_sizes[j] = vec.data[j].size();
        }

        for (int i = 0; i < num_iterations; i++) {
            MPI_Bcast(&outer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(inner_sizes.data(), outer_size, MPI_INT, 0, MPI_COMM_WORLD);

            std::vector<MPI_Request> requests;
            for (int j = 0; j < outer_size; j++) {
                MPI_Request req;
                MPI_Ibcast(vec.data[j].data(), inner_sizes[j], MPI_INT, 0, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }

        int ack;
        for (int dest = 1; dest < size; dest++) {
            MPI_Recv(&ack, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else {
        VectorOfVectors vec(0);
        for (int i = 0; i < num_iterations; i++) {
            int outer_size;
            MPI_Bcast(&outer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::vector<int> inner_sizes(outer_size);
            MPI_Bcast(inner_sizes.data(), outer_size, MPI_INT, 0, MPI_COMM_WORLD);

            vec.data.resize(outer_size);
            std::vector<MPI_Request> requests;
            for (int j = 0; j < outer_size; j++) {
                vec.data[j].resize(inner_sizes[j]);
                MPI_Request req;
                MPI_Ibcast(vec.data[j].data(), inner_sizes[j], MPI_INT, 0, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}
