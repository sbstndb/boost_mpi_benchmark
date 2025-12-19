#include "types.hpp"
#include <mpi.h>
#include <vector>

void benchmark_datatype_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec;

    if (rank == 0) {
        int outer_size = vec.data.size();
        std::vector<int> inner_sizes(outer_size);
        for (int j = 0; j < outer_size; j++) {
            inner_sizes[j] = vec.data[j].size();
        }

        for (int i = 0; i < num_iterations; i++) {
            std::vector<MPI_Request> requests;
            for (int dest = 1; dest < size; dest++) {
                MPI_Request req1, req2;
                MPI_Isend(&outer_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, &req1);
                MPI_Isend(inner_sizes.data(), outer_size, MPI_INT, dest, 1, MPI_COMM_WORLD, &req2);
                requests.push_back(req1);
                requests.push_back(req2);
            }

            for (int j = 0; j < outer_size; j++) {
                MPI_Datatype inner_type;
                MPI_Type_contiguous(inner_sizes[j], MPI_INT, &inner_type);
                MPI_Type_commit(&inner_type);
                for (int dest = 1; dest < size; dest++) {
                    MPI_Request req;
                    MPI_Isend(vec.data[j].data(), 1, inner_type, dest, 2 + j, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }
                MPI_Type_free(&inner_type);
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
            MPI_Recv(&outer_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::vector<int> inner_sizes(outer_size);
            MPI_Recv(inner_sizes.data(), outer_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            vec.data.resize(outer_size);
            for (int j = 0; j < outer_size; j++) {
                MPI_Datatype inner_type;
                MPI_Type_contiguous(inner_sizes[j], MPI_INT, &inner_type);
                MPI_Type_commit(&inner_type);
                vec.data[j].resize(inner_sizes[j]);
                MPI_Recv(vec.data[j].data(), 1, inner_type, 0, 2 + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Type_free(&inner_type);
            }
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}
