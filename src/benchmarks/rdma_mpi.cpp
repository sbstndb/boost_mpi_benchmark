#include "types.hpp"
#include <mpi.h>
#include <vector>
#include <algorithm>

void benchmark_rdma_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec;
    MPI_Win win;

    if (rank == 0) {
        int outer_size = vec.data.size();
        std::vector<int> inner_sizes(outer_size);
        int total_elements = 0;
        for (int j = 0; j < outer_size; j++) {
            inner_sizes[j] = vec.data[j].size();
            total_elements += inner_sizes[j];
        }

        std::vector<int> send_buffer(total_elements);
        int offset = 0;
        for (int j = 0; j < outer_size; j++) {
            std::copy(vec.data[j].begin(), vec.data[j].end(), send_buffer.begin() + offset);
            offset += inner_sizes[j];
        }

        MPI_Win_create(send_buffer.data(), total_elements * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        for (int i = 0; i < num_iterations; i++) {
            for (int dest = 1; dest < size; dest++) {
                MPI_Send(&outer_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                MPI_Send(inner_sizes.data(), outer_size, MPI_INT, dest, 1, MPI_COMM_WORLD);
            }

            MPI_Win_fence(0, win);
            MPI_Win_fence(0, win);
        }

        int ack;
        for (int dest = 1; dest < size; dest++) {
            MPI_Recv(&ack, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Win_free(&win);
    }
    else {
        VectorOfVectors vec(0);
        std::vector<int> recv_buffer;

        MPI_Win_create(nullptr, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        for (int i = 0; i < num_iterations; i++) {
            int outer_size;
            MPI_Recv(&outer_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::vector<int> inner_sizes(outer_size);
            MPI_Recv(inner_sizes.data(), outer_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int total_elements = 0;
            for (int j = 0; j < outer_size; j++) {
                total_elements += inner_sizes[j];
            }

            recv_buffer.resize(total_elements);
            vec.data.resize(outer_size);

            MPI_Win_fence(0, win);
            MPI_Get(recv_buffer.data(), total_elements, MPI_INT, 0, 0, total_elements, MPI_INT, win);
            MPI_Win_fence(0, win);

            int offset = 0;
            for (int j = 0; j < outer_size; j++) {
                vec.data[j].resize(inner_sizes[j]);
                std::copy(recv_buffer.begin() + offset, recv_buffer.begin() + offset + inner_sizes[j], vec.data[j].begin());
                offset += inner_sizes[j];
            }
        }

        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Win_free(&win);
    }
}
