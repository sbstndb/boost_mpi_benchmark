#include "types.hpp"
#include <mpi.h>
#include <vector>

void benchmark_pack_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec;

    if (rank == 0) {
        int outer_size = vec.data.size();
        std::vector<int> inner_sizes(outer_size);
        int total_elements = 0;
        for (int j = 0; j < outer_size; j++) {
            inner_sizes[j] = vec.data[j].size();
            total_elements += inner_sizes[j];
        }

        int int_pack_size;
        MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &int_pack_size);
        int sizes_pack_size;
        MPI_Pack_size(outer_size, MPI_INT, MPI_COMM_WORLD, &sizes_pack_size);
        int data_pack_size;
        MPI_Pack_size(total_elements, MPI_INT, MPI_COMM_WORLD, &data_pack_size);
        int total_size = int_pack_size + sizes_pack_size + data_pack_size;
        std::vector<char> buffer(total_size);

        for (int i = 0; i < num_iterations; i++) {
            std::vector<MPI_Request> requests;
            int position = 0;
            MPI_Pack(&outer_size, 1, MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            MPI_Pack(inner_sizes.data(), outer_size, MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            for (int j = 0; j < outer_size; j++) {
                MPI_Pack(vec.data[j].data(), inner_sizes[j], MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            }
            MPI_Request req1, req2;
            MPI_Ibcast(&position, 1, MPI_INT, 0, MPI_COMM_WORLD, &req1);
            requests.push_back(req1);
            MPI_Ibcast(buffer.data(), position, MPI_PACKED, 0, MPI_COMM_WORLD, &req2);
            requests.push_back(req2);
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
            int packed_size;
            MPI_Request req1, req2;
            MPI_Ibcast(&packed_size, 1, MPI_INT, 0, MPI_COMM_WORLD, &req1);
            MPI_Wait(&req1, MPI_STATUS_IGNORE);
            std::vector<char> buffer(packed_size);
            MPI_Ibcast(buffer.data(), packed_size, MPI_PACKED, 0, MPI_COMM_WORLD, &req2);
            MPI_Wait(&req2, MPI_STATUS_IGNORE);

            int position = 0;
            int outer_size;
            MPI_Unpack(buffer.data(), packed_size, &position, &outer_size, 1, MPI_INT, MPI_COMM_WORLD);

            std::vector<int> inner_sizes(outer_size);
            MPI_Unpack(buffer.data(), packed_size, &position, inner_sizes.data(), outer_size, MPI_INT, MPI_COMM_WORLD);

            vec.data.resize(outer_size);
            for (int j = 0; j < outer_size; j++) {
                vec.data[j].resize(inner_sizes[j]);
                MPI_Unpack(buffer.data(), packed_size, &position, vec.data[j].data(), inner_sizes[j], MPI_INT, MPI_COMM_WORLD);
            }
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}
