#include <iostream>
#include <vector>
#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>

// Définitions des tailles comme constantes
#define ARRAY_SIZE 100000
#define LARGE_ARRAY_SIZE 1
#define NUM_ITERATIONS 1000

struct SimpleData {
    int a;
    double b;
    double values[ARRAY_SIZE];
    double values2[ARRAY_SIZE];
    double values3[ARRAY_SIZE];
    double values4[ARRAY_SIZE];
    double values5[ARRAY_SIZE];
    double values6[ARRAY_SIZE];

    template <class Archive>
    void serialize(Archive & ar, const unsigned int) {
        ar & a;
        ar & b;
        ar & values;
        ar & values2;
        ar & values3;
        ar & values4;
        ar & values5;
        ar & values6;
    }
};

// Structure dérivée pour MPI_Datatype
struct SimpleDataType : public SimpleData {
    // serialize est hérité
};

// Spécialisation pour Boost MPI
namespace boost {
namespace mpi {
template <>
struct is_mpi_datatype<SimpleDataType> : public mpl::true_ {};
} // namespace mpi
} // namespace boost

struct VectorData {
    std::vector<int> vec;
};

struct ArrayData {
    int id;
    double values[LARGE_ARRAY_SIZE];
};

void benchmark_raw_mpi_simple(int rank, int size, int num_iterations) {
    SimpleData data;
    if (rank == 0) {
        for (int i = 0; i < num_iterations; i++) {
            MPI_Send(&data.a, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Send(&data.b, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
            MPI_Send(data.values, ARRAY_SIZE, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
            MPI_Send(data.values2, ARRAY_SIZE, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD);
            MPI_Send(data.values3, ARRAY_SIZE, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD);
            MPI_Send(data.values4, ARRAY_SIZE, MPI_DOUBLE, 1, 5, MPI_COMM_WORLD);
            MPI_Send(data.values5, ARRAY_SIZE, MPI_DOUBLE, 1, 6, MPI_COMM_WORLD);
            MPI_Send(data.values6, ARRAY_SIZE, MPI_DOUBLE, 1, 7, MPI_COMM_WORLD);
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        for (int i = 0; i < num_iterations; i++) {
            MPI_Recv(&data.a, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&data.b, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(data.values, ARRAY_SIZE, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(data.values2, ARRAY_SIZE, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(data.values3, ARRAY_SIZE, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(data.values4, ARRAY_SIZE, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(data.values5, ARRAY_SIZE, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(data.values6, ARRAY_SIZE, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
}

void benchmark_pack_mpi_simple(int rank, int size, int num_iterations) {
    SimpleData data;

    int int_pack_size;
    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &int_pack_size);
    int double_pack_size;
    MPI_Pack_size(1, MPI_DOUBLE, MPI_COMM_WORLD, &double_pack_size);
    int array_pack_size;
    MPI_Pack_size(ARRAY_SIZE, MPI_DOUBLE, MPI_COMM_WORLD, &array_pack_size);

    int total_size = int_pack_size + double_pack_size + 6 * array_pack_size;
    std::vector<char> buffer(total_size);

    if (rank == 0) {
        for (int i = 0; i < num_iterations; i++) {
            int position = 0;
            MPI_Pack(&data.a, 1, MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            MPI_Pack(&data.b, 1, MPI_DOUBLE, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            MPI_Pack(data.values, ARRAY_SIZE, MPI_DOUBLE, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            MPI_Pack(data.values2, ARRAY_SIZE, MPI_DOUBLE, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            MPI_Pack(data.values3, ARRAY_SIZE, MPI_DOUBLE, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            MPI_Pack(data.values4, ARRAY_SIZE, MPI_DOUBLE, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            MPI_Pack(data.values5, ARRAY_SIZE, MPI_DOUBLE, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            MPI_Pack(data.values6, ARRAY_SIZE, MPI_DOUBLE, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            MPI_Send(buffer.data(), position, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        for (int i = 0; i < num_iterations; i++) {
            MPI_Recv(buffer.data(), total_size, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int position = 0;
            MPI_Unpack(buffer.data(), total_size, &position, &data.a, 1, MPI_INT, MPI_COMM_WORLD);
            MPI_Unpack(buffer.data(), total_size, &position, &data.b, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buffer.data(), total_size, &position, data.values, ARRAY_SIZE, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buffer.data(), total_size, &position, data.values2, ARRAY_SIZE, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buffer.data(), total_size, &position, data.values3, ARRAY_SIZE, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buffer.data(), total_size, &position, data.values4, ARRAY_SIZE, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buffer.data(), total_size, &position, data.values5, ARRAY_SIZE, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buffer.data(), total_size, &position, data.values6, ARRAY_SIZE, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
}

void benchmark_datatype_mpi_simple(int rank, int size, int num_iterations) {
    SimpleData data;
    int block_lengths[4] = {1, 1, ARRAY_SIZE, ARRAY_SIZE};
    MPI_Aint displacements[4];
    MPI_Datatype types[4] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    displacements[0] = 0;
    displacements[1] = offsetof(SimpleData, b);
    displacements[2] = offsetof(SimpleData, values);
    displacements[3] = offsetof(SimpleData, values2);

    MPI_Datatype simpledata_type;
    MPI_Type_create_struct(4, block_lengths, displacements, types, &simpledata_type);
    MPI_Type_commit(&simpledata_type);

    if (rank == 0) {
        for (int i = 0; i < num_iterations; i++) {
            MPI_Send(&data, 1, simpledata_type, 1, 0, MPI_COMM_WORLD);
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        for (int i = 0; i < num_iterations; i++) {
            MPI_Recv(&data, 1, simpledata_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Type_free(&simpledata_type);
}

void benchmark_boost_mpi_simple(int num_iterations) {
    boost::mpi::communicator world;
    int rank = world.rank();
    SimpleData data;

    if (rank == 0) {
        double start = MPI_Wtime();
        for (int i = 0; i < num_iterations; ++i) {
            world.send(1, 0, data);
        }
        int ack;
        world.recv(1, 1, ack);
        double end = MPI_Wtime();
    } else if (rank == 1) {
        for (int i = 0; i < num_iterations; ++i) {
            world.recv(0, 0, data);
        }
        int ack = 1;
        world.send(0, 1, ack);
    }
}

void benchmark_boost_datatype_mpi_simple(int num_iterations) {
    boost::mpi::communicator world;
    int rank = world.rank();
    SimpleDataType data;

    if (rank == 0) {
        double start = MPI_Wtime();
        for (int i = 0; i < num_iterations; ++i) {
            world.send(1, 0, data);
        }
        int ack;
        world.recv(1, 1, ack);
        double end = MPI_Wtime();
    } else if (rank == 1) {
        for (int i = 0; i < num_iterations; ++i) {
            world.recv(0, 0, data);
        }
        int ack = 1;
        world.send(0, 1, ack);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = -1;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start, end;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_raw_mpi_simple(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Raw MPI SimpleData: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_pack_mpi_simple(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Pack MPI SimpleData: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_datatype_mpi_simple(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Datatype MPI SimpleData: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_boost_mpi_simple(NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Boost MPI SimpleData: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    benchmark_boost_datatype_mpi_simple(NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Boost Datatype MPI SimpleData: " << (end - start) / NUM_ITERATIONS << " s/op\n";

    MPI_Finalize();
    return 0;
}
