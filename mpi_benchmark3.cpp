#include <iostream>
#include <vector>
#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp> // Nécessaire pour sérialiser std::vector

// Définitions des tailles comme constantes
#define OUTER_SIZE 30    // Nombre de vecteurs internes
#define INNER_SIZE 10000   // Taille de chaque vecteur interne
#define NUM_ITERATIONS 1000

// Structure contenant un vecteur de vecteurs
struct VectorOfVectors {
    std::vector<std::vector<int>> data;

    // Constructeur pour initialiser avec des tailles prédéfinies
    VectorOfVectors() : data(OUTER_SIZE, std::vector<int>(INNER_SIZE, 0)) {}

    // Sérialisation pour Boost MPI
    template <class Archive>
    void serialize(Archive & ar, const unsigned int) {
        ar & data;
    }
};

// Spécialisation pour indiquer que VectorOfVectors peut être utilisé comme MPI_Datatype (via Boost)
namespace boost {
namespace mpi {
template <>
struct is_mpi_datatype<VectorOfVectors> : public mpl::false_ {}; // Pas nativement un MPI_Datatype
} // namespace mpi
} // namespace boost

// Benchmark avec MPI Send/Recv brut
void benchmark_raw_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec;

    if (rank == 0) {
        for (int i = 0; i < num_iterations; i++) {
            // Envoi de la taille externe
            int outer_size = vec.data.size();
            MPI_Send(&outer_size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

            // Envoi de chaque vecteur interne
            for (int j = 0; j < outer_size; j++) {
                int inner_size = vec.data[j].size();
                MPI_Send(&inner_size, 1, MPI_INT, 1, 1 + j, MPI_COMM_WORLD);
                MPI_Send(vec.data[j].data(), inner_size, MPI_INT, 1, OUTER_SIZE + j, MPI_COMM_WORLD);
            }
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        for (int i = 0; i < num_iterations; i++) {
            // Réception de la taille externe
            int outer_size;
            MPI_Recv(&outer_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            vec.data.resize(outer_size);
            // Réception de chaque vecteur interne
            for (int j = 0; j < outer_size; j++) {
                int inner_size;
                MPI_Recv(&inner_size, 1, MPI_INT, 0, 1 + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                vec.data[j].resize(inner_size);
                MPI_Recv(vec.data[j].data(), inner_size, MPI_INT, 0, OUTER_SIZE + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

// Benchmark avec MPI Pack/Unpack
void benchmark_pack_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec;

    // Calcul de la taille du buffer
    int int_pack_size;
    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &int_pack_size); // Pour outer_size
    int inner_size_pack;
    MPI_Pack_size(INNER_SIZE, MPI_INT, MPI_COMM_WORLD, &inner_size_pack); // Pour chaque vecteur interne

    int total_size = int_pack_size + OUTER_SIZE * (int_pack_size + inner_size_pack);
    std::vector<char> buffer(total_size);

    if (rank == 0) {
        for (int i = 0; i < num_iterations; i++) {
            int position = 0;
            int outer_size = vec.data.size();
            MPI_Pack(&outer_size, 1, MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);

            for (int j = 0; j < outer_size; j++) {
                int inner_size = vec.data[j].size();
                MPI_Pack(&inner_size, 1, MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
                MPI_Pack(vec.data[j].data(), inner_size, MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            }
            MPI_Send(buffer.data(), position, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        for (int i = 0; i < num_iterations; i++) {
            MPI_Recv(buffer.data(), total_size, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int position = 0;
            int outer_size;
            MPI_Unpack(buffer.data(), total_size, &position, &outer_size, 1, MPI_INT, MPI_COMM_WORLD);

            vec.data.resize(outer_size);
            for (int j = 0; j < outer_size; j++) {
                int inner_size;
                MPI_Unpack(buffer.data(), total_size, &position, &inner_size, 1, MPI_INT, MPI_COMM_WORLD);
                vec.data[j].resize(inner_size);
                MPI_Unpack(buffer.data(), total_size, &position, vec.data[j].data(), inner_size, MPI_INT, MPI_COMM_WORLD);
            }
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

// Benchmark avec MPI Datatype (plus complexe pour un vecteur de vecteurs)
void benchmark_datatype_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec;

    // Création d'un type pour un vecteur interne (INNER_SIZE ints)
    MPI_Datatype inner_type;
    MPI_Type_contiguous(INNER_SIZE, MPI_INT, &inner_type);
    MPI_Type_commit(&inner_type);

    // Création d'un type pour le vecteur externe (OUTER_SIZE inner_types)
    MPI_Datatype vector_type;
    MPI_Type_contiguous(OUTER_SIZE, inner_type, &vector_type);
    MPI_Type_commit(&vector_type);

    if (rank == 0) {
        for (int i = 0; i < num_iterations; i++) {
            // Note : Cette approche suppose une taille fixe et contiguë, ce qui n'est pas idéal pour std::vector
            // On envoie chaque vecteur interne séparément pour simplifier
            for (int j = 0; j < OUTER_SIZE; j++) {
                MPI_Send(vec.data[j].data(), 1, inner_type, 1, j, MPI_COMM_WORLD);
            }
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        for (int i = 0; i < num_iterations; i++) {
            for (int j = 0; j < OUTER_SIZE; j++) {
                MPI_Recv(vec.data[j].data(), 1, inner_type, 0, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&inner_type);
    MPI_Type_free(&vector_type);
}

// Benchmark avec Boost MPI
void benchmark_boost_mpi_vector(int num_iterations) {
    boost::mpi::communicator world;
    int rank = world.rank();
    VectorOfVectors vec;

    if (rank == 0) {
        double start = MPI_Wtime();
        for (int i = 0; i < num_iterations; i++) {
            world.send(1, 0, vec);
        }
        int ack;
        world.recv(1, 1, ack);
        double end = MPI_Wtime();
        if (rank == 0) std::cout << "Boost MPI VectorOfVectors: " << (end - start) / NUM_ITERATIONS << " s/op\n";
    }
    else if (rank == 1) {
        for (int i = 0; i < num_iterations; i++) {
            world.recv(0, 0, vec);
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
    benchmark_raw_mpi_vector(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Raw MPI VectorOfVectors: " << (end - start) / NUM_ITERATIONS << " s/op\n";

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
    benchmark_boost_mpi_vector(NUM_ITERATIONS); // Temps déjà calculé dans la fonction

    MPI_Finalize();
    return 0;
}
