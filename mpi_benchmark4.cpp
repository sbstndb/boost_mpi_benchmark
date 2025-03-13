#include <iostream>
#include <vector>
#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>

// Définitions des tailles par défaut pour l'émetteur
#define DEFAULT_OUTER_SIZE 30
#define DEFAULT_INNER_SIZE 10000
#define NUM_ITERATIONS 1000

// Structure contenant un vecteur de vecteurs
struct VectorOfVectors {
    std::vector<std::vector<int>> data;

    // Constructeur pour initialiser avec des tailles prédéfinies (uniquement pour l'émetteur)
    VectorOfVectors(int outer_size = DEFAULT_OUTER_SIZE, int inner_size = DEFAULT_INNER_SIZE)
        : data(outer_size, std::vector<int>(inner_size, 0)) {}

    // Sérialisation pour Boost MPI
    template <class Archive>
    void serialize(Archive & ar, const unsigned int) {
        ar & data;
    }
};

// Benchmark avec MPI Send/Recv brut
void benchmark_raw_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec; // Émetteur initialise avec tailles par défaut

    if (rank == 0) {
        int outer_size = vec.data.size();
        int inner_size = vec.data[0].size(); // Suppose tailles uniformes pour simplifier

        for (int i = 0; i < num_iterations; i++) {
            // Envoi des tailles
            MPI_Send(&outer_size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Send(&inner_size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);

            // Envoi des données
            for (int j = 0; j < outer_size; j++) {
                MPI_Send(vec.data[j].data(), inner_size, MPI_INT, 1, 2 + j, MPI_COMM_WORLD);
            }
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        int outer_size, inner_size;
        for (int i = 0; i < num_iterations; i++) {
            // Réception des tailles
            MPI_Recv(&outer_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&inner_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Redimensionnement de la structure
            vec.data.resize(outer_size);
            for (int j = 0; j < outer_size; j++) {
                vec.data[j].resize(inner_size);
                MPI_Recv(vec.data[j].data(), inner_size, MPI_INT, 0, 2 + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

void benchmark_pack_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec; // Émetteur initialise avec tailles par défaut

    if (rank == 0) {
        int outer_size = vec.data.size();
        int inner_size = vec.data[0].size();

        // Calcul de la taille du buffer pour une itération
        int int_pack_size;
        MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &int_pack_size);
        int data_pack_size;
        MPI_Pack_size(outer_size * inner_size, MPI_INT, MPI_COMM_WORLD, &data_pack_size);
        int total_size = 2 * int_pack_size + data_pack_size;
        std::vector<char> buffer(total_size);

        for (int i = 0; i < num_iterations; i++) {
            int position = 0;
            MPI_Pack(&outer_size, 1, MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            MPI_Pack(&inner_size, 1, MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            for (int j = 0; j < outer_size; j++) {
                MPI_Pack(vec.data[j].data(), inner_size, MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            }

            // Envoi de la taille du buffer empaqueté d'abord
            MPI_Send(&position, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            // Envoi des données empaquetées
            MPI_Send(buffer.data(), position, MPI_PACKED, 1, 1, MPI_COMM_WORLD);
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        for (int i = 0; i < num_iterations; i++) {
            // Réception de la taille du buffer empaqueté
            int packed_size;
            MPI_Recv(&packed_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Allocation du buffer avec la taille exacte
            std::vector<char> buffer(packed_size);

            // Réception des données empaquetées
            MPI_Recv(buffer.data(), packed_size, MPI_PACKED, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Dépaquetage
            int position = 0;
            int outer_size, inner_size;
            MPI_Unpack(buffer.data(), packed_size, &position, &outer_size, 1, MPI_INT, MPI_COMM_WORLD);
            MPI_Unpack(buffer.data(), packed_size, &position, &inner_size, 1, MPI_INT, MPI_COMM_WORLD);

            vec.data.resize(outer_size);
            for (int j = 0; j < outer_size; j++) {
                vec.data[j].resize(inner_size);
                MPI_Unpack(buffer.data(), packed_size, &position, vec.data[j].data(), inner_size, MPI_INT, MPI_COMM_WORLD);
            }
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}


// Benchmark avec MPI Datatype
void benchmark_datatype_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec; // Émetteur initialise avec tailles par défaut

    if (rank == 0) {
        int outer_size = vec.data.size();
        int inner_size = vec.data[0].size();

        // Création du type pour un vecteur interne
        MPI_Datatype inner_type;
        MPI_Type_contiguous(inner_size, MPI_INT, &inner_type);
        MPI_Type_commit(&inner_type);

        // Création du type pour le vecteur externe
        MPI_Datatype vector_type;
        MPI_Type_contiguous(outer_size, inner_type, &vector_type);
        MPI_Type_commit(&vector_type);

        for (int i = 0; i < num_iterations; i++) {
            // Envoi des tailles
            MPI_Send(&outer_size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Send(&inner_size, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
            // Envoi des données
            for (int j = 0; j < outer_size; j++) {
                MPI_Send(vec.data[j].data(), 1, inner_type, 1, 2 + j, MPI_COMM_WORLD);
            }
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Type_free(&inner_type);
        MPI_Type_free(&vector_type);
    }
    else if (rank == 1) {
        for (int i = 0; i < num_iterations; i++) {
            // Réception des tailles
            int outer_size, inner_size;
            MPI_Recv(&outer_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&inner_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Création dynamique des types MPI
            MPI_Datatype inner_type;
            MPI_Type_contiguous(inner_size, MPI_INT, &inner_type);
            MPI_Type_commit(&inner_type);

            // Redimensionnement
            vec.data.resize(outer_size);
            for (int j = 0; j < outer_size; j++) {
                vec.data[j].resize(inner_size);
                MPI_Recv(vec.data[j].data(), 1, inner_type, 0, 2 + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            MPI_Type_free(&inner_type);
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

// Benchmark avec Boost MPI
void benchmark_boost_mpi_vector(int num_iterations) {
    boost::mpi::communicator world;
    int rank = world.rank();
    VectorOfVectors vec; // Émetteur initialise avec tailles par défaut

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
            world.recv(0, 0, vec); // Boost gère automatiquement les tailles
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
