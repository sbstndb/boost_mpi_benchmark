#include <iostream>
#include <vector>
#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>

#define DEFAULT_OUTER_SIZE 20
#define DEFAULT_INNER_SIZE 1 // Utilisé comme base, mais chaque sous-vecteur aura une taille variable
#define NUM_ITERATIONS 10000

struct VectorOfVectors {
    std::vector<std::vector<int>> data;

    // Constructeur avec tailles variables pour l'émetteur
    VectorOfVectors() {
        data.resize(DEFAULT_OUTER_SIZE);
        for (int i = 0; i < DEFAULT_OUTER_SIZE; i++) {
            // Tailles variables : par exemple, INNER_SIZE + i pour différencier
            data[i].resize(DEFAULT_INNER_SIZE + i * i * i, 0);
        }
    }

    // Constructeur par défaut pour le récepteur
    VectorOfVectors(int) : data() {}

    template <class Archive>
    void serialize(Archive & ar, const unsigned int) {
        ar & data;
    }
};

// Benchmark avec MPI Send/Recv brut
void benchmark_raw_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec; // Émetteur initialise avec tailles variables

    if (rank == 0) {
        int outer_size = vec.data.size();
        std::vector<int> inner_sizes(outer_size);
        for (int j = 0; j < outer_size; j++) {
            inner_sizes[j] = vec.data[j].size();
        }

        for (int i = 0; i < num_iterations; i++) {
            // Envoi de outer_size
            MPI_Send(&outer_size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            // Envoi des tailles de chaque sous-vecteur
            MPI_Send(inner_sizes.data(), outer_size, MPI_INT, 1, 1, MPI_COMM_WORLD);
            // Envoi des données
            for (int j = 0; j < outer_size; j++) {
                MPI_Send(vec.data[j].data(), inner_sizes[j], MPI_INT, 1, 2 + j, MPI_COMM_WORLD);
            }
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        VectorOfVectors vec(0); // Récepteur n'initialise pas à l'avance
        for (int i = 0; i < num_iterations; i++) {
            int outer_size;
            MPI_Recv(&outer_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::vector<int> inner_sizes(outer_size);
            MPI_Recv(inner_sizes.data(), outer_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            vec.data.resize(outer_size);
            for (int j = 0; j < outer_size; j++) {
                vec.data[j].resize(inner_sizes[j]);
                MPI_Recv(vec.data[j].data(), inner_sizes[j], MPI_INT, 0, 2 + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        int ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

// Benchmark avec MPI Pack/Unpack
void benchmark_pack_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec; // Émetteur initialise avec tailles variables

    if (rank == 0) {
        int outer_size = vec.data.size();
        std::vector<int> inner_sizes(outer_size);
        int total_elements = 0;
        for (int j = 0; j < outer_size; j++) {
            inner_sizes[j] = vec.data[j].size();
            total_elements += inner_sizes[j];
        }

        // Calcul de la taille du buffer
        int int_pack_size;
        MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &int_pack_size);
        int sizes_pack_size;
        MPI_Pack_size(outer_size, MPI_INT, MPI_COMM_WORLD, &sizes_pack_size);
        int data_pack_size;
        MPI_Pack_size(total_elements, MPI_INT, MPI_COMM_WORLD, &data_pack_size);
        int total_size = int_pack_size + sizes_pack_size + data_pack_size;
        std::vector<char> buffer(total_size);

        for (int i = 0; i < num_iterations; i++) {
            int position = 0;
            MPI_Pack(&outer_size, 1, MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            MPI_Pack(inner_sizes.data(), outer_size, MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            for (int j = 0; j < outer_size; j++) {
                MPI_Pack(vec.data[j].data(), inner_sizes[j], MPI_INT, buffer.data(), total_size, &position, MPI_COMM_WORLD);
            }
            MPI_Send(&position, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Send(buffer.data(), position, MPI_PACKED, 1, 1, MPI_COMM_WORLD);
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        VectorOfVectors vec(0); // Récepteur n'initialise pas à l'avance
        for (int i = 0; i < num_iterations; i++) {
            int packed_size;
            MPI_Recv(&packed_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<char> buffer(packed_size);
            MPI_Recv(buffer.data(), packed_size, MPI_PACKED, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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

// Benchmark avec MPI Datatype
void benchmark_datatype_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec; // Émetteur initialise avec tailles variables

    if (rank == 0) {
        int outer_size = vec.data.size();
        std::vector<int> inner_sizes(outer_size);
        for (int j = 0; j < outer_size; j++) {
            inner_sizes[j] = vec.data[j].size();
        }

        for (int i = 0; i < num_iterations; i++) {
            // Envoi de outer_size et inner_sizes
            MPI_Send(&outer_size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Send(inner_sizes.data(), outer_size, MPI_INT, 1, 1, MPI_COMM_WORLD);

            // Création et envoi des données avec types dérivés pour chaque sous-vecteur
            for (int j = 0; j < outer_size; j++) {
                MPI_Datatype inner_type;
                MPI_Type_contiguous(inner_sizes[j], MPI_INT, &inner_type);
                MPI_Type_commit(&inner_type);
                MPI_Send(vec.data[j].data(), 1, inner_type, 1, 2 + j, MPI_COMM_WORLD);
                MPI_Type_free(&inner_type);
            }
        }
        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1) {
        VectorOfVectors vec(0); // Récepteur n'initialise pas à l'avance
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

// Benchmark avec Boost MPI
void benchmark_boost_mpi_vector(int num_iterations) {
    boost::mpi::communicator world;
    int rank = world.rank();
    VectorOfVectors vec; // Émetteur initialise avec tailles variables

    if (rank == 0) {
        double start = MPI_Wtime();
        for (int i = 0; i < num_iterations; i++) {
            world.send(1, 0, vec);
        }
        int ack;
        world.recv(1, 1, ack);
        double end = MPI_Wtime();
    }
    else if (rank == 1) {
        VectorOfVectors vec(0); // Récepteur n'initialise pas à l'avance
        for (int i = 0; i < num_iterations; i++) {
            world.recv(0, 0, vec); // Boost gère automatiquement les tailles variables
        }
        int ack = 1;
        world.send(0, 1, ack);
    }
}



// Benchmark avec RDMA (MPI RMA)
void benchmark_rdma_mpi_vector(int rank, int size, int num_iterations) {
    VectorOfVectors vec; // Émetteur initialise avec tailles variables
    MPI_Win win;

    if (rank == 0) {
        int outer_size = vec.data.size();
        std::vector<int> inner_sizes(outer_size);
        int total_elements = 0;
        for (int j = 0; j < outer_size; j++) {
            inner_sizes[j] = vec.data[j].size();
            total_elements += inner_sizes[j];
        }

        // Allocation d'un buffer contigu pour RDMA
        std::vector<int> send_buffer(total_elements);
        int offset = 0;
        for (int j = 0; j < outer_size; j++) {
            std::copy(vec.data[j].begin(), vec.data[j].end(), send_buffer.begin() + offset);
            offset += inner_sizes[j];
        }

        // Création de la fenêtre RMA pour exposer send_buffer
        MPI_Win_create(send_buffer.data(), total_elements * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        for (int i = 0; i < num_iterations; i++) {
            // Envoi des métadonnées (outer_size et inner_sizes) via communication classique
            MPI_Send(&outer_size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Send(inner_sizes.data(), outer_size, MPI_INT, 1, 1, MPI_COMM_WORLD);

            // Début de la période d'accès RMA
            MPI_Win_fence(0, win);

            // Le récepteur ira chercher les données via RDMA (MPI_Get), donc ici on attend juste
            MPI_Win_fence(0, win);
        }

        int ack;
        MPI_Recv(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Win_free(&win);
    }
    else if (rank == 1) {
        VectorOfVectors vec(0); // Récepteur n'initialise pas à l'avance

        // Allocation d'un buffer pour recevoir les données via RDMA
        std::vector<int> recv_buffer; // Taille déterminée dynamiquement après réception des métadonnées

        // Création de la fenêtre RMA (vide côté récepteur pour cet exemple, mais nécessaire pour MPI_Win_fence)
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

            // Début de la période d'accès RMA
            MPI_Win_fence(0, win);

            // Utilisation de MPI_Get pour récupérer les données directement depuis la mémoire de rank 0
            MPI_Get(recv_buffer.data(), total_elements, MPI_INT, 0, 0, total_elements, MPI_INT, win);

            // Fin de la période d'accès RMA
            MPI_Win_fence(0, win);

            // Reconstruction de vec.data à partir du buffer reçu
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
    start = MPI_Wtime();
    benchmark_rdma_mpi_vector(rank, size, NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "RDMA MPI VectorOfVectors: " << (end - start) / NUM_ITERATIONS << " s/op\n";


    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();    
    benchmark_boost_mpi_vector(NUM_ITERATIONS);
    end = MPI_Wtime();
    if (rank == 0) std::cout << "Boost MPI VectorOfVectors: " << (end - start) / NUM_ITERATIONS << " s/op\n";


    MPI_Finalize();
    return 0;
}
