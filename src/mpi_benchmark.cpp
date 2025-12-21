#include <iostream>
#include <vector>
#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/mpi/packed_oarchive.hpp>
#include <boost/mpi/packed_iarchive.hpp>
#include <benchmark/benchmark.h>

// Inner iterations scaled by data size to keep benchmark time reasonable
#define INNER_ITERATIONS_SMALL   10000
#define INNER_ITERATIONS_MEDIUM  10000
#define INNER_ITERATIONS_LARGE   1000
#define INNER_ITERATIONS_XLARGE  100
#define INNER_ITERATIONS_XXLARGE 10
#define INNER_ITERATIONS_XXXLARGE 1

// Helper to get inner iterations based on base_size
// Adapted for new formula: total = base_size * 55
inline int get_inner_iterations(int base_size) {
    if (base_size <= 50) return INNER_ITERATIONS_SMALL;      // ~11 KB
    if (base_size <= 500) return INNER_ITERATIONS_MEDIUM;    // ~107 KB
    if (base_size <= 5000) return INNER_ITERATIONS_LARGE;    // ~1 MB
    if (base_size <= 50000) return INNER_ITERATIONS_XLARGE;  // ~10 MB
    if (base_size <= 500000) return INNER_ITERATIONS_XXLARGE; // ~105 MB
    return INNER_ITERATIONS_XXXLARGE;                         // ~420 MB
}

// Variables globales MPI
static int g_rank = -1;
static int g_size = 0;

struct VectorOfVectors {
    std::vector<std::vector<int>> data;

    // Constructeur paramétré : outer_size vecteurs, taille = base_size * (i+1)²
    // Ratio 25:1 entre le plus grand et le plus petit vecteur
    VectorOfVectors(int outer_size, int base_size) {
        data.resize(outer_size);
        for (int i = 0; i < outer_size; i++) {
            int factor = (i + 1) * (i + 1);  // 1, 4, 9, 16, 25
            data[i].resize(base_size * factor, 0);
        }
    }

    // Constructeur vide pour réception
    VectorOfVectors() : data() {}

    template <class Archive>
    void serialize(Archive & ar, const unsigned int) {
        ar & data;
    }

    // Calcule la taille totale en nombre d'ints
    int total_elements() const {
        int total = 0;
        for (const auto& v : data) {
            total += v.size();
        }
        return total;
    }
};

// NullReporter pour les ranks > 0
class NullReporter : public benchmark::BenchmarkReporter {
public:
    bool ReportContext(const Context&) override { return true; }
    void ReportRuns(const std::vector<Run>&) override {}
    void Finalize() override {}
};

// Helper pour créer un nom de benchmark avec la taille
static void SetBytesProcessed(benchmark::State& state, const VectorOfVectors& vec, int inner_iters) {
    state.SetBytesProcessed(state.iterations() * inner_iters * vec.total_elements() * sizeof(int));
}

// ============================================================================
// Benchmark Raw MPI
// ============================================================================
static void BM_RawMPI(benchmark::State& state) {
    int outer_size_param = state.range(0);
    int base_size_param = state.range(1);
    int inner_iters = get_inner_iterations(base_size_param);

    VectorOfVectors vec(outer_size_param, base_size_param);
    int outer_size = vec.data.size();
    std::vector<int> inner_sizes(outer_size);
    for (int j = 0; j < outer_size; j++) {
        inner_sizes[j] = vec.data[j].size();
    }

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < inner_iters; iter++) {
            if (g_rank == 0) {
                std::vector<MPI_Request> requests;
                for (int dest = 1; dest < g_size; dest++) {
                    MPI_Request req1, req2;
                    MPI_Isend(&outer_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, &req1);
                    requests.push_back(req1);
                    MPI_Isend(inner_sizes.data(), outer_size, MPI_INT, dest, 1, MPI_COMM_WORLD, &req2);
                    requests.push_back(req2);
                    for (int j = 0; j < outer_size; j++) {
                        MPI_Request req3;
                        MPI_Isend(vec.data[j].data(), inner_sizes[j], MPI_INT, dest, 2 + j, MPI_COMM_WORLD, &req3);
                        requests.push_back(req3);
                    }
                }
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            } else {
                VectorOfVectors recv_vec;
                std::vector<MPI_Request> requests;
                int recv_outer_size;
                MPI_Request req1, req2;

                MPI_Irecv(&recv_outer_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req1);
                MPI_Wait(&req1, MPI_STATUS_IGNORE);
                std::vector<int> recv_inner_sizes(recv_outer_size);

                MPI_Irecv(recv_inner_sizes.data(), recv_outer_size, MPI_INT, 0, 1, MPI_COMM_WORLD, &req2);
                recv_vec.data.resize(recv_outer_size);
                MPI_Wait(&req2, MPI_STATUS_IGNORE);

                for (int j = 0; j < recv_outer_size; j++) {
                    MPI_Request req;
                    recv_vec.data[j].resize(recv_inner_sizes[j]);
                    MPI_Irecv(recv_vec.data[j].data(), recv_inner_sizes[j], MPI_INT, 0, 2 + j, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            }
        }

        // Synchronisation finale
        if (g_rank == 0) {
            int ack;
            for (int dest = 1; dest < g_size; dest++) {
                MPI_Recv(&ack, 1, MPI_INT, dest, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            int ack = 1;
            MPI_Send(&ack, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
        }

        double elapsed = MPI_Wtime() - start;
        double per_op = elapsed / inner_iters;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
    SetBytesProcessed(state, vec, inner_iters);
}

// ============================================================================
// Benchmark Bcast MPI
// ============================================================================
static void BM_BcastMPI(benchmark::State& state) {
    int outer_size_param = state.range(0);
    int base_size_param = state.range(1);
    int inner_iters = get_inner_iterations(base_size_param);

    VectorOfVectors vec(outer_size_param, base_size_param);
    int outer_size = vec.data.size();
    std::vector<int> inner_sizes(outer_size);
    for (int j = 0; j < outer_size; j++) {
        inner_sizes[j] = vec.data[j].size();
    }

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < inner_iters; iter++) {
            if (g_rank == 0) {
                MPI_Bcast(&outer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(inner_sizes.data(), outer_size, MPI_INT, 0, MPI_COMM_WORLD);

                std::vector<MPI_Request> requests;
                for (int j = 0; j < outer_size; j++) {
                    MPI_Request req;
                    MPI_Ibcast(vec.data[j].data(), inner_sizes[j], MPI_INT, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            } else {
                VectorOfVectors recv_vec;
                int recv_outer_size;
                MPI_Bcast(&recv_outer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
                std::vector<int> recv_inner_sizes(recv_outer_size);
                MPI_Bcast(recv_inner_sizes.data(), recv_outer_size, MPI_INT, 0, MPI_COMM_WORLD);

                recv_vec.data.resize(recv_outer_size);
                std::vector<MPI_Request> requests;
                for (int j = 0; j < recv_outer_size; j++) {
                    recv_vec.data[j].resize(recv_inner_sizes[j]);
                    MPI_Request req;
                    MPI_Ibcast(recv_vec.data[j].data(), recv_inner_sizes[j], MPI_INT, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            }
        }

        // Synchronisation finale
        if (g_rank == 0) {
            int ack;
            for (int dest = 1; dest < g_size; dest++) {
                MPI_Recv(&ack, 1, MPI_INT, dest, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            int ack = 1;
            MPI_Send(&ack, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
        }

        double elapsed = MPI_Wtime() - start;
        double per_op = elapsed / inner_iters;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
    SetBytesProcessed(state, vec, inner_iters);
}

// ============================================================================
// Benchmark Pack MPI
// ============================================================================
static void BM_PackMPI(benchmark::State& state) {
    int outer_size_param = state.range(0);
    int base_size_param = state.range(1);
    int inner_iters = get_inner_iterations(base_size_param);

    VectorOfVectors vec(outer_size_param, base_size_param);
    int outer_size = vec.data.size();
    std::vector<int> inner_sizes(outer_size);
    int total_elements = 0;
    for (int j = 0; j < outer_size; j++) {
        inner_sizes[j] = vec.data[j].size();
        total_elements += inner_sizes[j];
    }

    int int_pack_size, sizes_pack_size, data_pack_size;
    MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &int_pack_size);
    MPI_Pack_size(outer_size, MPI_INT, MPI_COMM_WORLD, &sizes_pack_size);
    MPI_Pack_size(total_elements, MPI_INT, MPI_COMM_WORLD, &data_pack_size);
    int total_size = int_pack_size + sizes_pack_size + data_pack_size;
    std::vector<char> buffer(total_size);

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < inner_iters; iter++) {
            if (g_rank == 0) {
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
            } else {
                VectorOfVectors recv_vec;
                int packed_size;
                MPI_Request req1, req2;
                MPI_Ibcast(&packed_size, 1, MPI_INT, 0, MPI_COMM_WORLD, &req1);
                MPI_Wait(&req1, MPI_STATUS_IGNORE);
                std::vector<char> recv_buffer(packed_size);
                MPI_Ibcast(recv_buffer.data(), packed_size, MPI_PACKED, 0, MPI_COMM_WORLD, &req2);
                MPI_Wait(&req2, MPI_STATUS_IGNORE);

                int position = 0;
                int recv_outer_size;
                MPI_Unpack(recv_buffer.data(), packed_size, &position, &recv_outer_size, 1, MPI_INT, MPI_COMM_WORLD);
                std::vector<int> recv_inner_sizes(recv_outer_size);
                MPI_Unpack(recv_buffer.data(), packed_size, &position, recv_inner_sizes.data(), recv_outer_size, MPI_INT, MPI_COMM_WORLD);

                recv_vec.data.resize(recv_outer_size);
                for (int j = 0; j < recv_outer_size; j++) {
                    recv_vec.data[j].resize(recv_inner_sizes[j]);
                    MPI_Unpack(recv_buffer.data(), packed_size, &position, recv_vec.data[j].data(), recv_inner_sizes[j], MPI_INT, MPI_COMM_WORLD);
                }
            }
        }

        // Synchronisation finale
        if (g_rank == 0) {
            int ack;
            for (int dest = 1; dest < g_size; dest++) {
                MPI_Recv(&ack, 1, MPI_INT, dest, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            int ack = 1;
            MPI_Send(&ack, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
        }

        double elapsed = MPI_Wtime() - start;
        double per_op = elapsed / inner_iters;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
    SetBytesProcessed(state, vec, inner_iters);
}

// ============================================================================
// Benchmark Datatype MPI
// ============================================================================
static void BM_DatatypeMPI(benchmark::State& state) {
    int outer_size_param = state.range(0);
    int base_size_param = state.range(1);
    int inner_iters = get_inner_iterations(base_size_param);

    VectorOfVectors vec(outer_size_param, base_size_param);
    int outer_size = vec.data.size();
    std::vector<int> inner_sizes(outer_size);
    for (int j = 0; j < outer_size; j++) {
        inner_sizes[j] = vec.data[j].size();
    }

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < inner_iters; iter++) {
            if (g_rank == 0) {
                std::vector<MPI_Request> requests;
                for (int dest = 1; dest < g_size; dest++) {
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
                    for (int dest = 1; dest < g_size; dest++) {
                        MPI_Request req;
                        MPI_Isend(vec.data[j].data(), 1, inner_type, dest, 2 + j, MPI_COMM_WORLD, &req);
                        requests.push_back(req);
                    }
                    MPI_Type_free(&inner_type);
                }
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            } else {
                VectorOfVectors recv_vec;
                int recv_outer_size;
                MPI_Recv(&recv_outer_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<int> recv_inner_sizes(recv_outer_size);
                MPI_Recv(recv_inner_sizes.data(), recv_outer_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                recv_vec.data.resize(recv_outer_size);
                for (int j = 0; j < recv_outer_size; j++) {
                    MPI_Datatype inner_type;
                    MPI_Type_contiguous(recv_inner_sizes[j], MPI_INT, &inner_type);
                    MPI_Type_commit(&inner_type);
                    recv_vec.data[j].resize(recv_inner_sizes[j]);
                    MPI_Recv(recv_vec.data[j].data(), 1, inner_type, 0, 2 + j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Type_free(&inner_type);
                }
            }
        }

        // Synchronisation finale
        if (g_rank == 0) {
            int ack;
            for (int dest = 1; dest < g_size; dest++) {
                MPI_Recv(&ack, 1, MPI_INT, dest, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            int ack = 1;
            MPI_Send(&ack, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
        }

        double elapsed = MPI_Wtime() - start;
        double per_op = elapsed / inner_iters;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
    SetBytesProcessed(state, vec, inner_iters);
}

// ============================================================================
// Benchmark RDMA MPI
// ============================================================================
static void BM_RDMAMPI(benchmark::State& state) {
    int outer_size_param = state.range(0);
    int base_size_param = state.range(1);
    int inner_iters = get_inner_iterations(base_size_param);

    VectorOfVectors vec(outer_size_param, base_size_param);
    MPI_Win win;

    int outer_size = vec.data.size();
    std::vector<int> inner_sizes(outer_size);
    int total_elements = 0;
    for (int j = 0; j < outer_size; j++) {
        inner_sizes[j] = vec.data[j].size();
        total_elements += inner_sizes[j];
    }

    std::vector<int> send_buffer;
    std::vector<int> recv_buffer;

    if (g_rank == 0) {
        send_buffer.resize(total_elements);
        int offset = 0;
        for (int j = 0; j < outer_size; j++) {
            std::copy(vec.data[j].begin(), vec.data[j].end(), send_buffer.begin() + offset);
            offset += inner_sizes[j];
        }
        MPI_Win_create(send_buffer.data(), total_elements * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    } else {
        MPI_Win_create(nullptr, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < inner_iters; iter++) {
            if (g_rank == 0) {
                for (int dest = 1; dest < g_size; dest++) {
                    MPI_Send(&outer_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                    MPI_Send(inner_sizes.data(), outer_size, MPI_INT, dest, 1, MPI_COMM_WORLD);
                }
                MPI_Win_fence(0, win);
                MPI_Win_fence(0, win);
            } else {
                VectorOfVectors recv_vec;
                int recv_outer_size;
                MPI_Recv(&recv_outer_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<int> recv_inner_sizes(recv_outer_size);
                MPI_Recv(recv_inner_sizes.data(), recv_outer_size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int recv_total = 0;
                for (int j = 0; j < recv_outer_size; j++) {
                    recv_total += recv_inner_sizes[j];
                }

                recv_buffer.resize(recv_total);
                recv_vec.data.resize(recv_outer_size);

                MPI_Win_fence(0, win);
                MPI_Get(recv_buffer.data(), recv_total, MPI_INT, 0, 0, recv_total, MPI_INT, win);
                MPI_Win_fence(0, win);

                int offset = 0;
                for (int j = 0; j < recv_outer_size; j++) {
                    recv_vec.data[j].resize(recv_inner_sizes[j]);
                    std::copy(recv_buffer.begin() + offset, recv_buffer.begin() + offset + recv_inner_sizes[j], recv_vec.data[j].begin());
                    offset += recv_inner_sizes[j];
                }
            }
        }

        // Synchronisation finale
        if (g_rank == 0) {
            int ack;
            for (int dest = 1; dest < g_size; dest++) {
                MPI_Recv(&ack, 1, MPI_INT, dest, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            int ack = 1;
            MPI_Send(&ack, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
        }

        double elapsed = MPI_Wtime() - start;
        double per_op = elapsed / inner_iters;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }

    MPI_Win_free(&win);
    SetBytesProcessed(state, vec, inner_iters);
}

// ============================================================================
// Benchmark Boost MPI
// ============================================================================
static void BM_BoostMPI(benchmark::State& state) {
    int outer_size_param = state.range(0);
    int base_size_param = state.range(1);
    int inner_iters = get_inner_iterations(base_size_param);

    boost::mpi::communicator world;
    VectorOfVectors vec(outer_size_param, base_size_param);

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < inner_iters; iter++) {
            if (g_rank == 0) {
                for (int dest = 1; dest < g_size; dest++) {
                    world.send(dest, 0, vec);
                }
            } else {
                VectorOfVectors recv_vec;
                world.recv(0, 0, recv_vec);
            }
        }

        // Synchronisation finale
        if (g_rank == 0) {
            int ack;
            for (int dest = 1; dest < g_size; dest++) {
                world.recv(dest, 1, ack);
            }
        } else {
            int ack = 1;
            world.send(0, 1, ack);
        }

        double elapsed = MPI_Wtime() - start;
        double per_op = elapsed / inner_iters;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
    SetBytesProcessed(state, vec, inner_iters);
}

// ============================================================================
// Benchmark Boost Packed MPI
// ============================================================================
static void BM_BoostPackedMPI(benchmark::State& state) {
    int outer_size_param = state.range(0);
    int base_size_param = state.range(1);
    int inner_iters = get_inner_iterations(base_size_param);

    boost::mpi::communicator world;
    VectorOfVectors vec(outer_size_param, base_size_param);

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < inner_iters; iter++) {
            if (g_rank == 0) {
                boost::mpi::packed_oarchive::buffer_type buffer;
                boost::mpi::packed_oarchive oa(world, buffer);
                oa << vec;
                for (int dest = 1; dest < g_size; dest++) {
                    world.send(dest, 0, buffer);
                }
            } else {
                VectorOfVectors recv_vec;
                world.recv(0, 0, recv_vec);
            }
        }

        // Synchronisation finale
        if (g_rank == 0) {
            int ack;
            for (int dest = 1; dest < g_size; dest++) {
                world.recv(dest, 1, ack);
            }
        } else {
            int ack = 1;
            world.send(0, 1, ack);
        }

        double elapsed = MPI_Wtime() - start;
        double per_op = elapsed / inner_iters;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
    SetBytesProcessed(state, vec, inner_iters);
}

// ============================================================================
// Benchmarks 1D - Mesure du coût de communication pur (buffer contigu)
// ============================================================================

// Helper pour obtenir les inner iterations pour 1D basé sur la taille du tableau
// Tailles équivalentes aux benchmarks 2D (base_size * 55)
inline int get_inner_iterations_1d(int array_size) {
    if (array_size <= 2750) return INNER_ITERATIONS_SMALL;        // ~11 KB
    if (array_size <= 27500) return INNER_ITERATIONS_MEDIUM;      // ~107 KB
    if (array_size <= 275000) return INNER_ITERATIONS_LARGE;      // ~1 MB
    if (array_size <= 2750000) return INNER_ITERATIONS_XLARGE;    // ~10 MB
    if (array_size <= 27500000) return INNER_ITERATIONS_XXLARGE;  // ~105 MB
    return INNER_ITERATIONS_XXXLARGE;                              // ~420 MB
}

// Helper pour SetBytesProcessed pour 1D
static void SetBytesProcessed1D(benchmark::State& state, int array_size, int inner_iters) {
    state.SetBytesProcessed(state.iterations() * inner_iters * array_size * sizeof(int));
}

// ============================================================================
// Benchmark Raw MPI 1D - Point-to-point avec buffer contigu
// ============================================================================
static void BM_RawMPI_1D(benchmark::State& state) {
    int array_size = state.range(0);
    int inner_iters = get_inner_iterations_1d(array_size);

    std::vector<int> send_buffer(array_size, 42);
    std::vector<int> recv_buffer(array_size);

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < inner_iters; iter++) {
            if (g_rank == 0) {
                std::vector<MPI_Request> requests(g_size - 1);
                for (int dest = 1; dest < g_size; dest++) {
                    MPI_Isend(send_buffer.data(), array_size, MPI_INT, dest, 0, MPI_COMM_WORLD, &requests[dest - 1]);
                }
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            } else {
                MPI_Recv(recv_buffer.data(), array_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Synchronisation finale
        if (g_rank == 0) {
            int ack;
            for (int dest = 1; dest < g_size; dest++) {
                MPI_Recv(&ack, 1, MPI_INT, dest, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            int ack = 1;
            MPI_Send(&ack, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
        }

        double elapsed = MPI_Wtime() - start;
        double per_op = elapsed / inner_iters;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
    SetBytesProcessed1D(state, array_size, inner_iters);
}

// ============================================================================
// Benchmark Bcast MPI 1D - Broadcast collectif avec buffer contigu
// ============================================================================
static void BM_BcastMPI_1D(benchmark::State& state) {
    int array_size = state.range(0);
    int inner_iters = get_inner_iterations_1d(array_size);

    std::vector<int> buffer(array_size, g_rank == 0 ? 42 : 0);

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < inner_iters; iter++) {
            MPI_Bcast(buffer.data(), array_size, MPI_INT, 0, MPI_COMM_WORLD);
        }

        // Synchronisation finale
        if (g_rank == 0) {
            int ack;
            for (int dest = 1; dest < g_size; dest++) {
                MPI_Recv(&ack, 1, MPI_INT, dest, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            int ack = 1;
            MPI_Send(&ack, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
        }

        double elapsed = MPI_Wtime() - start;
        double per_op = elapsed / inner_iters;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
    SetBytesProcessed1D(state, array_size, inner_iters);
}

// ============================================================================
// Benchmark RDMA MPI 1D - One-sided avec buffer contigu
// ============================================================================
static void BM_RDMAMPI_1D(benchmark::State& state) {
    int array_size = state.range(0);
    int inner_iters = get_inner_iterations_1d(array_size);

    std::vector<int> buffer(array_size, g_rank == 0 ? 42 : 0);
    MPI_Win win;

    if (g_rank == 0) {
        MPI_Win_create(buffer.data(), array_size * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    } else {
        MPI_Win_create(nullptr, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }

    std::vector<int> recv_buffer(array_size);

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < inner_iters; iter++) {
            MPI_Win_fence(0, win);
            if (g_rank != 0) {
                MPI_Get(recv_buffer.data(), array_size, MPI_INT, 0, 0, array_size, MPI_INT, win);
            }
            MPI_Win_fence(0, win);
        }

        // Synchronisation finale
        if (g_rank == 0) {
            int ack;
            for (int dest = 1; dest < g_size; dest++) {
                MPI_Recv(&ack, 1, MPI_INT, dest, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            int ack = 1;
            MPI_Send(&ack, 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
        }

        double elapsed = MPI_Wtime() - start;
        double per_op = elapsed / inner_iters;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }

    MPI_Win_free(&win);
    SetBytesProcessed1D(state, array_size, inner_iters);
}

// ============================================================================
// Benchmark Boost MPI 1D - Broadcast avec std::vector<int>
// ============================================================================
static void BM_BoostMPI_1D(benchmark::State& state) {
    int array_size = state.range(0);
    int inner_iters = get_inner_iterations_1d(array_size);

    boost::mpi::communicator world;
    std::vector<int> buffer(array_size, g_rank == 0 ? 42 : 0);

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < inner_iters; iter++) {
            boost::mpi::broadcast(world, buffer, 0);
        }

        // Synchronisation finale
        if (g_rank == 0) {
            int ack;
            for (int dest = 1; dest < g_size; dest++) {
                world.recv(dest, 1, ack);
            }
        } else {
            int ack = 1;
            world.send(0, 1, ack);
        }

        double elapsed = MPI_Wtime() - start;
        double per_op = elapsed / inner_iters;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
    SetBytesProcessed1D(state, array_size, inner_iters);
}

// ============================================================================
// Configurations de benchmark
// Args: {outer_size, base_size}
// Formule: size[i] = base_size * (i+1)² avec ratio 25:1 entre max et min
// Taille totale = base_size * (1 + 4 + 9 + 16 + 25) = base_size * 55
// ============================================================================

// Configuration: 5 vecteurs avec ratio 25:1
// Tailles relatives: 1x, 4x, 9x, 16x, 25x (total = 55x base_size)
// base=50:      2,750 ints = 11 KB      (Small)
// base=500:     27,500 ints = 107 KB    (Medium)
// base=5000:    275,000 ints = 1.05 MB  (Large)
// base=50000:   2,750,000 ints = 10.5 MB (XLarge)
// base=500000:  27,500,000 ints = 105 MB (XXLarge)
// base=2000000: 110,000,000 ints = 420 MB (XXXLarge)

#define BENCHMARK_WITH_CONFIGS(name) \
    BENCHMARK(name)->Args({5, 50})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10); \
    BENCHMARK(name)->Args({5, 500})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10); \
    BENCHMARK(name)->Args({5, 5000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10); \
    BENCHMARK(name)->Args({5, 50000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10); \
    BENCHMARK(name)->Args({5, 500000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(5); \
    BENCHMARK(name)->Args({5, 2000000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(3);

// Boost benchmarks need fewer iterations for large sizes due to serialization overhead
#define BENCHMARK_BOOST_CONFIGS(name) \
    BENCHMARK(name)->Args({5, 50})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10); \
    BENCHMARK(name)->Args({5, 500})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10); \
    BENCHMARK(name)->Args({5, 5000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10); \
    BENCHMARK(name)->Args({5, 50000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(3); \
    BENCHMARK(name)->Args({5, 500000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(2); \
    BENCHMARK(name)->Args({5, 2000000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(1);

BENCHMARK_WITH_CONFIGS(BM_RawMPI)
BENCHMARK_WITH_CONFIGS(BM_BcastMPI)
BENCHMARK_WITH_CONFIGS(BM_PackMPI)
BENCHMARK_WITH_CONFIGS(BM_DatatypeMPI)
BENCHMARK_WITH_CONFIGS(BM_RDMAMPI)
BENCHMARK_BOOST_CONFIGS(BM_BoostMPI)
BENCHMARK_BOOST_CONFIGS(BM_BoostPackedMPI)

// ============================================================================
// Configuration 1D - Tailles équivalentes aux benchmarks 2D
// Args: {array_size}
// 2750 = 11 KB, 27500 = 107 KB, 275000 = 1 MB, 2750000 = 10 MB,
// 27500000 = 105 MB, 110000000 = 420 MB
// ============================================================================

#define BENCHMARK_1D_CONFIGS(name) \
    BENCHMARK(name)->Args({2750})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10); \
    BENCHMARK(name)->Args({27500})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10); \
    BENCHMARK(name)->Args({275000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10); \
    BENCHMARK(name)->Args({2750000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10); \
    BENCHMARK(name)->Args({27500000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(5); \
    BENCHMARK(name)->Args({110000000})->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(3);

BENCHMARK_1D_CONFIGS(BM_RawMPI_1D)
BENCHMARK_1D_CONFIGS(BM_BcastMPI_1D)
BENCHMARK_1D_CONFIGS(BM_RDMAMPI_1D)
BENCHMARK_1D_CONFIGS(BM_BoostMPI_1D)

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    benchmark::Initialize(&argc, argv);

    if (g_rank == 0) {
        benchmark::RunSpecifiedBenchmarks();
    } else {
        NullReporter null_reporter;
        benchmark::RunSpecifiedBenchmarks(&null_reporter);
    }

    benchmark::Shutdown();
    MPI_Finalize();
    return 0;
}
