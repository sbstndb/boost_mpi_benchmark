#include <iostream>
#include <vector>
#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/mpi/packed_oarchive.hpp>
#include <boost/mpi/packed_iarchive.hpp>
#include <benchmark/benchmark.h>

#define DEFAULT_OUTER_SIZE 10
#define DEFAULT_INNER_SIZE 1
#define INNER_ITERATIONS 10000

// Variables globales MPI
static int g_rank = -1;
static int g_size = 0;

struct VectorOfVectors {
    std::vector<std::vector<int>> data;

    VectorOfVectors() {
        data.resize(DEFAULT_OUTER_SIZE);
        for (int i = 0; i < DEFAULT_OUTER_SIZE; i++) {
            data[i].resize(DEFAULT_INNER_SIZE + i * i * i * i, 0);
        }
    }

    VectorOfVectors(int) : data() {}

    template <class Archive>
    void serialize(Archive & ar, const unsigned int) {
        ar & data;
    }
};

// NullReporter pour les ranks > 0
class NullReporter : public benchmark::BenchmarkReporter {
public:
    bool ReportContext(const Context&) override { return true; }
    void ReportRuns(const std::vector<Run>&) override {}
    void Finalize() override {}
};

// ============================================================================
// Benchmark Raw MPI
// ============================================================================
static void BM_RawMPI(benchmark::State& state) {
    VectorOfVectors vec;
    int outer_size = vec.data.size();
    std::vector<int> inner_sizes(outer_size);
    for (int j = 0; j < outer_size; j++) {
        inner_sizes[j] = vec.data[j].size();
    }

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < INNER_ITERATIONS; iter++) {
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
                VectorOfVectors recv_vec(0);
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
        double per_op = elapsed / INNER_ITERATIONS;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
}
BENCHMARK(BM_RawMPI)->Name("Raw MPI")->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10);

// ============================================================================
// Benchmark Bcast MPI
// ============================================================================
static void BM_BcastMPI(benchmark::State& state) {
    VectorOfVectors vec;
    int outer_size = vec.data.size();
    std::vector<int> inner_sizes(outer_size);
    for (int j = 0; j < outer_size; j++) {
        inner_sizes[j] = vec.data[j].size();
    }

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < INNER_ITERATIONS; iter++) {
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
                VectorOfVectors recv_vec(0);
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
        double per_op = elapsed / INNER_ITERATIONS;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
}
BENCHMARK(BM_BcastMPI)->Name("Bcast MPI")->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10);

// ============================================================================
// Benchmark Pack MPI
// ============================================================================
static void BM_PackMPI(benchmark::State& state) {
    VectorOfVectors vec;
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

        for (int iter = 0; iter < INNER_ITERATIONS; iter++) {
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
                VectorOfVectors recv_vec(0);
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
        double per_op = elapsed / INNER_ITERATIONS;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
}
BENCHMARK(BM_PackMPI)->Name("Pack MPI")->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10);

// ============================================================================
// Benchmark Datatype MPI
// ============================================================================
static void BM_DatatypeMPI(benchmark::State& state) {
    VectorOfVectors vec;
    int outer_size = vec.data.size();
    std::vector<int> inner_sizes(outer_size);
    for (int j = 0; j < outer_size; j++) {
        inner_sizes[j] = vec.data[j].size();
    }

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < INNER_ITERATIONS; iter++) {
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
                VectorOfVectors recv_vec(0);
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
        double per_op = elapsed / INNER_ITERATIONS;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
}
BENCHMARK(BM_DatatypeMPI)->Name("Datatype MPI")->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10);

// ============================================================================
// Benchmark RDMA MPI
// ============================================================================
static void BM_RDMAMPI(benchmark::State& state) {
    VectorOfVectors vec;
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

        for (int iter = 0; iter < INNER_ITERATIONS; iter++) {
            if (g_rank == 0) {
                for (int dest = 1; dest < g_size; dest++) {
                    MPI_Send(&outer_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                    MPI_Send(inner_sizes.data(), outer_size, MPI_INT, dest, 1, MPI_COMM_WORLD);
                }
                MPI_Win_fence(0, win);
                MPI_Win_fence(0, win);
            } else {
                VectorOfVectors recv_vec(0);
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
        double per_op = elapsed / INNER_ITERATIONS;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }

    MPI_Win_free(&win);
}
BENCHMARK(BM_RDMAMPI)->Name("RDMA MPI")->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10);

// ============================================================================
// Benchmark Boost MPI
// ============================================================================
static void BM_BoostMPI(benchmark::State& state) {
    boost::mpi::communicator world;
    VectorOfVectors vec;

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < INNER_ITERATIONS; iter++) {
            if (g_rank == 0) {
                for (int dest = 1; dest < g_size; dest++) {
                    world.send(dest, 0, vec);
                }
            } else {
                VectorOfVectors recv_vec(0);
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
        double per_op = elapsed / INNER_ITERATIONS;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
}
BENCHMARK(BM_BoostMPI)->Name("Boost MPI")->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10);

// ============================================================================
// Benchmark Boost Packed MPI
// ============================================================================
static void BM_BoostPackedMPI(benchmark::State& state) {
    boost::mpi::communicator world;
    VectorOfVectors vec;

    for (auto _ : state) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int iter = 0; iter < INNER_ITERATIONS; iter++) {
            if (g_rank == 0) {
                boost::mpi::packed_oarchive::buffer_type buffer;
                boost::mpi::packed_oarchive oa(world, buffer);
                oa << vec;
                for (int dest = 1; dest < g_size; dest++) {
                    world.send(dest, 0, buffer);
                }
            } else {
                VectorOfVectors recv_vec(0);
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
        double per_op = elapsed / INNER_ITERATIONS;
        double max_per_op;
        MPI_Allreduce(&per_op, &max_per_op, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_per_op);
    }
}
BENCHMARK(BM_BoostPackedMPI)->Name("Boost Packed MPI")->UseManualTime()->Unit(benchmark::kMicrosecond)->Iterations(10);

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
