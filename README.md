# MPI & Boost Performance Benchmark

## Objective

This repository provides an experimental C++ benchmark to measure and compare the performance of various data serialization and communication strategies using MPI and Boost. It specifically focuses on transferring complex, variably-sized data structures (`std::vector<std::vector<int>>`) between MPI processes.

## Prerequisites

- A C++11 compliant compiler (e.g., GCC, Clang)
- CMake (>= 3.10)
- An MPI implementation (e.g., Open MPI, MPICH)
- Boost libraries (>= 1.83) with `mpi` and `serialization` components.

## Build & Run

To compile and run the benchmark, follow these steps:

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd boost_mpi_benchmark

# 2. Create a build directory
mkdir build && cd build

# 3. Configure with CMake (with -O3 optimization)
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3" ..

# 4. Compile the project
make -j$(nproc)

# 5. Run the benchmark
mpirun -np 4 ./mpi_benchmark
```

## Technical Overview

The benchmark transfers a `std::vector<std::vector<int>>` where inner vectors have non-uniform sizes. It compares the following communication methods:

- **Raw MPI**: Manual point-to-point transfer using asynchronous `MPI_Isend`/`MPI_Irecv`. Metadata (sizes) are sent first.
- **Bcast MPI**: Collective communication using `MPI_Bcast` and asynchronous `MPI_Ibcast`.
- **Packed MPI**: Data is manually serialized into a contiguous buffer using `MPI_Pack` and transferred via `MPI_Ibcast`.
- **Datatype MPI**: Uses `MPI_Type_contiguous` to create and transfer derived MPI datatypes.
- **RDMA (One-Sided)**: Leverages MPI Remote Memory Access (`MPI_Win_create`, `MPI_Get`) for one-sided communication.
- **Boost MPI**: Relies on Boost.MPI's built-in serialization for direct object transfer.
- **Boost Packed MPI**: Uses Boost's `packed_oarchive` and `packed_iarchive` for manual serialization before transfer.

## Data Structure

The benchmark transfers a `VectorOfVectors` structure containing variably-sized inner vectors:

```
VectorOfVectors (std::vector<std::vector<int>>)
┌─────────────────────────────────────────────────────────────┐
│                   Outer Vector Container                    │
├─────────────────────────────────────────────────────────────┤
│ [0] ┌───────────────────────────────────────┐ size = N0     │
│     │ int │ int │ int │ int │ ... │ int │                   │
│     └───────────────────────────────────────┘               │
│                                                             │
│ [1] ┌─────────────────────┐ size = N1                       │
│     │ int │ int │ ... │                                     │
│     └─────────────────────┘                                 │
│                                                             │
│ [2] ┌───────────────────────────────┐ size = N2             │
│     │ int │ int │ int │ ... │ int │                         │
│     └───────────────────────────────┘                       │
│                                                             │
│ ... (up to DEFAULT_OUTER_SIZE vectors)                      │
└─────────────────────────────────────────────────────────────┘

Where: N0, N1, N2... are variable sizes (N_i = 1 + i^4)
```

## Performance Results

Results below were obtained on a 4-process run, compiled with GCC and **-O3** optimization. The metric is the average time per operation.

| Communication Method          | Performance (s/op) | Rank |
|-------------------------------|--------------------|:----:|
| **Datatype MPI**              | **1.20e-05**       | 1    |
| **RDMA MPI**                  | **1.44e-05**       | 2    |
| **Raw MPI**                   | **1.70e-05**       | 3    |
| **Bcast MPI**                 | **1.90e-05**       | 4    |
| **Pack MPI**                  | **2.14e-05**       | 5    |
| **Boost Packed MPI**          | **2.38e-05**       | 6    |
| **Boost MPI**                 | **4.09e-05**       | 7    |

## Conclusion

For transferring complex, non-contiguous data structures, creating and committing derived **MPI Datatypes** consistently delivers the best performance. **RDMA (one-sided communication)** is also a highly competitive alternative.

While convenient, standard **Boost.MPI serialization** introduces significant overhead, making it the slowest method in this benchmark. For performance-critical applications, manual memory layout management via MPI Datatypes or RDMA is highly recommended over automated serialization libraries.

**Note:** These results may vary significantly depending on the MPI implementation (Open MPI, MPICH, Intel MPI, etc.) and the hardware configuration used.

## Context

This benchmark was developed to support decision-making within the [Samurai](https://github.com/hpc-maths/samurai) project, in order to understand the current state of MPI communication performance. 