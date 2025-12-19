# MPI & Boost Performance Benchmark

## Objective

This repository provides an experimental C++ benchmark to measure and compare the performance of various data serialization and communication strategies using MPI and Boost. It specifically focuses on transferring complex, variably-sized data structures (`std::vector<std::vector<int>>`) between MPI processes.

## Prerequisites

- A C++17 compliant compiler (e.g., GCC, Clang)
- CMake (>= 3.14)
- An MPI implementation (e.g., Open MPI, MPICH)
- Boost libraries (>= 1.83) with `mpi` and `serialization` components

**Note:** [Google Benchmark](https://github.com/google/benchmark) is automatically fetched and built via CMake's FetchContent.

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

The benchmark transfers a `VectorOfVectors` structure containing variably-sized inner vectors. The structure is parameterized by `outer_size` (number of vectors) and `base_size` (minimum elements per vector):

```
VectorOfVectors (std::vector<std::vector<int>>)
┌─────────────────────────────────────────────────────────────┐
│                   Outer Vector Container                    │
├─────────────────────────────────────────────────────────────┤
│ [0] size = base_size + 0                                    │
│ [1] size = base_size + 1                                    │
│ [2] size = base_size + 16                                   │
│ [3] size = base_size + 81                                   │
│ [4] size = base_size + 256                                  │
│ ...                                                         │
│ [i] size = base_size + i^4                                  │
└─────────────────────────────────────────────────────────────┘
```

### Benchmark Configurations

| Config | outer_size | base_size | Total ints | Data Size |
|--------|------------|-----------|------------|-----------|
| Small  | 5          | 1         | 359        | ~1.4 KB   |
| Medium | 5          | 1,000     | 5,354      | ~21 KB    |
| Large  | 5          | 10,000    | 50,354     | ~197 KB   |
| XLarge | 5          | 100,000   | 500,354    | ~1.9 MB   |

## Benchmarking Methodology

The benchmark uses [Google Benchmark](https://github.com/google/benchmark) integrated with MPI:

- **Inner loop pattern**: Each Google Benchmark iteration runs 10,000 inner iterations of the MPI communication pattern to amortize synchronization overhead.
- **Manual timing**: Uses `MPI_Wtime()` for precise timing with `UseManualTime()` to report per-operation time.
- **Multi-process synchronization**: A `NullReporter` suppresses output on non-root MPI processes while ensuring all processes execute the same number of iterations.
- **Timing accuracy**: The maximum time across all processes is reported via `MPI_Allreduce` to capture the true communication cost.

## Performance Results

Results below were obtained using [Google Benchmark](https://github.com/google/benchmark) on a 4-process run, compiled with GCC and **-O3** optimization. The metric is the average time per operation in microseconds (μs).

### Small Data (~1.4 KB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Pack MPI**        | 1.90      | 1    |
| **Raw MPI**         | 2.09      | 2    |
| **RDMA MPI**        | 2.79      | 3    |
| **Bcast MPI**       | 3.00      | 4    |
| **Boost Packed MPI**| 3.15      | 5    |
| **Datatype MPI**    | 3.17      | 6    |
| **Boost MPI**       | 5.37      | 7    |

### Medium Data (~21 KB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Raw MPI**         | 6.74      | 1    |
| **Pack MPI**        | 7.62      | 2    |
| **RDMA MPI**        | 8.34      | 3    |
| **Bcast MPI**       | 8.64      | 4    |
| **Datatype MPI**    | 10.2      | 5    |
| **Boost Packed MPI**| 12.8      | 6    |
| **Boost MPI**       | 21.3      | 7    |

### Large Data (~197 KB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Raw MPI**         | 22.2      | 1    |
| **Bcast MPI**       | 28.5      | 2    |
| **Datatype MPI**    | 31.2      | 3    |
| **Pack MPI**        | 42.0      | 4    |
| **RDMA MPI**        | 44.8      | 5    |
| **Boost Packed MPI**| 67.1      | 6    |
| **Boost MPI**       | 106       | 7    |

### XLarge Data (~1.9 MB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Datatype MPI**    | 278       | 1    |
| **RDMA MPI**        | 472       | 2    |
| **Raw MPI**         | 672       | 3    |
| **Bcast MPI**       | 759       | 4    |
| **Pack MPI**        | 1635      | 5    |
| **Boost Packed MPI**| 1905      | 6    |
| **Boost MPI**       | 4175      | 7    |

## Conclusion

The optimal communication method **depends on data size**:

- **Small/Medium data (< 200 KB)**: **Raw MPI** and **Pack MPI** deliver the best performance. The overhead of MPI derived datatypes is not worth it for small transfers.

- **Large data (> 1 MB)**: **Datatype MPI** becomes the clear winner, outperforming Raw MPI by ~2.4x. The upfront cost of creating derived datatypes is amortized over the larger transfer.

**Key findings:**
- Native MPI methods (Raw, Bcast, Pack, Datatype, RDMA) consistently outperform Boost.MPI by 2-15x depending on data size.
- **Boost.MPI** introduces significant serialization overhead that grows dramatically with data size (5.37 μs for 1.4 KB vs 4175 μs for 1.9 MB).
- **Boost Packed MPI** offers a middle ground, being ~2x faster than Boost.MPI for large data by serializing once and reusing the buffer.
- For **performance-critical applications** with large data structures, prefer **Datatype MPI** or **RDMA**.
- For **small, frequent transfers**, **Raw MPI** or **Pack MPI** are recommended.

**Note:** These results may vary significantly depending on the MPI implementation (Open MPI, MPICH, Intel MPI, etc.) and the hardware configuration used.

## Context

This benchmark was developed to support decision-making within the [Samurai](https://github.com/hpc-maths/samurai) project, in order to understand the current state of MPI communication performance. 