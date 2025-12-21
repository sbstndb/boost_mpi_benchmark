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

The benchmark transfers a `VectorOfVectors` structure containing variably-sized inner vectors. The structure uses a **quadratic formula** to ensure significant size variation (25:1 ratio) at all scales:

```
VectorOfVectors (std::vector<std::vector<int>>)
┌─────────────────────────────────────────────────────────────┐
│                   Outer Vector Container                    │
├─────────────────────────────────────────────────────────────┤
│ [0] size = base_size × 1² = base_size                       │
│ [1] size = base_size × 2² = 4 × base_size                   │
│ [2] size = base_size × 3² = 9 × base_size                   │
│ [3] size = base_size × 4² = 16 × base_size                  │
│ [4] size = base_size × 5² = 25 × base_size                  │
│                                                             │
│ Formula: size[i] = base_size × (i+1)²                       │
│ Total = base_size × 55                                      │
└─────────────────────────────────────────────────────────────┘
```

### Benchmark Configurations

| Config   | base_size | [0] | [1] | [2] | [3] | [4] | Total |
|----------|-----------|-----|-----|-----|-----|-----|-------|
| Small    | 50        | 50  | 200 | 450 | 800 | 1.25K | **11 KB** |
| Medium   | 500       | 500 | 2K  | 4.5K | 8K | 12.5K | **107 KB** |
| Large    | 5,000     | 5K  | 20K | 45K | 80K | 125K | **1.05 MB** |
| XLarge   | 50,000    | 50K | 200K | 450K | 800K | 1.25M | **10.5 MB** |
| XXLarge  | 500,000   | 500K | 2M | 4.5M | 8M | 12.5M | **105 MB** |
| XXXLarge | 2,000,000 | 2M | 8M | 18M | 32M | 50M | **420 MB** |

## Benchmarking Methodology

The benchmark uses [Google Benchmark](https://github.com/google/benchmark) integrated with MPI:

- **Inner loop pattern**: Each Google Benchmark iteration runs multiple inner iterations (scaled by data size: 10,000 for small, down to 10 for XXLarge) to amortize synchronization overhead.
- **Manual timing**: Uses `MPI_Wtime()` for precise timing with `UseManualTime()` to report per-operation time.
- **Multi-process synchronization**: A `NullReporter` suppresses output on non-root MPI processes while ensuring all processes execute the same number of iterations.
- **Timing accuracy**: The maximum time across all processes is reported via `MPI_Allreduce` to capture the true communication cost.

## Performance Results

Results below were obtained using [Google Benchmark](https://github.com/google/benchmark) on a 4-process run, compiled with GCC and **-O3** optimization. The metric is the average time per operation.

### Small Data (~11 KB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Pack MPI**        | 7.1       | 1    |
| **Raw MPI**         | 7.6       | 2    |
| **RDMA MPI**        | 8.2       | 3    |
| **Bcast MPI**       | 8.6       | 4    |
| **Datatype MPI**    | 9.3       | 5    |
| **Boost Packed MPI**| 12.8      | 6    |
| **Boost MPI**       | 21.5      | 7    |

### Medium Data (~107 KB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Bcast MPI**       | 23.9      | 1    |
| **Datatype MPI**    | 24.9      | 2    |
| **Raw MPI**         | 25.9      | 3    |
| **RDMA MPI**        | 42.1      | 4    |
| **Pack MPI**        | 45.3      | 5    |
| **Boost Packed MPI**| 61.5      | 6    |
| **Boost MPI**       | 103       | 7    |

### Large Data (~1 MB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Bcast MPI**       | 187       | 1    |
| **Datatype MPI**    | 192       | 2    |
| **Pack MPI**        | 280       | 3    |
| **RDMA MPI**        | 304       | 4    |
| **Raw MPI**         | 512       | 5    |
| **Boost Packed MPI**| 534       | 6    |
| **Boost MPI**       | 810       | 7    |

### XLarge Data (~10.5 MB)

| Communication Method | Time (ms) | Rank |
|---------------------|----------:|:----:|
| **Bcast MPI**       | 4.2       | 1    |
| **Datatype MPI**    | 5.9       | 2    |
| **RDMA MPI**        | 6.4       | 3    |
| **Pack MPI**        | 6.5       | 4    |
| **Raw MPI**         | 10.3      | 5    |
| **Boost Packed MPI**| 18.4      | 6    |
| **Boost MPI**       | 25.1      | 7    |

### XXLarge Data (~105 MB)

| Communication Method | Time (ms) | Rank |
|---------------------|----------:|:----:|
| **Raw MPI**         | 88        | 1    |
| **Bcast MPI**       | 90        | 2    |
| **Datatype MPI**    | 98        | 3    |
| **Pack MPI**        | 182       | 4    |
| **RDMA MPI**        | 269       | 5    |
| **Boost Packed MPI**| 444       | 6    |
| **Boost MPI**       | 715       | 7    |

### XXXLarge Data (~420 MB)

| Communication Method | Time (ms) | Rank |
|---------------------|----------:|:----:|
| **Raw MPI**         | 455       | 1    |
| **Datatype MPI**    | 505       | 2    |
| **Bcast MPI**       | 524       | 3    |
| **Pack MPI**        | 834       | 4    |
| **Boost Packed MPI**| 2152      | 5    |
| **Boost MPI**       | 3310      | 6    |
| **RDMA MPI**        | 3452      | 7    |

## Conclusion

The optimal communication method **depends on data size**:

- **Small data (< 100 KB)**: **Pack MPI** and **Raw MPI** deliver the best performance.

- **Medium/Large data (100 KB - 10 MB)**: **Bcast MPI** and **Datatype MPI** become the clear winners, with collective operations outperforming point-to-point by up to 2.5x.

- **Very large data (> 100 MB)**: **Raw MPI** returns as the fastest method, closely followed by Bcast and Datatype MPI.

**Key findings:**
- Native MPI methods consistently outperform Boost.MPI by 3-8x depending on data size.
- **Boost.MPI** introduces significant serialization overhead that grows dramatically (21 μs for 11 KB → 3.3 sec for 420 MB).
- **Boost Packed MPI** is ~1.5x faster than Boost.MPI for large data by serializing once and reusing the buffer.
- **RDMA** becomes extremely slow for very large data (3.5 sec for 420 MB) - likely due to memory copy overhead.
- For **medium data (100 KB - 10 MB)**, prefer **Bcast MPI** or **Datatype MPI**.
- For **very large data (> 100 MB)**, prefer **Raw MPI**.
- For **small, frequent transfers**, **Pack MPI** or **Raw MPI** are recommended.

**Note:** These results may vary significantly depending on the MPI implementation (Open MPI, MPICH, Intel MPI, etc.) and the hardware configuration used.

## Context

This benchmark was developed to support decision-making within the [Samurai](https://github.com/hpc-maths/samurai) project, in order to understand the current state of MPI communication performance. 