# MPI & Boost Performance Benchmark

## Objective

This repository provides an experimental C++ benchmark to measure and compare the performance of various data serialization and communication strategies using MPI and Boost. It includes two benchmark suites:

1. **2D Benchmarks**: Transfer of complex, variably-sized nested structures (`std::vector<std::vector<int>>`)
2. **1D Benchmarks**: Transfer of simple contiguous arrays (`std::vector<int>`) to measure pure communication cost

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

### 2D Benchmarks (Nested Vectors)

The 2D benchmark transfers a `std::vector<std::vector<int>>` where inner vectors have non-uniform sizes. It compares the following communication methods:

- **Raw MPI**: Manual point-to-point transfer using asynchronous `MPI_Isend`/`MPI_Irecv`. Metadata (sizes) are sent first.
- **Bcast MPI**: Collective communication using `MPI_Bcast` and asynchronous `MPI_Ibcast`.
- **Packed MPI**: Data is manually serialized into a contiguous buffer using `MPI_Pack` and transferred via `MPI_Ibcast`.
- **Datatype MPI**: Uses `MPI_Type_contiguous` to create and transfer derived MPI datatypes.
- **RDMA (One-Sided)**: Leverages MPI Remote Memory Access (`MPI_Win_create`, `MPI_Get`) for one-sided communication.
- **Boost MPI**: Relies on Boost.MPI's built-in serialization for direct object transfer.
- **Boost Packed MPI**: Uses Boost's `packed_oarchive` and `packed_iarchive` for manual serialization before transfer.

### 1D Benchmarks (Contiguous Buffer)

The 1D benchmark transfers a simple `std::vector<int>` to measure pure communication cost without serialization overhead:

- **Raw MPI 1D**: Point-to-point transfer using `MPI_Isend`/`MPI_Recv` on a contiguous buffer.
- **Bcast MPI 1D**: Collective broadcast using `MPI_Bcast` on a contiguous buffer.
- **RDMA MPI 1D**: One-sided communication using `MPI_Win_create` and `MPI_Get`.
- **Boost MPI 1D**: Uses `boost::mpi::broadcast` with native `std::vector<int>` support.

## Data Structures

### 2D Structure (VectorOfVectors)

The 2D benchmark transfers a `VectorOfVectors` structure containing variably-sized inner vectors. The structure uses a **quadratic formula** to ensure significant size variation (25:1 ratio) at all scales:

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

#### 2D Configurations

| Config   | base_size | [0] | [1] | [2] | [3] | [4] | Total |
|----------|-----------|-----|-----|-----|-----|-----|-------|
| Small    | 50        | 50  | 200 | 450 | 800 | 1.25K | **11 KB** |
| Medium   | 500       | 500 | 2K  | 4.5K | 8K | 12.5K | **107 KB** |
| Large    | 5,000     | 5K  | 20K | 45K | 80K | 125K | **1.05 MB** |
| XLarge   | 50,000    | 50K | 200K | 450K | 800K | 1.25M | **10.5 MB** |
| XXLarge  | 500,000   | 500K | 2M | 4.5M | 8M | 12.5M | **105 MB** |
| XXXLarge | 2,000,000 | 2M | 8M | 18M | 32M | 50M | **420 MB** |

### 1D Structure (Contiguous Buffer)

The 1D benchmark uses a simple `std::vector<int>` with the **same total number of elements** as the 2D benchmarks to enable fair comparison:

```
Vector1D (std::vector<int>)
┌─────────────────────────────────────────────────────────────┐
│              Contiguous Integer Buffer                      │
├─────────────────────────────────────────────────────────────┤
│ [0] [1] [2] ... [array_size - 1]                            │
│                                                             │
│ array_size = base_size × 55 (matching 2D total)             │
└─────────────────────────────────────────────────────────────┘
```

#### 1D Configurations

| Config   | array_size | Total |
|----------|------------|-------|
| Small    | 2,750      | **11 KB** |
| Medium   | 27,500     | **107 KB** |
| Large    | 275,000    | **1.05 MB** |
| XLarge   | 2,750,000  | **10.5 MB** |
| XXLarge  | 27,500,000 | **105 MB** |
| XXXLarge | 110,000,000| **420 MB** |

## Benchmarking Methodology

The benchmark uses [Google Benchmark](https://github.com/google/benchmark) integrated with MPI:

- **Inner loop pattern**: Each Google Benchmark iteration runs multiple inner iterations (scaled by data size: 10,000 for small, down to 10 for XXLarge) to amortize synchronization overhead.
- **Manual timing**: Uses `MPI_Wtime()` for precise timing with `UseManualTime()` to report per-operation time.
- **Multi-process synchronization**: A `NullReporter` suppresses output on non-root MPI processes while ensuring all processes execute the same number of iterations.
- **Timing accuracy**: The maximum time across all processes is reported via `MPI_Allreduce` to capture the true communication cost.

## Performance Results

Results below were obtained using [Google Benchmark](https://github.com/google/benchmark) on a 4-process run, compiled with GCC and **-O3** optimization. The metric is the average time per operation.

### 2D Results (Nested Vectors)

#### Small Data (~11 KB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Pack MPI**        | 7.5       | 1    |
| **Raw MPI**         | 7.6       | 2    |
| **RDMA MPI**        | 8.4       | 3    |
| **Bcast MPI**       | 8.5       | 4    |
| **Datatype MPI**    | 8.7       | 5    |
| **Boost Packed MPI**| 12.3      | 6    |
| **Boost MPI**       | 20.8      | 7    |

#### Medium Data (~107 KB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Bcast MPI**       | 24.9      | 1    |
| **Datatype MPI**    | 25.3      | 2    |
| **Raw MPI**         | 27.2      | 3    |
| **RDMA MPI**        | 42.2      | 4    |
| **Pack MPI**        | 45.4      | 5    |
| **Boost Packed MPI**| 61.1      | 6    |
| **Boost MPI**       | 103       | 7    |

#### Large Data (~1 MB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Datatype MPI**    | 188       | 1    |
| **Bcast MPI**       | 193       | 2    |
| **Pack MPI**        | 282       | 3    |
| **RDMA MPI**        | 306       | 4    |
| **Raw MPI**         | 510       | 5    |
| **Boost Packed MPI**| 542       | 6    |
| **Boost MPI**       | 821       | 7    |

#### XLarge Data (~10.5 MB)

| Communication Method | Time (ms) | Rank |
|---------------------|----------:|:----:|
| **Bcast MPI**       | 4.0       | 1    |
| **Datatype MPI**    | 5.9       | 2    |
| **Pack MPI**        | 6.2       | 3    |
| **RDMA MPI**        | 6.3       | 4    |
| **Raw MPI**         | 10.5      | 5    |
| **Boost Packed MPI**| 18.4      | 6    |
| **Boost MPI**       | 25.4      | 7    |

#### XXLarge Data (~105 MB)

| Communication Method | Time (ms) | Rank |
|---------------------|----------:|:----:|
| **Raw MPI**         | 85        | 1    |
| **Bcast MPI**       | 86        | 2    |
| **Datatype MPI**    | 99        | 3    |
| **Pack MPI**        | 179       | 4    |
| **RDMA MPI**        | 268       | 5    |
| **Boost Packed MPI**| 452       | 6    |
| **Boost MPI**       | 705       | 7    |

#### XXXLarge Data (~420 MB)

| Communication Method | Time (ms) | Rank |
|---------------------|----------:|:----:|
| **Bcast MPI**       | 496       | 1    |
| **Raw MPI**         | 501       | 2    |
| **Datatype MPI**    | 522       | 3    |
| **Pack MPI**        | 829       | 4    |
| **Boost Packed MPI**| 2162      | 5    |
| **Boost MPI**       | 3268      | 6    |
| **RDMA MPI**        | 3428      | 7    |

### 1D Results (Contiguous Buffer)

These benchmarks measure pure communication cost with a simple `std::vector<int>`.

#### Small Data (~11 KB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Bcast MPI 1D**    | 4.2       | 1    |
| **Raw MPI 1D**      | 4.2       | 2    |
| **Boost MPI 1D**    | 7.2       | 3    |
| **RDMA MPI 1D**     | 7.5       | 4    |

#### Medium Data (~107 KB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Bcast MPI 1D**    | 26.7      | 1    |
| **Raw MPI 1D**      | 26.9      | 2    |
| **RDMA MPI 1D**     | 33.9      | 3    |
| **Boost MPI 1D**    | 46.4      | 4    |

#### Large Data (~1 MB)

| Communication Method | Time (μs) | Rank |
|---------------------|----------:|:----:|
| **Bcast MPI 1D**    | 212       | 1    |
| **RDMA MPI 1D**     | 219       | 2    |
| **Raw MPI 1D**      | 256       | 3    |
| **Boost MPI 1D**    | 329       | 4    |

#### XLarge Data (~10.5 MB)

| Communication Method | Time (ms) | Rank |
|---------------------|----------:|:----:|
| **Raw MPI 1D**      | 3.7       | 1    |
| **RDMA MPI 1D**     | 4.5       | 2    |
| **Bcast MPI 1D**    | 5.1       | 3    |
| **Boost MPI 1D**    | 8.5       | 4    |

#### XXLarge Data (~105 MB)

| Communication Method | Time (ms) | Rank |
|---------------------|----------:|:----:|
| **Raw MPI 1D**      | 52.8      | 1    |
| **Bcast MPI 1D**    | 70.8      | 2    |
| **Boost MPI 1D**    | 180       | 3    |
| **RDMA MPI 1D**     | 211       | 4    |

#### XXXLarge Data (~420 MB)

| Communication Method | Time (ms) | Rank |
|---------------------|----------:|:----:|
| **Raw MPI 1D**      | 205       | 1    |
| **Bcast MPI 1D**    | 359       | 2    |
| **Boost MPI 1D**    | 891       | 3    |
| **RDMA MPI 1D**     | 3050      | 4    |

## Conclusion

### 2D Benchmarks (Nested Vectors)

The optimal communication method **depends on data size**:

- **Small data (< 100 KB)**: **Pack MPI** and **Raw MPI** deliver the best performance.
- **Medium/Large data (100 KB - 10 MB)**: **Bcast MPI** and **Datatype MPI** become the clear winners.
- **Very large data (> 100 MB)**: **Raw MPI** and **Bcast MPI** are the fastest methods.

### 1D Benchmarks (Contiguous Buffer)

- **Small to Medium data (< 1 MB)**: **Bcast MPI** and **Raw MPI** are nearly equivalent.
- **Large data (> 10 MB)**: **Raw MPI** is consistently the fastest.
- **Boost MPI 1D** is ~2x faster than Boost MPI 2D for equivalent data sizes (no serialization overhead).

### 1D vs 2D Comparison

| Data Size | Best 2D Method | Best 1D Method | 2D/1D Ratio |
|-----------|---------------|----------------|-------------|
| 11 KB     | Pack (7.5 μs) | Bcast (4.2 μs) | 1.8x |
| 107 KB    | Bcast (24.9 μs) | Bcast (26.7 μs) | 0.9x |
| 1 MB      | Datatype (188 μs) | Bcast (212 μs) | 0.9x |
| 10 MB     | Bcast (4.0 ms) | Raw (3.7 ms) | 1.1x |
| 105 MB    | Raw (85 ms) | Raw (52.8 ms) | 1.6x |
| 420 MB    | Bcast (496 ms) | Raw (205 ms) | 2.4x |

### Key Findings

**2D (Nested Vectors):**
- Native MPI methods outperform Boost.MPI by 3-8x depending on data size.
- **Boost.MPI** introduces significant serialization overhead (21 μs for 11 KB → 3.3 sec for 420 MB).
- **RDMA** becomes extremely slow for very large data (3.4 sec for 420 MB) due to fence overhead.

**1D (Contiguous Buffer):**
- **Raw MPI** and **Bcast MPI** achieve near-optimal performance for contiguous data.
- **Boost MPI 1D** has minimal overhead compared to native MPI (no element-by-element serialization).
- **RDMA** suffers from fence synchronization overhead even for contiguous data (3 sec for 420 MB).

**Recommendations:**
- For **nested/complex structures**: use **Bcast MPI** or **Datatype MPI**.
- For **contiguous data**: use **Raw MPI** for large sizes, **Bcast MPI** for small sizes.
- **Avoid RDMA** for very large data transfers (> 100 MB).

**Note:** These results may vary significantly depending on the MPI implementation (Open MPI, MPICH, Intel MPI, etc.) and the hardware configuration used.

## Context

This benchmark was developed to support decision-making within the [Samurai](https://github.com/hpc-maths/samurai) project, in order to understand the current state of MPI communication performance. 