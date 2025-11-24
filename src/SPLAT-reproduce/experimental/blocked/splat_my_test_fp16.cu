#include "blocked.cuh"
#include <cmath>
#include "../../utils/utils.h"
#include "cuda_runtime.h"
#include <cstring>
#include <cassert>
#include <iostream>
#include <cuda_fp16.h>  // <-- added
#include <vector>
#include <iomanip>
#include <chrono>

#define BLOCK_SIZE_X 16 // This is b_c. -> this is also our coarsening factor.
#define BLOCK_SIZE_Y 16 // This is b_r.
#define INNER_DIM 8 // This is the inner dim we use in all the mat-muls. 
// Ensure that INNER_DIM is always smaller than Min(BLOCK_SIZE_X, BLOCK_SIZE_Y).

#define CHECK_CUDA(func)                                            \
{                                                                   \
  cudaError_t status = (func);                                      \
  if (status != cudaSuccess) {                                      \
    printf("CUDA API failed at line %d file: %s, with error: %s (%d)\n",      \
           __LINE__,__FILE__, cudaGetErrorString(status), status);           \
    exit(EXIT_FAILURE);                                             \
  }                                                                 \
}

struct coord {
    int start_col; 
    int end_col;

    coord(int start_col, int end_col) : start_col(start_col), 
                                        end_col(end_col) { }
};

__device__ bool windowed_is_computed(int row, int col, int sparsity_parameter) {
    if (row - sparsity_parameter < col && col < row + sparsity_parameter) {
        return true;
    }

    return false;
}

__device__ bool blocked_is_computed(int row, int col, int sparsity_parameter) {
    // Figure out what block we are in.
    int block_num = row / sparsity_parameter;

    if (block_num < 1) {
        return col < sparsity_parameter;
    }

    // Otherwise we have some check to make.
    if (((block_num - 1) * sparsity_parameter) < col && col < (block_num * sparsity_parameter)) {
        return true;
    }

    return false;
}

// 使用不同的函数名避免冲突
auto my_time_now() {
    return std::chrono::high_resolution_clock::now();
}

auto my_time_elapsed_us(std::chrono::high_resolution_clock::time_point start) {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}


// 测试用例结构体
struct TestCase {
    int batch_size;
    int seq_len;
    std::string test_name;
};

// Queries -> row-major, Keys^T -> row-major, Values -> row-major. [batch, num_heads, seq_length, hidden_dim].
// Kernel converted to use __half inputs/outputs; internal arithmetic done in float for stability.
__global__ void blocked_kernel(__half* queries, __half* keys, __half* values, __half* answer, __half * l, __half * m, int sparsity_param, int batch, 
                            int num_heads, int seq_length, int hidden_dim, coord* col_metadata_dev) {

    int head_hidden_dim = hidden_dim / num_heads;

    // For much better memory coalescing, we need to index as follows:
    #define idx_queries(b,s,n,h) (((b)*num_heads+(n))*seq_length+(s))*head_hidden_dim+(h)
    #define idx_keys(b,s,n,h) (((b)*num_heads+(n))*head_hidden_dim+(h))*seq_length+(s)
    #define idx_values(b,s,n,h) (((b)*num_heads+(n))*seq_length+(s))*head_hidden_dim+(h)
    #define idx_output(b,s,n,h) (((b)*num_heads+(n))*seq_length+(s))*head_hidden_dim+(h)

    int batch_num = blockIdx.z / batch;
    int head_num = blockIdx.z % batch;

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int row = by * blockDim.y + ty; int col = bx * blockDim.x + tx;

    // Shared memory: inputs in half, accumulators in float
    __shared__ __half queries_shmem[BLOCK_SIZE_Y][INNER_DIM];
    __shared__ __half keys_shmem[INNER_DIM][BLOCK_SIZE_X];
    __shared__ __half v_j[BLOCK_SIZE_X][BLOCK_SIZE_Y];

    __shared__ float answer_shmem_f[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float o_i_f[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float m_i_f[BLOCK_SIZE_Y]; 
    __shared__ float m_tilde_ij_f[BLOCK_SIZE_Y]; 
    __shared__ float m_new_i_f[BLOCK_SIZE_Y];
    __shared__ float l_i_f[BLOCK_SIZE_Y];
    __shared__ float l_tilde_ij_f[BLOCK_SIZE_Y];
    __shared__ float l_new_i_f[BLOCK_SIZE_Y]; 

    // Initialization of: l_i, m_i. TODO: initialize O_i.
    if (tx == 0 && ty+blockDim.y*blockIdx.y < seq_length) {
        l_i_f[ty] = __half2float(l[ty+blockDim.y*blockIdx.y]);
        m_i_f[ty] = __half2float(m[ty+blockDim.y*blockIdx.y]);
        l_tilde_ij_f[ty] = 0.0f;
        l_new_i_f[ty] = 0.0f;
    } else if (tx == 0) {
        l_i_f[ty] = 0.0f;
        m_i_f[ty] = -1e9f;
        l_tilde_ij_f[ty] = 0.0f;
        l_new_i_f[ty] = 0.0f;
    }

    // Load O_i (answer) into float shared mem.
    if (row < seq_length && col < head_hidden_dim) {
        o_i_f[ty][tx] = __half2float(answer[idx_output(batch_num, row, head_num, col)]);
    } else {
        o_i_f[ty][tx] = 0.0f;
    }

    // Outer loop over blocks of J
    for (int j = 0; j < ceilf(float(seq_length) / float(BLOCK_SIZE_X)); j++) {

        // reset per-j accumulators
        if (tx < 1) {
            m_tilde_ij_f[ty] = -1e9f;
            l_tilde_ij_f[ty] = 0.0f;
        }
        // reset answer_shmem accumulator
        answer_shmem_f[ty][tx] = 0.0f;
        __syncthreads();

        // Load V_j (transposed placement)
        if (tx+blockDim.x*bx < seq_length && j*BLOCK_SIZE_X+ty < hidden_dim) {
            v_j[tx][ty] = values[idx_values(batch_num, tx+blockDim.x*bx, head_num, j*BLOCK_SIZE_X+ty)];
        } else {
            v_j[tx][ty] = __float2half(0.0f);
        }

        // Step 1: Compute S_{i,j} -> tiled Q_i @ K_j
        for (int a = 0; a < ceilf(float(hidden_dim) / float(INNER_DIM)); a++) {
            __syncthreads();
            // load queries tile (half)
            if (row < seq_length && INNER_DIM*a+tx < hidden_dim) {
                queries_shmem[ty][tx] = queries[idx_queries(batch_num, row, head_num, INNER_DIM*a+tx)];
            } else if (tx < INNER_DIM) {
                queries_shmem[ty][tx] = __float2half(0.0f);
            }

            // load keys tile (half)
            if (tx+blockDim.x*bx < seq_length && INNER_DIM*a+ty < hidden_dim) {
                keys_shmem[tx][ty] = keys[idx_keys(batch_num, tx+blockDim.x*bx, head_num, INNER_DIM*a+ty)];
            } else if (ty < INNER_DIM) {
                keys_shmem[tx][ty] = __float2half(0.0f);
            }

            __syncthreads();

            // multiply-accumulate in float
            for (int k = 0; k < INNER_DIM; k++) {
                float qf = __half2float(queries_shmem[ty][k]);
                float kk = __half2float(keys_shmem[k][tx]);
                answer_shmem_f[ty][tx] += qf * kk;
            }
            __syncthreads();
        }

        // compute m_tilde_ij = max_k S_{i,j}[k]
        if (tx == 0) {
            float local_max = m_tilde_ij_f[ty];
            for (int k = 0; k < BLOCK_SIZE_X; k++) {
                local_max = fmaxf(local_max, answer_shmem_f[ty][k]);
            }
            m_tilde_ij_f[ty] = local_max;
        }
        __syncthreads();

        // compute ~P_{i,j} = exp(S - m_tilde)
        float my_m_tilde = m_tilde_ij_f[ty];
        float my_ans = answer_shmem_f[ty][tx];
        answer_shmem_f[ty][tx] = expf(my_ans - my_m_tilde);
        __syncthreads();

        // compute l_tilde_ij = sum_k ~P_{i,j}[k]
        if (tx == 0) {
            float acc = 0.0f;
            for (int k = 0; k < BLOCK_SIZE_X; k++) {
                acc += answer_shmem_f[ty][k];
            }
            l_tilde_ij_f[ty] = acc;
        }
        __syncthreads();

        // m_new_i
        if (tx == 0) {
            m_new_i_f[ty] = fmaxf(m_i_f[ty], m_tilde_ij_f[ty]);
        }
        __syncthreads();

        // l_new_i
        if (tx == 0) {
            l_new_i_f[ty] = expf(m_i_f[ty]-m_new_i_f[ty])*l_i_f[ty] + expf(m_tilde_ij_f[ty]-m_new_i_f[ty])*l_tilde_ij_f[ty];
        }
        __syncthreads();

        // Compute O_i update and write back (convert to half when writing)
        if (row < seq_length && col < hidden_dim) {
            float temp_answer = 0.0f;
            for (int a = 0; a < BLOCK_SIZE_X; a++) {
                temp_answer += answer_shmem_f[ty][a] * __half2float(v_j[a][tx]);
            }

            temp_answer *= expf(m_tilde_ij_f[ty] - m_new_i_f[ty]);
            temp_answer += l_i_f[ty]*expf(m_i_f[ty] - m_new_i_f[ty])*o_i_f[ty][tx];

            if (l_new_i_f[ty] != 0.0f) {
                temp_answer /= l_new_i_f[ty];
            }

            answer[idx_output(batch_num, row, head_num, col)] = __float2half(temp_answer);
        }
        __syncthreads();

        // update l_i and m_i (only tx==0 does this for each ty lane)
        if (tx == 0) {
            l_i_f[ty] = l_new_i_f[ty];
            m_i_f[ty] = m_new_i_f[ty];
        }
        __syncthreads();
    }
}


template<class T>
void blocked_launcher(T* queries_dev, T* keys_dev, T* values_dev, T* answer_dev, T * l_dev, T* m_dev, 
                        int batch, int num_heads, int seq_length, int hidden_dim, int sparsity_param, coord* col_metadata_dev) {

    // ...existing code...
    dim3 GridSize(1, ceil(float(seq_length) / float(BLOCK_SIZE_Y)), batch*num_heads); // Let's see if this is correct.
    dim3 BlockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

    // instantiate the fp16 kernel
    blocked_kernel<<<GridSize,BlockSize>>>((__half*)queries_dev, (__half*)keys_dev, (__half*)values_dev, 
                                                    (__half*)answer_dev, (__half*)l_dev, (__half*)m_dev, sparsity_param, batch, num_heads, 
                                                    seq_length, hidden_dim, col_metadata_dev); 
}

coord* populate_col_metadata_blocked(int s, int p, int block_height) {
    int num_blocks = ceil(float(s)/float(block_height));

    coord* metadata = (coord*)malloc(sizeof(coord)*num_blocks);

    for (int i = 0; i < num_blocks; i++) {
        // Find the start and end column within this range. 
        int top_row = block_height*i;
        int bottom_row = min(s-1, top_row+block_height);

        // Figure out what block top_row is in.
        int start_col = -1;
        int end_col = -1;
        int top_block_num = top_row / p;
        if (top_block_num < 1) {
            start_col = 0;
        } else {
            start_col = min((top_block_num-1) * p, s);
        }

        int bottom_block_num = bottom_row / p;
        if (bottom_block_num < 1) {
            end_col = p;
        } else {
            end_col = min(bottom_block_num*p + p, s);
        }
        metadata[i] = coord(start_col, end_col);
    }

    return metadata;
} 



// 格式化输出函数
void print_result(const std::string& test_name, int batch_size, int seq_len, float time_ms, const std::string& implementation = "SPLAT") {
    std::cout << "Attn | bs:" << std::setw(2) << batch_size 
              << " | seq:" << std::setw(4) << seq_len 
              << " | " << std::setw(10) << std::left << implementation 
              << " : " << std::fixed << std::setprecision(3) << time_ms 
              << " ms / iter" << std::endl;
}


void print_usage(const char* program_name) {
    std::cout << "用法: " << program_name << " <batch_size> <seq_length>" << std::endl;
    std::cout << "示例: " << program_name << " 1 1024" << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    int batch = std::stoi(argv[1]);
    int seq_length = std::stoi(argv[2]);

    int warmup_iters = 10;
    int running_iters = 20;
    int num_heads = 12; int hidden_dim = 768; int sparsity_param = 256;
    int tensor_size = batch * seq_length * hidden_dim;

    // Create float temporaries for filling, then convert to __half
    float * queries_f = new float[tensor_size]; float * keys_f = new float [tensor_size]; float * values_f = new float[tensor_size];
    float * answer_f = new float[tensor_size];
    float * l_f = new float[seq_length];
    float * m_f = new float[seq_length];

    // initialize float buffers
    std::memset(l_f, 0, sizeof(float)*seq_length);
    for(int i=0;i<seq_length;i++) m_f[i] = -1e9f;

    matrix_fill(queries_f, tensor_size);
    matrix_fill(keys_f, tensor_size);
    matrix_fill(values_f, tensor_size);
    // answer_f can be zero
    std::fill_n(answer_f, tensor_size, 0.0f);

    // Now allocate host-side __half arrays (to copy to device)
    __half * queries = new __half[tensor_size];
    __half * keys = new __half[tensor_size];
    __half * values = new __half[tensor_size];
    __half * answer = new __half[tensor_size];
    __half * l = new __half[seq_length];
    __half * m = new __half[seq_length];

    for (int i = 0; i < tensor_size; ++i) {
        queries[i] = __float2half(queries_f[i]);
        keys[i]    = __float2half(keys_f[i]);
        values[i]  = __float2half(values_f[i]);
        answer[i]  = __float2half(answer_f[i]);
    }
    for (int i = 0; i < seq_length; ++i) {
        l[i] = __float2half(l_f[i]);
        m[i] = __float2half(m_f[i]);
    }

    // free temporaries
    delete[] queries_f; delete[] keys_f; delete[] values_f; delete[] answer_f; delete[] l_f; delete[] m_f;

    coord * col_metadata = populate_col_metadata_blocked(seq_length, sparsity_param, BLOCK_SIZE_Y);

    // GPU memory (fp16)
    __half * queries_dev; __half * keys_dev; __half * values_dev; __half * answer_dev; __half * l_dev; __half * m_dev; coord* col_metadata_dev;
    CHECK_CUDA(cudaMalloc(&queries_dev, sizeof(__half)*tensor_size));
    CHECK_CUDA(cudaMalloc(&keys_dev, sizeof(__half)*tensor_size));
    CHECK_CUDA(cudaMalloc(&values_dev, sizeof(__half)*tensor_size));
    CHECK_CUDA(cudaMalloc(&answer_dev, sizeof(__half)*tensor_size));
    CHECK_CUDA(cudaMalloc(&l_dev, sizeof(__half)*seq_length));
    CHECK_CUDA(cudaMalloc(&m_dev, sizeof(__half)*seq_length));
    CHECK_CUDA(cudaMalloc(&col_metadata_dev, sizeof(coord)* int(ceil(float(seq_length)/float(BLOCK_SIZE_Y))) ));

    // Copy tensors to GPU.
    CHECK_CUDA(cudaMemcpy(queries_dev,queries, sizeof(__half)*tensor_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(keys_dev,keys, sizeof(__half)*tensor_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(values_dev,values, sizeof(__half)*tensor_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(answer_dev,answer, sizeof(__half)*tensor_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(l_dev,l, sizeof(__half)*seq_length, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m_dev,m, sizeof(__half)*seq_length, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(col_metadata_dev,col_metadata, sizeof(coord)*int(ceil(float(seq_length)/float(BLOCK_SIZE_Y))), cudaMemcpyHostToDevice));


    // std::cout << "=== Attention Performance ===" << std::endl;
    // std::cout << "SPLAT: num_heads=12, hidden_dim=768, sparsity_param=256" << std::endl;
    // std::cout << "=================================" << std::endl;

    
    auto time_start = time_now();
    for(int i = 0; i < warmup_iters + running_iters; i++) {
        if(i == warmup_iters){
            cudaDeviceSynchronize();
            auto time_start = time_now();
        }

        blocked_launcher<__half>(queries_dev, keys_dev, values_dev, 
                        answer_dev,l_dev,m_dev, batch, num_heads, seq_length, 
                            hidden_dim, sparsity_param, col_metadata_dev);
    }
    cudaDeviceSynchronize();
    auto time_elapsed = time_elapsed_us(time_start);
    float avg_time_ms = (time_elapsed / 1000.0f) / running_iters;


    std::cout << " bs:" << batch 
        << " | h_num:12 | seq:" << seq_length 
        << "  |  SPLAT "  << " :    " << std::fixed << std::setprecision(3) << avg_time_ms 
        << " ms / iter" << std::endl;

    return 0;
}