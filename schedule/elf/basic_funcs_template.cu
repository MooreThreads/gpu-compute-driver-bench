/**
 * Copyright 2025 Moore Threads Technology Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>

template<typename T> __global__ void vectorAddition(const T* a, const T* b, T* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}
template __global__ void vectorAddition<int8_t>(const int8_t* a, const int8_t* b, int8_t* result, int size);
template __global__ void vectorAddition<int16_t>(const int16_t* a, const int16_t* b, int16_t* result, int size);
template __global__ void vectorAddition<int32_t>(const int32_t* a, const int32_t* b, int32_t* result, int size);
template __global__ void vectorAddition<int64_t>(const int64_t* a, const int64_t* b, int64_t* result, int size);
template __global__ void vectorAddition<uint8_t>(const uint8_t* a, const uint8_t* b, uint8_t* result, int size);
template __global__ void vectorAddition<uint16_t>(const uint16_t* a, const uint16_t* b, uint16_t* result, int size);
template __global__ void vectorAddition<uint32_t>(const uint32_t* a, const uint32_t* b, uint32_t* result, int size);
template __global__ void vectorAddition<uint64_t>(const uint64_t* a, const uint64_t* b, uint64_t* result, int size);
template __global__ void vectorAddition<float>(const float* a, const float* b, float* result, int size);
template __global__ void vectorAddition<double>(const double* a, const double* b, double* result, int size);

template<typename T> __global__ void matrixMultiplication(const T* A, const T* B, T* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        T sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}
template __global__ void matrixMultiplication<int8_t>(const int8_t* A, const int8_t* B, int8_t* C, int m, int n, int k);
template __global__ void
matrixMultiplication<int16_t>(const int16_t* A, const int16_t* B, int16_t* C, int m, int n, int k);
template __global__ void
matrixMultiplication<int32_t>(const int32_t* A, const int32_t* B, int32_t* C, int m, int n, int k);
template __global__ void
matrixMultiplication<int64_t>(const int64_t* A, const int64_t* B, int64_t* C, int m, int n, int k);
template __global__ void
matrixMultiplication<uint8_t>(const uint8_t* A, const uint8_t* B, uint8_t* C, int m, int n, int k);
template __global__ void
matrixMultiplication<uint16_t>(const uint16_t* A, const uint16_t* B, uint16_t* C, int m, int n, int k);
template __global__ void
matrixMultiplication<uint32_t>(const uint32_t* A, const uint32_t* B, uint32_t* C, int m, int n, int k);
template __global__ void
matrixMultiplication<uint64_t>(const uint64_t* A, const uint64_t* B, uint64_t* C, int m, int n, int k);
template __global__ void matrixMultiplication<float>(const float* A, const float* B, float* C, int m, int n, int k);
template __global__ void matrixMultiplication<double>(const double* A, const double* B, double* C, int m, int n, int k);

template<typename T> __global__ void parallelReduction(const T* input, T* output, int size) {
    __shared__ T sdata[100];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
template __global__ void parallelReduction<int8_t>(const int8_t* input, int8_t* output, int size);
template __global__ void parallelReduction<int16_t>(const int16_t* input, int16_t* output, int size);
template __global__ void parallelReduction<int32_t>(const int32_t* input, int32_t* output, int size);
template __global__ void parallelReduction<int64_t>(const int64_t* input, int64_t* output, int size);
template __global__ void parallelReduction<uint8_t>(const uint8_t* input, uint8_t* output, int size);
template __global__ void parallelReduction<uint16_t>(const uint16_t* input, uint16_t* output, int size);
template __global__ void parallelReduction<uint32_t>(const uint32_t* input, uint32_t* output, int size);
template __global__ void parallelReduction<uint64_t>(const uint64_t* input, uint64_t* output, int size);
template __global__ void parallelReduction<float>(const float* input, float* output, int size);
template __global__ void parallelReduction<double>(const double* input, double* output, int size);

template<typename T> __global__ void elementWiseMultiplication(const T* a, const T* b, T* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}
template __global__ void elementWiseMultiplication<int8_t>(const int8_t* a, const int8_t* b, int8_t* result, int size);
template __global__ void
elementWiseMultiplication<int16_t>(const int16_t* a, const int16_t* b, int16_t* result, int size);
template __global__ void
elementWiseMultiplication<int32_t>(const int32_t* a, const int32_t* b, int32_t* result, int size);
template __global__ void
elementWiseMultiplication<int64_t>(const int64_t* a, const int64_t* b, int64_t* result, int size);
template __global__ void
elementWiseMultiplication<uint8_t>(const uint8_t* a, const uint8_t* b, uint8_t* result, int size);
template __global__ void
elementWiseMultiplication<uint16_t>(const uint16_t* a, const uint16_t* b, uint16_t* result, int size);
template __global__ void
elementWiseMultiplication<uint32_t>(const uint32_t* a, const uint32_t* b, uint32_t* result, int size);
template __global__ void
elementWiseMultiplication<uint64_t>(const uint64_t* a, const uint64_t* b, uint64_t* result, int size);
template __global__ void elementWiseMultiplication<float>(const float* a, const float* b, float* result, int size);
template __global__ void elementWiseMultiplication<double>(const double* a, const double* b, double* result, int size);

template<typename T> __global__ void parallelScan(const T* input, T* output, int size) {
    __shared__ T sdata[100];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index + 2 * s - 1] += sdata[index + s - 1];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[blockDim.x - 1];
    }
}
template __global__ void parallelScan<int8_t>(const int8_t* input, int8_t* output, int size);
template __global__ void parallelScan<int16_t>(const int16_t* input, int16_t* output, int size);
template __global__ void parallelScan<int32_t>(const int32_t* input, int32_t* output, int size);
template __global__ void parallelScan<int64_t>(const int64_t* input, int64_t* output, int size);
template __global__ void parallelScan<uint8_t>(const uint8_t* input, uint8_t* output, int size);
template __global__ void parallelScan<uint16_t>(const uint16_t* input, uint16_t* output, int size);
template __global__ void parallelScan<uint32_t>(const uint32_t* input, uint32_t* output, int size);
template __global__ void parallelScan<uint64_t>(const uint64_t* input, uint64_t* output, int size);
template __global__ void parallelScan<float>(const float* input, float* output, int size);
template __global__ void parallelScan<double>(const double* input, double* output, int size);

template<typename T> __global__ void histogramCalculation(const T* data, int* histogram, int size, int bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&histogram[data[idx]], 1);
    }
}
template __global__ void histogramCalculation<int8_t>(const int8_t* data, int* histogram, int size, int bins);
template __global__ void histogramCalculation<int16_t>(const int16_t* data, int* histogram, int size, int bins);
template __global__ void histogramCalculation<int32_t>(const int32_t* data, int* histogram, int size, int bins);
template __global__ void histogramCalculation<int64_t>(const int64_t* data, int* histogram, int size, int bins);
template __global__ void histogramCalculation<uint8_t>(const uint8_t* data, int* histogram, int size, int bins);
template __global__ void histogramCalculation<uint16_t>(const uint16_t* data, int* histogram, int size, int bins);
template __global__ void histogramCalculation<uint32_t>(const uint32_t* data, int* histogram, int size, int bins);
template __global__ void histogramCalculation<uint64_t>(const uint64_t* data, int* histogram, int size, int bins);

template<typename T> __global__ void matrixTranspose(const T* input, T* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}
template __global__ void matrixTranspose<int8_t>(const int8_t* input, int8_t* output, int rows, int cols);
template __global__ void matrixTranspose<int16_t>(const int16_t* input, int16_t* output, int rows, int cols);
template __global__ void matrixTranspose<int32_t>(const int32_t* input, int32_t* output, int rows, int cols);
template __global__ void matrixTranspose<int64_t>(const int64_t* input, int64_t* output, int rows, int cols);
template __global__ void matrixTranspose<uint8_t>(const uint8_t* input, uint8_t* output, int rows, int cols);
template __global__ void matrixTranspose<uint16_t>(const uint16_t* input, uint16_t* output, int rows, int cols);
template __global__ void matrixTranspose<uint32_t>(const uint32_t* input, uint32_t* output, int rows, int cols);
template __global__ void matrixTranspose<uint64_t>(const uint64_t* input, uint64_t* output, int rows, int cols);
template __global__ void matrixTranspose<float>(const float* input, float* output, int rows, int cols);
template __global__ void matrixTranspose<double>(const double* input, double* output, int rows, int cols);