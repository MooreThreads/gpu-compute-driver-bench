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

#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <assert.h>
#include <string>
#include <iomanip>
#include <cmath>

#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "musa.h"
#include "musa_runtime.h"
#include "timer_his.h"
#include "helper_musa.h"
#include "helper_musa_drvapi.h"

static const int SamplesCount    = 1;
static const int IterationsCount = 1;

__global__ void compute_bound_kernel(float* a, float* b, float* out, int N) {
    int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int idx = tid; idx < N; idx += total_threads) {
        float x = 1.0f + 0.0001f * idx;
#pragma unroll 100
        for (int i = 0; i < 100; ++i) {
            x = x * 1.000001f + 0.00001f;
        }
        out[idx] = x;
    }
}

__global__ void memory_bound_kernel(float* a, float* b, float* out, int N) {
    int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int idx = tid; idx < N; idx += total_threads) {
        out[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char** argv) {
    int deviceCount;
    checkMusaErrors(musaGetDeviceCount(&deviceCount));
    musaDeviceProp prop;
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    Printer::get().TableSetPbName("Bytes");
    Run(argc, argv);
    return 0;
}

class LanchFixture : public TestFixture {
public:
    enum class GPUAsyncCmdType : uint32_t {
        MemcpyDevice2Device     = 1,
        MemcpyDevice2PinnedHost = 2,
        MemcpyPinnedHost2Device = 3,
        Memset                  = 4,
        ComputeBoundKernel      = 5,
        MemoryBoundKernel       = 6,
    };

    struct AsyncCmdParameter {
        GPUAsyncCmdType cmdType;
    };

    struct MemsetParameter : public AsyncCmdParameter {
        void* devicePtr;
        size_t size;
    };

    struct KernelParameter : public AsyncCmdParameter {
        void* inputPtr1;
        void* inputPtr2;
        void* outputPtr1;
        size_t size;
    };

    struct MemcpyParameter : public AsyncCmdParameter {
        void* dstPtr;
        void* srcPtr;
        size_t size;
    };

    LanchFixture() {}

    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;

        problemSpace.push_back(1ull << 30);

        return problemSpace;
    }

    void PrepareCommand(AsyncCmdParameter& para);
    void PrepareMemsetCommand(MemsetParameter& para);
    void PrepareMemcpyCommand(MemcpyParameter& para);
    void PrepareKernelCommand(KernelParameter& para);

    void ExecuteCommand(AsyncCmdParameter& para, MUstream stream, MUevent begin, MUevent end);
    void ExecuteMemsetCommand(MemsetParameter& para, MUstream stream, MUevent begin, MUevent end);
    void ExecuteMemcpyCommand(MemcpyParameter& para, MUstream stream, MUevent begin, MUevent end);
    void ExecuteKernelCommand(KernelParameter& para, MUstream stream, MUevent begin, MUevent end);

    void ReleaseCommand(AsyncCmdParameter& para);
    void ReleaseMemsetCommand(MemsetParameter& para);
    void ReleaseMemcpyCommand(MemcpyParameter& para);
    void ReleaseKernelCommand(KernelParameter& para);

    void ParallelTwoCommands(AsyncCmdParameter& cmd1, AsyncCmdParameter& cmd2, float& time1, float& time2, float& time);
    void SerialTwoCommands(AsyncCmdParameter& cmd1, AsyncCmdParameter& cmd2, float& time1, float& time2, float& time);

    void setUp(const TestFixture::ExperimentValue& experimentValue) override { size = experimentValue.Value; }
    void tearDown() override {}
    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->utimeCmdSerial1, this->utimeCmdSerial2, this->utimeCmdSerial, this->utimeCmdParallel1,
            this->utimeCmdParallel2, this->utimeCmdParallel, this->uspeedUp};
    }

    std::shared_ptr<UDMGPUTime> utimeCmdSerial1{new UDMGPUTime("t+1")};
    std::shared_ptr<UDMGPUTime> utimeCmdSerial2{new UDMGPUTime("t+2")};
    std::shared_ptr<UDMGPUTime> utimeCmdSerial{new UDMGPUTime("t1+2")};

    std::shared_ptr<UDMGPUTime> utimeCmdParallel1{new UDMGPUTime("t|1")};
    std::shared_ptr<UDMGPUTime> utimeCmdParallel2{new UDMGPUTime("t|2")};
    std::shared_ptr<UDMGPUTime> utimeCmdParallel{new UDMGPUTime("t1|2")};

    std::shared_ptr<UDMRatio> uspeedUp{new UDMRatio("*t+/t|")};

    uint64_t size;
};

void LanchFixture::PrepareCommand(AsyncCmdParameter& para) {
    switch (para.cmdType) {
    case GPUAsyncCmdType::ComputeBoundKernel:
    case GPUAsyncCmdType::MemoryBoundKernel:
        PrepareKernelCommand(static_cast<KernelParameter&>(para));
        break;
    case GPUAsyncCmdType::Memset:
        PrepareMemsetCommand(static_cast<MemsetParameter&>(para));
        break;
    case GPUAsyncCmdType::MemcpyDevice2Device:
    case GPUAsyncCmdType::MemcpyDevice2PinnedHost:
    case GPUAsyncCmdType::MemcpyPinnedHost2Device:
        PrepareMemcpyCommand(static_cast<MemcpyParameter&>(para));
        break;
    default:
        assert(0);
    }
}

void LanchFixture::PrepareMemsetCommand(MemsetParameter& para) {
    para.cmdType = GPUAsyncCmdType::Memset;
    checkMusaErrors(musaMalloc(&para.devicePtr, para.size));
}

void LanchFixture::PrepareMemcpyCommand(MemcpyParameter& para) {
    enum MemoryType { DeviceMemory, PinnedHostMemory };

    MemoryType srcMemtype, dstMemtype;

    switch (para.cmdType) {
    case GPUAsyncCmdType::MemcpyDevice2Device:
        srcMemtype = DeviceMemory;
        dstMemtype = DeviceMemory;
        break;
    case GPUAsyncCmdType::MemcpyDevice2PinnedHost:
        srcMemtype = DeviceMemory;
        dstMemtype = PinnedHostMemory;
        break;
    case GPUAsyncCmdType::MemcpyPinnedHost2Device:
        srcMemtype = PinnedHostMemory;
        dstMemtype = DeviceMemory;
        break;
    default:
        assert(0);
    }

    switch (srcMemtype) {
    case DeviceMemory:
        checkMusaErrors(musaMalloc(&para.srcPtr, para.size));
        break;
    case PinnedHostMemory:
        checkMusaErrors(musaHostAlloc(&para.srcPtr, para.size, musaHostAllocDefault));
        break;
    default:
        assert(0);
    }

    switch (dstMemtype) {
    case DeviceMemory:
        checkMusaErrors(musaMalloc(&para.dstPtr, para.size));
        break;
    case PinnedHostMemory:
        checkMusaErrors(musaHostAlloc(&para.dstPtr, para.size, musaHostAllocDefault));
        break;
    default:
        assert(0);
    }
}

void LanchFixture::PrepareKernelCommand(KernelParameter& para) {
    checkMusaErrors(musaMalloc(&para.inputPtr1, para.size));
    checkMusaErrors(musaMalloc(&para.inputPtr2, para.size));
    checkMusaErrors(musaMalloc(&para.outputPtr1, para.size));
}

void LanchFixture::ExecuteCommand(AsyncCmdParameter& para, MUstream stream, MUevent begin, MUevent end) {
    checkMusaErrors(musaEventRecord(begin, stream));
    switch (para.cmdType) {
    case GPUAsyncCmdType::ComputeBoundKernel:
    case GPUAsyncCmdType::MemoryBoundKernel:
        ExecuteKernelCommand(static_cast<KernelParameter&>(para), stream, begin, end);
        break;
    case GPUAsyncCmdType::Memset:
        ExecuteMemsetCommand(static_cast<MemsetParameter&>(para), stream, begin, end);
        break;
    case GPUAsyncCmdType::MemcpyDevice2Device:
    case GPUAsyncCmdType::MemcpyDevice2PinnedHost:
    case GPUAsyncCmdType::MemcpyPinnedHost2Device:
        ExecuteMemcpyCommand(static_cast<MemcpyParameter&>(para), stream, begin, end);
        break;
    default:
        assert(0);
    }
}

void LanchFixture::ExecuteMemsetCommand(MemsetParameter& para, MUstream stream, MUevent begin, MUevent end) {
    checkMusaErrors(musaEventRecord(begin, stream));
    checkMusaErrors(musaMemsetAsync(para.devicePtr, 0, para.size, stream));
    checkMusaErrors(musaEventRecord(end, stream));
}

void LanchFixture::ExecuteMemcpyCommand(MemcpyParameter& para, MUstream stream, MUevent begin, MUevent end) {
    checkMusaErrors(musaEventRecord(begin, stream));
    checkMusaErrors(musaMemcpyAsync(para.dstPtr, para.srcPtr, para.size, musaMemcpyDefault, stream));
    checkMusaErrors(musaEventRecord(end, stream));
}

void LanchFixture::ExecuteKernelCommand(KernelParameter& para, MUstream stream, MUevent begin, MUevent end) {
    checkMusaErrors(musaEventRecord(begin, stream));

    int N         = para.size / sizeof(float);
    int blockSize = 512;
    int gridSize  = 32;

    switch (para.cmdType) {
    case GPUAsyncCmdType::ComputeBoundKernel:
        compute_bound_kernel<<<gridSize, blockSize, 0, stream>>>(static_cast<float*>(para.inputPtr1),
            static_cast<float*>(para.inputPtr2), static_cast<float*>(para.outputPtr1), N);
        break;

    case GPUAsyncCmdType::MemoryBoundKernel:
        memory_bound_kernel<<<gridSize, blockSize, 0, stream>>>(static_cast<float*>(para.inputPtr1),
            static_cast<float*>(para.inputPtr2), static_cast<float*>(para.outputPtr1), N);
        break;

    default:
        assert(0);
    }

    checkMusaErrors(musaGetLastError());

    checkMusaErrors(musaEventRecord(end, stream));
}

void LanchFixture::ReleaseCommand(AsyncCmdParameter& para) {
    switch (para.cmdType) {
    case GPUAsyncCmdType::ComputeBoundKernel:
    case GPUAsyncCmdType::MemoryBoundKernel:
        ReleaseKernelCommand(static_cast<KernelParameter&>(para));
        break;
    case GPUAsyncCmdType::Memset:
        ReleaseMemsetCommand(static_cast<MemsetParameter&>(para));
        break;
    case GPUAsyncCmdType::MemcpyDevice2Device:
    case GPUAsyncCmdType::MemcpyDevice2PinnedHost:
    case GPUAsyncCmdType::MemcpyPinnedHost2Device:
        ReleaseMemcpyCommand(static_cast<MemcpyParameter&>(para));
        break;
    default:
        assert(0);
    }
}

void LanchFixture::ReleaseMemsetCommand(MemsetParameter& para) {
    checkMusaErrors(musaFree(para.devicePtr));
}

void LanchFixture::ReleaseKernelCommand(KernelParameter& para) {
    checkMusaErrors(musaFree(para.inputPtr1));
    checkMusaErrors(musaFree(para.inputPtr2));
    checkMusaErrors(musaFree(para.outputPtr1));
}

void LanchFixture::ReleaseMemcpyCommand(MemcpyParameter& para) {
    enum MemoryType { DeviceMemory, PinnedHostMemory };

    MemoryType srcMemtype, dstMemtype;

    switch (para.cmdType) {
    case GPUAsyncCmdType::MemcpyDevice2Device:
        srcMemtype = DeviceMemory;
        dstMemtype = DeviceMemory;
        break;
    case GPUAsyncCmdType::MemcpyDevice2PinnedHost:
        srcMemtype = DeviceMemory;
        dstMemtype = PinnedHostMemory;
        break;
    case GPUAsyncCmdType::MemcpyPinnedHost2Device:
        srcMemtype = PinnedHostMemory;
        dstMemtype = DeviceMemory;
        break;
    default:
        assert(0);
    }

    switch (srcMemtype) {
    case DeviceMemory:
        checkMusaErrors(musaFree(para.srcPtr));
        break;
    case PinnedHostMemory:
        checkMusaErrors(musaFreeHost(para.srcPtr));
        break;
    default:
        assert(0);
    }

    switch (dstMemtype) {
    case DeviceMemory:
        checkMusaErrors(musaFree(para.dstPtr));
        break;
    case PinnedHostMemory:
        checkMusaErrors(musaFreeHost(para.dstPtr));
        break;
    default:
        assert(0);
    }
}

void LanchFixture::ParallelTwoCommands(
    AsyncCmdParameter& cmd1, AsyncCmdParameter& cmd2, float& time1, float& time2, float& time) {
    MUstream stream1;
    MUstream stream2;
    checkMusaErrors(musaStreamCreate(&stream1));
    checkMusaErrors(musaStreamCreate(&stream2));

    MUevent begin1;
    MUevent end1;
    MUevent begin2;
    MUevent end2;

    checkMusaErrors(musaEventCreate(&begin1));
    checkMusaErrors(musaEventCreate(&end1));
    checkMusaErrors(musaEventCreate(&begin2));
    checkMusaErrors(musaEventCreate(&end2));

    CPerfCounter perfCounter;
    perfCounter.Restart();
    checkMusaErrors(musaEventRecord(begin1, stream1));
    checkMusaErrors(musaEventRecord(begin2, stream2));

    ExecuteCommand(cmd1, stream1, begin1, end1);
    ExecuteCommand(cmd2, stream2, begin2, end2);

    checkMusaErrors(musaEventRecord(end1, stream1));
    checkMusaErrors(musaEventRecord(end2, stream2));
    checkMusaErrors(musaEventSynchronize(end1));
    checkMusaErrors(musaEventSynchronize(end2));
    perfCounter.Stop();

    checkMusaErrors(musaEventElapsedTime(&time1, begin1, end1));
    checkMusaErrors(musaEventElapsedTime(&time2, begin2, end2));

    time = perfCounter.GetElapsedSeconds() * 1000.0f;

    checkMusaErrors(musaStreamDestroy(stream1));
    checkMusaErrors(musaStreamDestroy(stream2));
    checkMusaErrors(musaEventDestroy(begin1));
    checkMusaErrors(musaEventDestroy(end1));
    checkMusaErrors(musaEventDestroy(begin2));
    checkMusaErrors(musaEventDestroy(end2));
}

void LanchFixture::SerialTwoCommands(
    AsyncCmdParameter& cmd1, AsyncCmdParameter& cmd2, float& time1, float& time2, float& time) {
    MUstream stream1;
    MUstream stream2;
    checkMusaErrors(musaStreamCreate(&stream1));
    checkMusaErrors(musaStreamCreate(&stream2));
    MUevent begin1;
    MUevent end1;
    MUevent begin2;
    MUevent end2;
    checkMusaErrors(musaEventCreate(&begin1));
    checkMusaErrors(musaEventCreate(&end1));
    checkMusaErrors(musaEventCreate(&begin2));
    checkMusaErrors(musaEventCreate(&end2));
    checkMusaErrors(musaEventRecord(begin1, stream1));
    checkMusaErrors(musaEventRecord(begin2, stream2));
    CPerfCounter perfCounter;
    perfCounter.Restart();
    ExecuteCommand(cmd1, stream1, begin1, end1);
    checkMusaErrors(musaEventRecord(end1, stream1));
    checkMusaErrors(musaEventSynchronize(end1));
    checkMusaErrors(musaEventRecord(begin2, stream2));
    ExecuteCommand(cmd2, stream2, begin2, end2);
    checkMusaErrors(musaEventRecord(end2, stream2));
    checkMusaErrors(musaEventSynchronize(end2));
    perfCounter.Stop();
    checkMusaErrors(musaEventElapsedTime(&time1, begin1, end1));
    checkMusaErrors(musaEventElapsedTime(&time2, begin2, end2));
    time = perfCounter.GetElapsedSeconds() * 1000.0f;
}

BASELINE_F(Parallelism, DtoD_DtoD, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    MemcpyParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyDevice2Device;
    cmd2.cmdType = GPUAsyncCmdType::MemcpyDevice2Device;

    cmd1.size = size;
    cmd2.size = size;

    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, DtoD_DtoPH, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    MemcpyParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyDevice2Device;
    cmd2.cmdType = GPUAsyncCmdType::MemcpyDevice2PinnedHost;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, DtoD_PHtoD, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    MemcpyParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyDevice2Device;
    cmd2.cmdType = GPUAsyncCmdType::MemcpyDevice2PinnedHost;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, DtoD_SET, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    MemsetParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyDevice2Device;
    cmd2.cmdType = GPUAsyncCmdType::Memset;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, DtoD_CBK, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    KernelParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyDevice2Device;
    cmd2.cmdType = GPUAsyncCmdType::ComputeBoundKernel;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, DtoD_MBK, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    KernelParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyDevice2Device;
    cmd2.cmdType = GPUAsyncCmdType::MemoryBoundKernel;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, DtoPH_DtoPH, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    MemcpyParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyDevice2PinnedHost;
    cmd2.cmdType = GPUAsyncCmdType::MemcpyDevice2PinnedHost;
    cmd1.size    = 1024 * 1024 * 1024;
    cmd2.size    = 1024 * 1024 * 1024;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, DtoPH_PHtoD, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    MemcpyParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyDevice2PinnedHost;
    cmd2.cmdType = GPUAsyncCmdType::MemcpyPinnedHost2Device;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, DtoPH_SET, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    MemsetParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyDevice2PinnedHost;
    cmd2.cmdType = GPUAsyncCmdType::Memset;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, DtoPH_CBK, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    KernelParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyDevice2PinnedHost;
    cmd2.cmdType = GPUAsyncCmdType::ComputeBoundKernel;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, DtoPH_MBK, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    KernelParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyDevice2PinnedHost;
    cmd2.cmdType = GPUAsyncCmdType::MemoryBoundKernel;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, PHtoD_PHtoD, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    MemcpyParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyPinnedHost2Device;
    cmd2.cmdType = GPUAsyncCmdType::MemcpyPinnedHost2Device;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, PHtoD_SET, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    MemsetParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyPinnedHost2Device;
    cmd2.cmdType = GPUAsyncCmdType::Memset;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, PHtoD_CBK, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    KernelParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyPinnedHost2Device;
    cmd2.cmdType = GPUAsyncCmdType::ComputeBoundKernel;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, PHtoD_MBK, LanchFixture, SamplesCount, IterationsCount) {
    MemcpyParameter cmd1;
    KernelParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemcpyPinnedHost2Device;
    cmd2.cmdType = GPUAsyncCmdType::MemoryBoundKernel;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, SET_SET, LanchFixture, SamplesCount, IterationsCount) {
    MemsetParameter cmd1;
    MemsetParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::Memset;
    cmd2.cmdType = GPUAsyncCmdType::Memset;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, SET_CBK, LanchFixture, SamplesCount, IterationsCount) {
    MemsetParameter cmd1;
    KernelParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::Memset;
    cmd2.cmdType = GPUAsyncCmdType::ComputeBoundKernel;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, SET_MBK, LanchFixture, SamplesCount, IterationsCount) {
    MemsetParameter cmd1;
    KernelParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::Memset;
    cmd2.cmdType = GPUAsyncCmdType::MemoryBoundKernel;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, CBK_CBK, LanchFixture, SamplesCount, IterationsCount) {
    KernelParameter cmd1;
    KernelParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::ComputeBoundKernel;
    cmd2.cmdType = GPUAsyncCmdType::ComputeBoundKernel;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, CBK_MBK, LanchFixture, SamplesCount, IterationsCount) {
    KernelParameter cmd1;
    KernelParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::ComputeBoundKernel;
    cmd2.cmdType = GPUAsyncCmdType::MemoryBoundKernel;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}

BENCHMARK_F(Parallelism, MBK_MBK, LanchFixture, SamplesCount, IterationsCount) {
    KernelParameter cmd1;
    KernelParameter cmd2;
    cmd1.cmdType = GPUAsyncCmdType::MemoryBoundKernel;
    cmd2.cmdType = GPUAsyncCmdType::MemoryBoundKernel;
    cmd1.size    = size;
    cmd2.size    = size;
    PrepareCommand(cmd1);
    PrepareCommand(cmd2);

    float time  = 0;
    float time0 = 0;
    float time1 = 0;
    float time2 = 0;

    ParallelTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdParallel1->addValue(time1);
    utimeCmdParallel2->addValue(time2);
    utimeCmdParallel->addValue(time);
    time0 = time;

    SerialTwoCommands(cmd1, cmd2, time1, time2, time);
    utimeCmdSerial1->addValue(time1);
    utimeCmdSerial2->addValue(time2);
    utimeCmdSerial->addValue(time);

    float radio = time / time0;
    // printf("debug:%f\n", radio);
    uspeedUp->addValue(radio);

    ReleaseCommand(cmd1);
    ReleaseCommand(cmd2);
}
