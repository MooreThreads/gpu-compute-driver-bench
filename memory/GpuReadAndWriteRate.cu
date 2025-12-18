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

#include "Celero.h"
#include "UserDefinedMeasurements.h"

#include "musa.h"
#include "musa_runtime.h"
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <errno.h>
#include <string>

#include "timer_his.h"

#include "helper_musa.h"
#include "helper_musa_drvapi.h"

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

__global__ void writeKernel(int4* d_data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_data[idx] = make_int4(0xff, 0xff, 0xff, 0xff);
    }
}

__global__ void readKernel(int4* d_data, int4* d_out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[0] = d_data[idx];
    }
}

__global__ void readWriteKernel(int4* d_data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int4 temp = d_data[idx];
        temp.x += 1;
        temp.y += 1;
        temp.z += 1;
        temp.w += 1;
        d_data[idx] = temp;
    }
}

class GpuReadAndWriteRateFixture : public TestFixture {
public:
    enum TestMode : int { Read, Write, ReadWrite };

    GpuReadAndWriteRateFixture() {}

    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t elements = 1LL << 4; elements <= (1LL << 32); elements *= 8) {
            // 1LL << 4 = 16, as the minimum size of the problem space is sizeof(int4) = 16 bytes
            problemSpace.push_back(elements);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        this->mallocSize = experimentValue.Value;
    }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->ubandRead, this->ubandWrite, this->ubandRW};
    }

    void ReadOrWriteTest(TestMode mode, void* hostPtr, size_t size, double* duration);

    int64_t mallocSize;
    std::shared_ptr<UDMBandWidth> ubandRead{new UDMBandWidth("*R(MB/s)")};
    std::shared_ptr<UDMBandWidth> ubandWrite{new UDMBandWidth("*W(MB/s)")};
    std::shared_ptr<UDMBandWidth> ubandRW{new UDMBandWidth("*RW(MB/s)")};
};

void GpuReadAndWriteRateFixture::ReadOrWriteTest(TestMode mode, void* devicePtr, size_t size, double* durationUs) {
    CPerfCounter timer;
    void* warmupDevicePtr = nullptr;
    checkMusaErrors(musaMalloc(&warmupDevicePtr, 4096));
    writeKernel<<<1, 1>>>((int4*)warmupDevicePtr, 1);
    readKernel<<<1, 1>>>((int4*)warmupDevicePtr, (int4*)warmupDevicePtr, 1);
    readWriteKernel<<<1, 1>>>((int4*)warmupDevicePtr, 1);
    checkMusaErrors(musaDeviceSynchronize());

    int threadsPerBlock = 1024;
    int blocksPerGrid   = (size / sizeof(int4) + threadsPerBlock - 1) / threadsPerBlock;
    timer.Restart();
    switch (mode) {
    case Read:
        readKernel<<<blocksPerGrid, threadsPerBlock>>>((int4*)devicePtr, (int4*)devicePtr, size / sizeof(int4));
        break;
    case Write:
        writeKernel<<<blocksPerGrid, threadsPerBlock>>>((int4*)devicePtr, size / sizeof(int4));
        break;
    case ReadWrite:
        readWriteKernel<<<blocksPerGrid, threadsPerBlock>>>((int4*)devicePtr, size / sizeof(int4));
        break;
    }
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    *durationUs = timer.GetElapsedSeconds() * 1000 * 1000;
    checkMusaErrors(musaFree(warmupDevicePtr));
};

static const int SamplesCount    = 3;
static const int IterationsCount = 1;

BASELINE_F(hostRW, musaMalloc, GpuReadAndWriteRateFixture, SamplesCount, IterationsCount) {
    void* devicePtr = nullptr;
    checkMusaErrors(musaMalloc(&devicePtr, this->mallocSize));
    double durationUs = 0.0;
    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::Write, devicePtr, this->mallocSize, &durationUs);
    this->ubandWrite->addValue(static_cast<double>(mallocSize) / (durationUs));

    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::Read, devicePtr, this->mallocSize, &durationUs);
    this->ubandRead->addValue(static_cast<double>(mallocSize) / (durationUs));

    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::ReadWrite, devicePtr, this->mallocSize, &durationUs);
    this->ubandRW->addValue(static_cast<double>(mallocSize) * 2 / (durationUs));

    checkMusaErrors(musaFree(devicePtr));
}

BENCHMARK_F(hostRW, musaMallocHost, GpuReadAndWriteRateFixture, SamplesCount, IterationsCount) {
    char* hostPtr   = nullptr;
    char* devicePtr = nullptr;

    checkMusaErrors(musaMallocHost((void**)&hostPtr, this->mallocSize));
    checkMusaErrors(musaHostGetDevicePointer((void**)&devicePtr, (void*)hostPtr, 0));

    double durationUs = 0.0;
    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::Write, devicePtr, this->mallocSize, &durationUs);
    this->ubandWrite->addValue(static_cast<double>(mallocSize) / (durationUs));

    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::Read, devicePtr, this->mallocSize, &durationUs);
    this->ubandRead->addValue(static_cast<double>(mallocSize) / (durationUs));

    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::ReadWrite, devicePtr, this->mallocSize, &durationUs);
    this->ubandRW->addValue(static_cast<double>(mallocSize) * 2 / (durationUs));
    checkMusaErrors(musaFreeHost(hostPtr));
}

BENCHMARK_F(hostRW, musaMallocManaged, GpuReadAndWriteRateFixture, SamplesCount, IterationsCount) {
    char* hostPtr   = nullptr;
    char* devicePtr = nullptr;

    checkMusaErrors(musaMallocManaged((void**)&devicePtr, this->mallocSize));
    hostPtr = devicePtr; // unified memory

    double durationUs = 0.0;
    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::Write, devicePtr, this->mallocSize, &durationUs);
    this->ubandWrite->addValue(static_cast<double>(mallocSize) / (durationUs));

    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::Read, devicePtr, this->mallocSize, &durationUs);
    this->ubandRead->addValue(static_cast<double>(mallocSize) / (durationUs));

    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::ReadWrite, devicePtr, this->mallocSize, &durationUs);
    this->ubandRW->addValue(static_cast<double>(mallocSize) * 2 / (durationUs));
    checkMusaErrors(musaFree(devicePtr));
}

BENCHMARK_F(hostRW, musaHostRegister, GpuReadAndWriteRateFixture, SamplesCount, IterationsCount) {
    char* devicePtr = nullptr;
    char* hostPtr   = reinterpret_cast<char*>(malloc(this->mallocSize));
    if (hostPtr == nullptr) {
        std::cerr << "malloc failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    checkMusaErrors(musaHostRegister((void*)hostPtr, this->mallocSize, 0));
    checkMusaErrors(musaHostGetDevicePointer((void**)&devicePtr, (void*)hostPtr, 0));

    double durationUs = 0.0;
    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::Write, devicePtr, this->mallocSize, &durationUs);
    this->ubandWrite->addValue(static_cast<double>(mallocSize) / (durationUs));

    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::Read, devicePtr, this->mallocSize, &durationUs);
    this->ubandRead->addValue(static_cast<double>(mallocSize) / (durationUs));

    ReadOrWriteTest(GpuReadAndWriteRateFixture::TestMode::ReadWrite, devicePtr, this->mallocSize, &durationUs);
    this->ubandRW->addValue(static_cast<double>(mallocSize) * 2 / (durationUs));
    checkMusaErrors(musaHostUnregister(hostPtr));
    free(hostPtr);
}