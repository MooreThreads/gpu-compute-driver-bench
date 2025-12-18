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

class CpuReadAndWriteRateFixture : public TestFixture {
public:
    CpuReadAndWriteRateFixture() {}

    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t elements = 1LL << 0; elements <= (1LL << 32); elements *= 8) {
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
        return {this->ubandRead, this->ubandWrite};
    }

    void ReadOrWriteTest(bool rnw, void* hostPtr, size_t size, double* duration);

    int64_t mallocSize;
    std::shared_ptr<UDMBandWidth> ubandRead{new UDMBandWidth("*BR(MB/s)")};
    std::shared_ptr<UDMBandWidth> ubandWrite{new UDMBandWidth("*BW(MB/s)")};
};

void CpuReadAndWriteRateFixture::ReadOrWriteTest(bool rnw, void* hostPtr, size_t size, double* durationUs) {
    CPerfCounter timer;
    char tempChar   = Random() & 0xff;
    char* pTempChar = (char*)malloc(size);
    timer.Start();

    if (rnw) {
        memcpy(pTempChar, hostPtr, size);
    } else {
        memcpy(hostPtr, pTempChar, size);
    }
    timer.Stop();
    *durationUs = timer.GetElapsedSeconds() * 1000.f * 1000.f;
    free(pTempChar);
};

static const int SamplesCount    = 3;
static const int IterationsCount = 1;

BASELINE_F(hostRW, malloc, CpuReadAndWriteRateFixture, SamplesCount, IterationsCount) {
    char* hostPtr = reinterpret_cast<char*>(malloc(this->mallocSize));
    if (hostPtr == nullptr) {
        std::cerr << "malloc failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    double durationUs = 0.0;
    ReadOrWriteTest(true, hostPtr, this->mallocSize, &durationUs);
    this->ubandWrite->addValue(static_cast<double>(mallocSize) / (durationUs));
    ReadOrWriteTest(false, hostPtr, this->mallocSize, &durationUs);
    this->ubandRead->addValue(static_cast<double>(mallocSize) / (durationUs));
    free(hostPtr);
}

BENCHMARK_F(hostRW, musaMallocHost, CpuReadAndWriteRateFixture, SamplesCount, IterationsCount) {
    char* hostPtr   = nullptr;
    char* devicePtr = nullptr;

    checkMusaErrors(musaMallocHost((void**)&hostPtr, this->mallocSize));

    double durationUs = 0.0;
    ReadOrWriteTest(true, hostPtr, this->mallocSize, &durationUs);
    this->ubandWrite->addValue(static_cast<double>(mallocSize) / (durationUs)); // B/us -> MB/s
    ReadOrWriteTest(false, hostPtr, this->mallocSize, &durationUs);
    this->ubandRead->addValue(static_cast<double>(mallocSize) / (durationUs)); // B/us -> MB/s
    checkMusaErrors(musaFreeHost(hostPtr));
}

BENCHMARK_F(hostRW, musaMallocManaged, CpuReadAndWriteRateFixture, SamplesCount, IterationsCount) {
    char* hostPtr   = nullptr;
    char* devicePtr = nullptr;

    checkMusaErrors(musaMallocManaged((void**)&devicePtr, this->mallocSize));
    hostPtr = devicePtr; // unified memory

    double durationUs = 0.0;
    ReadOrWriteTest(true, hostPtr, this->mallocSize, &durationUs);
    this->ubandWrite->addValue(static_cast<double>(mallocSize) / (durationUs)); // B/us -> MB/s
    ReadOrWriteTest(false, hostPtr, this->mallocSize, &durationUs);
    this->ubandRead->addValue(static_cast<double>(mallocSize) / (durationUs)); // B/us -> MB/s
    checkMusaErrors(musaFree(devicePtr));
}

BENCHMARK_F(hostRW, musaHostRegister, CpuReadAndWriteRateFixture, SamplesCount, IterationsCount) {
    char* devicePtr = nullptr;
    char* hostPtr   = reinterpret_cast<char*>(malloc(this->mallocSize));
    if (hostPtr == nullptr) {
        std::cerr << "malloc failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    checkMusaErrors(musaHostRegister((void*)hostPtr, this->mallocSize, 0));

    double durationUs = 0.0;
    ReadOrWriteTest(true, hostPtr, this->mallocSize, &durationUs);
    this->ubandWrite->addValue(static_cast<double>(mallocSize) / (durationUs)); // B/us -> MB/s
    ReadOrWriteTest(false, hostPtr, this->mallocSize, &durationUs);
    this->ubandRead->addValue(static_cast<double>(mallocSize) / (durationUs)); // B/us -> MB/s

    checkMusaErrors(musaHostUnregister(hostPtr));
    free(hostPtr);
}