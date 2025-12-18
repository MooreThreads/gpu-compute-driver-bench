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

__global__ void delay(volatile int flag, unsigned long long timeout_clocks = 100000000) {
    long long int start_clock, sample_clock;
    start_clock = clock64();
    while (flag) {
        sample_clock = clock64();
        if (sample_clock - start_clock > timeout_clocks) {
            break;
        }
    }
}

__global__ void emptyKernel() {
}

int main(int argc, char** argv) {
    int deviceCount;
    checkMusaErrors(musaGetDeviceCount(&deviceCount));
    musaDeviceProp prop;
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    Printer::get().TableSetPbName("LaunchCnt");
    Run(argc, argv);
    return 0;
}

class SyncFixture : public TestFixture {
public:
    SyncFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t i = 100; i <= 1000; i += 100) {
            problemSpace.push_back(i);
        }
        return problemSpace;
    }
    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        synchronizedConut = experimentValue.Value;
        totalTime         = 0.f;
        totalCnt          = 0;
    }

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    int testNCommands(int n, float* t1, float* t2);

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->utime1, this->utime2};
    }

    uint32_t synchronizedConut;
    static float totalTime;
    static uint32_t totalCnt;
    std::shared_ptr<UDMGPUTime> utime1{new UDMGPUTime("KERNEL-us")};
    std::shared_ptr<UDMGPUTime> utime2{new UDMGPUTime("D2DCPY-us")};
};

float SyncFixture::totalTime   = 0.f;
uint32_t SyncFixture::totalCnt = 0;

int SyncFixture::testNCommands(int n, float* t1, float* t2) {
    CPerfCounter timer;
    musaStream_t stream;
    checkMusaErrors(musaStreamCreate(&stream));

    musaEvent_t start;
    musaEvent_t stop;
    checkMusaErrors(musaEventCreate(&start));
    checkMusaErrors(musaEventCreate(&stop));

    // warm up
    emptyKernel<<<1, 1>>>();
    checkMusaErrors(musaDeviceSynchronize());

    int flag                         = 1;
    const unsigned long long tickNum = 125000000;
    float pDurationMs                = 0;

    // test depend on kernel on default stream
    delay<<<1, 1>>>(flag, tickNum);
    checkMusaErrors(musaEventRecord(start, stream));
    for (uint64_t i = 0; i < n; ++i) {
        emptyKernel<<<1, 1, 0, stream>>>();
    }
    checkMusaErrors(musaEventRecord(stop, stream));
    checkMusaErrors(musaEventSynchronize(stop));
    checkMusaErrors(musaEventElapsedTime(&pDurationMs, start, stop));
    checkMusaErrors(musaDeviceSynchronize());
    float result1 = pDurationMs * 1000.f;
    *t1           = result1;

    // test depend on d2d memcpy on default stream
    pDurationMs = 0;
    int* device_x;
    int* device_y;
    int64_t cpySize = 1LL << 27;
    checkMusaErrors(musaMalloc(&device_x, cpySize));
    checkMusaErrors(musaMalloc(&device_y, cpySize));
    checkMusaErrors(musaMemcpy(device_y, device_x, cpySize, musaMemcpyDeviceToDevice));
    checkMusaErrors(musaEventRecord(start, stream));
    for (uint64_t i = 0; i < n; ++i) {
        emptyKernel<<<1, 1, 0, stream>>>();
    }
    checkMusaErrors(musaEventRecord(stop, stream));
    checkMusaErrors(musaEventSynchronize(stop));
    checkMusaErrors(musaEventElapsedTime(&pDurationMs, start, stop));
    checkMusaErrors(musaDeviceSynchronize());
    float result2 = pDurationMs * 1000.f;
    *t2           = result2;

    checkMusaErrors(musaFree(device_x));
    checkMusaErrors(musaFree(device_y));
    checkMusaErrors(musaEventDestroy(start));
    checkMusaErrors(musaEventDestroy(stop));
    checkMusaErrors(musaStreamDestroy(stream));
    return 0;
}

BASELINE_F(launchWithDepend, kernelLaunch, SyncFixture, SamplesCount, IterationsCount) {
    float result1, result2;
    int ans = testNCommands(synchronizedConut, &result1, &result2);
    this->utime1->addValue(result1 / synchronizedConut); // us
    this->utime2->addValue(result2 / synchronizedConut); // us
}