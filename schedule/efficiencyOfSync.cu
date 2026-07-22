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
#include "helper_musa.h"
#include "helper_musa_drvapi.h"

static const int SamplesCount    = 10;
static const int IterationsCount = 1;

__global__ void delay(volatile int* flag, unsigned long long timeout_clocks = 100000000) {
    long long int start_clock, sample_clock;
    start_clock = clock64();
    while (*flag) {
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
    Printer::get().TableSetPbName("SyncTime");
    Run(argc, argv);
    return 0;
}

class SyncFixture : public TestFixture {
public:
    SyncFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t i = 10000; i <= 50000; i += 10000) {
            problemSpace.push_back(i);
        }
        return problemSpace;
    }
    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        synchronizedConut = experimentValue.Value;
        totalTime         = 0.f;
        totalCnt          = 0;
    }

    void tearDown() override { this->utp->addValue(totalCnt * 1000.f * 1000.f / float(totalTime)); }

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    int testNCommands(int n, float* t1, float* t2);

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->utime1, this->utime2, this->utime3, this->utp};
    }

    uint32_t synchronizedConut;
    static float totalTime;
    static uint32_t totalCnt;
    std::shared_ptr<UDMGPUTime> utime1{new UDMGPUTime("t1str-us")};
    std::shared_ptr<UDMGPUTime> utime2{new UDMGPUTime("t2str-us")};
    std::shared_ptr<UDMGPUTime> utime3{new UDMGPUTime("twait-us")};
    // TP (throughput) is derived from the aggregate synchronization overhead:
    //   totalTime = sum(result2 - result1)
    //   totalCnt  = sum(synchronizedConut - 1)
    //   utp = totalCnt * 1e6 / totalTime  (events per second)
    // This measures the rate at which event-serialized operations are processed.
    // TODO: report single-stream and dual-stream throughput as separate metrics.
    std::shared_ptr<UDMThroughPut> utp{new UDMThroughPut("*TP(s^-1)")};
};

float SyncFixture::totalTime   = 0.f;
uint32_t SyncFixture::totalCnt = 0;

int SyncFixture::testNCommands(int n, float* t1, float* t2) {
    musaStream_t streams[2];
    checkMusaErrors(musaStreamCreate(&streams[0]));
    checkMusaErrors(musaStreamCreate(&streams[1]));

    musaEvent_t events[2];
    checkMusaErrors(musaEventCreate(&events[0]));
    checkMusaErrors(musaEventCreate(&events[1]));
    musaEvent_t baselineStart;
    musaEvent_t baselineStop;
    musaEvent_t eventStart;
    musaEvent_t eventStop;
    checkMusaErrors(musaEventCreate(&baselineStart));
    checkMusaErrors(musaEventCreate(&baselineStop));
    checkMusaErrors(musaEventCreate(&eventStart));
    checkMusaErrors(musaEventCreate(&eventStop));

    // warm up
    int* flag;
    checkMusaErrors(musaMalloc(&flag, 4 * n));
    checkMusaErrors(musaMemset((void*)flag, 1, 4 * n));
    // tickNum is sensitive: each kernel spins for tickNum clock64() cycles.
    // tickNum=1750  -> per-kernel ~2.9us @600MHz, command dispatch overhead (~1-3us/kernel)
    // tickNum=17500 -> per-kernel ~29us @600MHz, command overhead <5% of kernel time
    const int tickNum = 17500;
    for (int i = 0; i < 100; ++i) {
        delay<<<1, 1, 0, streams[0]>>>(flag + i, tickNum);
        delay<<<1, 1, 0, streams[1]>>>(flag + i, tickNum);
    }
    checkMusaErrors(musaDeviceSynchronize());

    // test if we submit all commands to the same stream
    checkMusaErrors(musaEventRecord(baselineStart, streams[0]));
    for (uint64_t i = 0; i < n; ++i) {
        delay<<<1, 1, 0, streams[0]>>>(flag + i, tickNum);
    }
    checkMusaErrors(musaEventRecord(baselineStop, streams[0]));
    checkMusaErrors(musaEventSynchronize(baselineStop));
    float elapsedMilliseconds = 0.0f;
    checkMusaErrors(musaEventElapsedTime(&elapsedMilliseconds, baselineStart, baselineStop));
    *t1 = elapsedMilliseconds * 1000.f;

    // warm up
    for (int i = 0; i < 100; ++i) {
        delay<<<1, 1, 0, streams[0]>>>(flag + i, tickNum);
        delay<<<1, 1, 0, streams[1]>>>(flag + i, tickNum);
    }
    checkMusaErrors(musaDeviceSynchronize());

    // test if we submit all commands to different streams
    checkMusaErrors(musaEventRecord(eventStart, streams[0]));
    for (uint64_t i = 0; i < n; ++i) {
        if (i != 0) {
            checkMusaErrors(musaStreamWaitEvent(streams[i % 2], events[(i + 1) % 2], 0));
        }
        delay<<<1, 1, 0, streams[i % 2]>>>(flag + i, tickNum);
        checkMusaErrors(musaEventRecord(events[i % 2], streams[i % 2]));
    }
    checkMusaErrors(musaEventRecord(eventStop, streams[(n - 1) % 2]));
    checkMusaErrors(musaEventSynchronize(eventStop));
    checkMusaErrors(musaEventElapsedTime(&elapsedMilliseconds, eventStart, eventStop));
    *t2 = elapsedMilliseconds * 1000.f;

    checkMusaErrors(musaEventDestroy(events[0]));
    checkMusaErrors(musaEventDestroy(events[1]));
    checkMusaErrors(musaEventDestroy(baselineStart));
    checkMusaErrors(musaEventDestroy(baselineStop));
    checkMusaErrors(musaEventDestroy(eventStart));
    checkMusaErrors(musaEventDestroy(eventStop));
    checkMusaErrors(musaStreamDestroy(streams[0]));
    checkMusaErrors(musaStreamDestroy(streams[1]));
    checkMusaErrors(musaFree(flag));

    return 0;
}

BASELINE_F(efficiencyOfSync, syncByEvent, SyncFixture, SamplesCount, IterationsCount) {
    float result1, result2;
    int ans = testNCommands(synchronizedConut, &result1, &result2);
    this->utime1->addValue(result1);                                            // us
    this->utime2->addValue(result2);                                            // us
    if (result2 > result1) {
        this->utime3->addValue((result2 - result1) / float(synchronizedConut - 1)); // us
        totalTime += (result2 - result1);
        totalCnt += (synchronizedConut - 1);
    } else {
        std::cerr << "efficiencyOfSync warning, result2: " << result2
                  << " is not bigger than result1: " << result1
                  << ", skip derived synchronization overhead" << std::endl;
    }
}
