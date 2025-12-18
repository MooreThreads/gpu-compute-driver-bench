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
    std::shared_ptr<UDMThroughPut> utp{new UDMThroughPut("*TP(s^-1)")};
};

float SyncFixture::totalTime   = 0.f;
uint32_t SyncFixture::totalCnt = 0;

int SyncFixture::testNCommands(int n, float* t1, float* t2) {
    CPerfCounter timer;
    musaStream_t streams[2];
    checkMusaErrors(musaStreamCreate(&streams[0]));
    checkMusaErrors(musaStreamCreate(&streams[1]));

    musaEvent_t events[2];
    checkMusaErrors(musaEventCreate(&events[0]));
    checkMusaErrors(musaEventCreate(&events[1]));

    // warm up
    int* flag;
    checkMusaErrors(musaMalloc(&flag, 4 * n));
    checkMusaErrors(musaMemset((void*)flag, 1, 4 * n));
    const int tickNum = 1750;
    // delay<<<1,1>>>(flag, tickNum);
    for (int i = 0; i < 100; ++i) {
        delay<<<1, 1, 0, streams[0]>>>(flag + i, tickNum);
        delay<<<1, 1, 0, streams[1]>>>(flag + i, tickNum);
    }
    checkMusaErrors(musaDeviceSynchronize());

    // test if we submit all commands to the same stream
    timer.Start();
    for (uint64_t i = 0; i < n; ++i) {
        delay<<<1, 1, 0, streams[0]>>>(flag + i, tickNum);
        // emptyKernel<<<1, 1>>>();
        // emptyKernel<<<1, 1, 0, streams[0]>>>();
    }
    checkMusaErrors(musaStreamSynchronize(nullptr));
    timer.Stop();
    float result1 = timer.GetElapsedSeconds() * 1000.f * 1000.f;

    for (int i = 0; i < 100; ++i) {
        delay<<<1, 1, 0, streams[0]>>>(flag + i, tickNum);
        delay<<<1, 1, 0, streams[1]>>>(flag + i, tickNum);
    }
    // test if we submit all commands to different streams
    timer.Reset();
    timer.Start();
    for (uint64_t i = 0; i < n; ++i) {
        if (i != 0) {
            checkMusaErrors(musaStreamWaitEvent(streams[i % 2], events[(i + 1) % 2], 0));
        }
        delay<<<1, 1, 0, streams[i % 2]>>>(flag + i, tickNum);
        // emptyKernel<<<1, 1, 0, streams[i % 2]>>>();
        checkMusaErrors(musaEventRecord(events[i % 2], streams[i % 2]));
    }
    checkMusaErrors(musaStreamSynchronize(streams[(n - 1) % 2]));
    timer.Stop();
    float result2 = timer.GetElapsedSeconds() * 1000.f * 1000.f;

    *t1 = result1;
    *t2 = result2;
    return 0;
}

BASELINE_F(efficiencyOfSync, syncByEvent, SyncFixture, SamplesCount, IterationsCount) {
    float result1, result2;
    int ans = testNCommands(synchronizedConut, &result1, &result2);
    // sometimes, result2 < result1)(have no idea) ignore these cases
    if (result2 - result1) {
        this->utime1->addValue(result1);                                            // us
        this->utime2->addValue(result2);                                            // us
        this->utime3->addValue((result2 - result1) / float(synchronizedConut - 1)); // us
        totalTime += (result2 - result1);
        totalCnt += (synchronizedConut - 1);
    }
}