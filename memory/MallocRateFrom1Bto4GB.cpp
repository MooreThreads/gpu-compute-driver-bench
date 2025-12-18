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
#include <string.h>
#include <unistd.h>

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
    Run(argc, argv);
    return 0;
}

// test time from event, we can define some other measure method for developers
class MallocFixture : public TestFixture {
public:
    MallocFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t elements = 1LL << 0; elements <= (1LL << 32); elements *= 2) {
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
        return {this->uband1, this->utime1, this->uband2, this->utime2};
    }
    int64_t mallocSize;
    musaEvent_t start;
    musaEvent_t stop;
    void* deviceMemory;
    std::shared_ptr<UDMBandWidth> uband1{new UDMBandWidth("*Bmall")};
    std::shared_ptr<UDMGPUTime> utime1{new UDMGPUTime("tmall")};
    std::shared_ptr<UDMBandWidth> uband2{new UDMBandWidth("*Mfree")};
    std::shared_ptr<UDMGPUTime> utime2{new UDMGPUTime("tfree")};
};

static const int SamplesCount    = 3;
static const int IterationsCount = 1;
BASELINE_F(musa_malloc, normal_mem, MallocFixture, SamplesCount, IterationsCount) {
    // 01. warm up
    void* dTemp;
    checkMusaErrors(musaMalloc(&dTemp, 4));
    checkMusaErrors(musaFree(dTemp));

    // 02. test musaMalloc
    float milliseconds = 0.f;
    CPerfCounter timer;
    timer.Restart();
    checkMusaErrors(musaMalloc(&deviceMemory, this->mallocSize));
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;

    this->utime1->addValue(milliseconds);
    this->uband1->addValue(static_cast<double>(mallocSize) / (milliseconds * 1000));

    // 03. test musaFree
    timer.Restart();
    checkMusaErrors(musaFree(deviceMemory));
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;

    this->utime2->addValue(milliseconds);
    this->uband2->addValue(static_cast<double>(mallocSize) / (milliseconds * 1000));
}

BENCHMARK_F(musa_malloc, pinned_mem, MallocFixture, SamplesCount, IterationsCount) {
    // 01. warm up
    void* dTemp;
    checkMusaErrors(musaHostAlloc(&dTemp, 4, musaHostAllocDefault));
    checkMusaErrors(musaFreeHost(dTemp));

    // 02. test musaHostAlloc
    float milliseconds = 0.f;
    CPerfCounter timer;
    timer.Restart();
    checkMusaErrors(musaHostAlloc(&deviceMemory, this->mallocSize, musaHostAllocDefault));
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->utime1->addValue(milliseconds);
    this->uband1->addValue(static_cast<double>(mallocSize) / (milliseconds * 1000));

    // 03. test musaFreeHost
    timer.Restart();
    checkMusaErrors(musaFreeHost(deviceMemory));
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->utime2->addValue(milliseconds);
    this->uband2->addValue(static_cast<double>(mallocSize) / (milliseconds * 1000));
}
