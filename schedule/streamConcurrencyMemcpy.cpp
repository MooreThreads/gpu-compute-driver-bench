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
static const int IterationsCount = 5;
static const int MaxStreamNum    = 16;

int main(int argc, char** argv) {
    int deviceCount;
    checkMusaErrors(musaGetDeviceCount(&deviceCount));
    musaDeviceProp prop;
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    Printer::get().TableSetPbName("streamsCnt");
    Run(argc, argv);
    return 0;
}

class LanchFixture : public TestFixture {
public:
    LanchFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t i = 0; i <= std::log2(MaxStreamNum); ++i) {
            problemSpace.push_back(1 << i);
        }
        return problemSpace;
    }
    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        streamNum    = experimentValue.Value;
        elementBytes = 1024 * 1024 * 1024;
        checkMuErrors(muInit(0));
        MUcontext ctx = nullptr;
        checkMuErrors(muCtxCreate(&ctx, 0, 0));
        MUdevice device = -1;
        checkMuErrors(muCtxGetDevice(&device));
        unsigned flags = -1;
        for (uint32_t i = 0; i < streamNum; i++) {
            checkMuErrors(muStreamCreate(&stream[i], 0));
        }
    }

    void tearDown() override {
        if (streamNum == 1) {
            basicTime = this->utime1->getMean();
        }
        this->uratio->addValue(basicTime / this->utime1->getMean());
        for (uint32_t i = 0; i < streamNum; i++) {
            checkMuErrors(muStreamDestroy(stream[i]));
        }
        muCtxDestroy(ctx);
    }

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->utime1, this->uband1, this->uratio};
    }
    float memcpyByStreams();
    uint32_t elementBytes;
    uint32_t streamNum;
    MUcontext ctx       = nullptr;
    MUstream stream[16] = {0};
    std::shared_ptr<UDMGPUTime> utime1{new UDMGPUTime("Cpy(ms)")};
    std::shared_ptr<UDMBandWidth> uband1{new UDMBandWidth("Band")};
    std::shared_ptr<UDMRatio> uratio{new UDMRatio("*Ratio")};
    static float basicTime;
};

float LanchFixture::basicTime = 0.f;

float LanchFixture::memcpyByStreams() {
    int* ha        = nullptr;
    int* hb        = nullptr;
    MUdeviceptr da = 0;
    MUdeviceptr db = 0;
    ha             = reinterpret_cast<int*>(malloc(elementBytes));
    hb             = reinterpret_cast<int*>(malloc(elementBytes));
    checkMuErrors(muMemAlloc(&da, elementBytes));
    checkMuErrors(muMemAlloc(&db, elementBytes));
    for (uint32_t i = 0; i < elementBytes / sizeof(int); ++i) {
        ha[i] = i;
    }
    checkMuErrors(muMemcpyHtoD(da, ha, elementBytes));
    MUevent startEvent, stopEvent;
    checkMuErrors(muEventCreate(&startEvent, 0));
    checkMuErrors(muEventCreate(&stopEvent, 0));
    checkMuErrors(muEventRecord(startEvent, 0));
    for (uint32_t i = 0; i < streamNum; ++i) {
        if (streamNum == 1) {
            checkMuErrors(muMemcpyDtoDAsync(db, da, elementBytes, nullptr));
        } else {
            checkMuErrors(muMemcpyDtoDAsync(db + i * (elementBytes / streamNum), da + i * (elementBytes / streamNum),
                elementBytes / streamNum, stream[i % streamNum]));
        }
    }
    checkMuErrors(muEventRecord(stopEvent, 0));
    checkMuErrors(muEventSynchronize(stopEvent));
    float timeElapsed = 0.0f;
    checkMuErrors(muEventElapsedTime(&timeElapsed, startEvent, stopEvent));
    checkMuErrors(muMemcpyDtoH(hb, db, elementBytes));
    for (uint32_t i = 0; i < elementBytes / sizeof(int); ++i) {
        if (hb[i] != ha[i]) {
            std::cout << "Error at " << i << " " << hb[i] << " " << ha[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    checkMuErrors(muEventDestroy(startEvent));
    checkMuErrors(muEventDestroy(stopEvent));
    checkMuErrors(muMemFree(da));
    checkMuErrors(muMemFree(db));
    free(ha);
    free(hb);
    return timeElapsed;
}

BASELINE_F(Memcpy, MulStreams, LanchFixture, SamplesCount, IterationsCount) {
    float time = memcpyByStreams();
    this->utime1->addValue(time);
    this->uband1->addValue(2 * 1024 / (time / 1000));
}