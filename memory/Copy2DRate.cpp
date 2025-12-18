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

class CopyFixture : public TestFixture {
public:
    CopyFixture() {}

    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t i = 0; i <= 26; ++i) {
            ExperimentValue ev;
            ev.User_Data.push_back(1 << i);
            ev.User_Data.push_back(16);
            ev.Value = ev.User_Data[0] * ev.User_Data[1] * sizeof(uint32_t);
            problemSpace.push_back(ev);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        width  = experimentValue.User_Data[0];
        height = experimentValue.User_Data[1];
    }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override { checkMuErrors(muInit(0)); }
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->uband_d2d, this->uband_d2h, this->uband_h2d};
    }

    int64_t width;
    int64_t height;
    musaEvent_t start;
    musaEvent_t stop;
    std::shared_ptr<UDMBandWidth> uband_h2d{new UDMBandWidth("*H2D")};
    std::shared_ptr<UDMBandWidth> uband_d2d{new UDMBandWidth("*D2D")};
    std::shared_ptr<UDMBandWidth> uband_d2h{new UDMBandWidth("*D2H")};
};

static const int SamplesCount    = 3;
static const int IterationsCount = 1;

BASELINE_F(musaCopy, copyRate2D, CopyFixture, SamplesCount, IterationsCount) {
    // 01. prepare data
    float milliseconds = 0.f;
    CPerfCounter timer;
    MUresult status = MUSA_SUCCESS;
    musaError_t err = musaSuccess;
    MUdeviceptr d_B, d_C;
    const size_t numElements = width * height;
    const size_t bytes       = numElements * sizeof(int);

    int* hA = static_cast<int*>(malloc(bytes));
    if (hA == nullptr) {
        printf("host A malloc FAILED!\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < numElements; ++i) {
        hA[i] = static_cast<int>(i);
    }

    int* hB = static_cast<int*>(malloc(bytes));
    if (hB == nullptr) {
        printf("host B malloc FAILED!\n");
        exit(EXIT_FAILURE);
    }

    int* hC = static_cast<int*>(malloc(bytes));
    if (hC == nullptr) {
        printf("host C malloc FAILED!\n");
        exit(EXIT_FAILURE);
    }

    size_t pitch;
    void* dTemp;
    checkMusaErrors(musaMalloc(&dTemp, 4));
    checkMusaErrors(musaFree(dTemp));
    checkMuErrors(muMemAllocPitch(&d_B, &pitch, width * sizeof(int), height, 4));
    checkMuErrors(muMemAllocPitch(&d_C, &pitch, width * sizeof(int), height, 4));

    memset(hB, 0, bytes);
    memset(hC, 0, bytes);

    // 02. test H2D
    MUSA_MEMCPY2D copy2D;
    copy2D.srcXInBytes   = 0;
    copy2D.srcY          = 0;
    copy2D.srcMemoryType = MU_MEMORYTYPE_HOST;
    copy2D.srcHost       = hA;
    copy2D.srcDevice     = 0;
    copy2D.srcArray      = 0;
    copy2D.srcPitch      = width * sizeof(int);

    copy2D.dstXInBytes   = 0;
    copy2D.dstY          = 0;
    copy2D.dstMemoryType = MU_MEMORYTYPE_DEVICE;
    copy2D.dstHost       = nullptr;
    copy2D.dstDevice     = d_B;
    copy2D.dstArray      = 0;
    copy2D.dstPitch      = pitch;

    copy2D.WidthInBytes = width * sizeof(int);
    copy2D.Height       = height;

    timer.Restart();
    checkMuErrors(muMemcpy2DAsync(&copy2D, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_h2d->addValue(static_cast<double>(bytes) / (milliseconds * 1000));

    // 03. test D2D
    copy2D.srcMemoryType = MU_MEMORYTYPE_DEVICE;
    copy2D.srcPitch      = pitch;
    copy2D.srcHost       = nullptr;
    copy2D.srcDevice     = d_B;
    copy2D.dstMemoryType = MU_MEMORYTYPE_HOST;
    copy2D.dstPitch      = width * sizeof(int);
    copy2D.dstHost       = hB;
    copy2D.dstDevice     = 0;
    checkMuErrors(muMemcpy2D(&copy2D));
    copy2D.srcMemoryType = MU_MEMORYTYPE_DEVICE;
    copy2D.srcPitch      = pitch;
    copy2D.srcHost       = nullptr;
    copy2D.srcDevice     = d_B;
    copy2D.dstMemoryType = MU_MEMORYTYPE_DEVICE;
    copy2D.dstPitch      = pitch;
    copy2D.dstHost       = nullptr;
    copy2D.dstDevice     = d_C;

    timer.Restart();
    checkMuErrors(muMemcpy2DAsync(&copy2D, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_d2d->addValue(2 * static_cast<double>(bytes) / (milliseconds * 1000));

    // 04. test D2H
    copy2D.srcMemoryType = MU_MEMORYTYPE_DEVICE;
    copy2D.srcPitch      = pitch;
    copy2D.srcHost       = nullptr;
    copy2D.srcDevice     = d_C;
    copy2D.dstMemoryType = MU_MEMORYTYPE_HOST;
    copy2D.dstPitch      = width * sizeof(int);
    copy2D.dstHost       = hC;
    copy2D.dstDevice     = 0;

    timer.Restart();
    checkMuErrors(muMemcpy2DAsync(&copy2D, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_d2h->addValue(static_cast<double>(bytes) / (milliseconds * 1000));

    // 05. check result
    checkMusaErrors(musaDeviceSynchronize());
    size_t errorCnt = 0;
    if (memcmp(hB, hA, bytes)) {
        for (size_t i = 0; i < numElements; ++i) {
            if (hB[i] != hA[i]) {
                errorCnt++;
                printf("Result check FAILED at hB[%d]=%d\n", static_cast<int>(i), hB[i]);
                break;
            }
        }
    }

    if (memcmp(hC, hA, bytes)) {
        for (size_t i = 0; i < numElements; ++i) {
            if (hC[i] != hA[i]) {
                errorCnt++;
                printf("Result check FAILED at hC[%d]=%d\n", static_cast<int>(i), hC[i]);
                break;
            }
        }
    }

    if (errorCnt != 0) {
        printf("Failed in result verification!\n");
        exit(EXIT_FAILURE);
    }
    free(hA);
    free(hB);
    free(hC);
    checkMuErrors(muMemFree(d_B));
    checkMuErrors(muMemFree(d_C));
}