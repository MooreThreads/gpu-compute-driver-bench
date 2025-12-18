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
            ev.User_Data.push_back(4);
            ev.User_Data.push_back(4);
            ev.Value = ev.User_Data[0] * ev.User_Data[1] * ev.User_Data[2] * sizeof(uint32_t);
            problemSpace.push_back(ev);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        width  = experimentValue.User_Data[0];
        height = experimentValue.User_Data[1];
        depth  = experimentValue.User_Data[2];
    }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->uband_d2d, this->uband_d2h, this->uband_h2d};
    }

    int64_t width;
    int64_t height;
    int64_t depth;
    musaEvent_t start;
    musaEvent_t stop;
    std::shared_ptr<UDMBandWidth> uband_h2d{new UDMBandWidth("*H2D")};
    std::shared_ptr<UDMBandWidth> uband_d2d{new UDMBandWidth("*D2D")};
    std::shared_ptr<UDMBandWidth> uband_d2h{new UDMBandWidth("*D2H")};
};

static const int SamplesCount    = 3;
static const int IterationsCount = 1;

BASELINE_F(musaCopy, copyRate3D, CopyFixture, SamplesCount, IterationsCount) {
    float milliseconds = 0.f;
    CPerfCounter timer;
    MUresult status          = MUSA_SUCCESS;
    musaError_t err          = musaSuccess;
    const size_t numElements = width * height * depth;
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

    checkMusaErrors(musaSetDevice(0));

    struct musaExtent extent = {width * sizeof(int), size_t(height), size_t(depth)};
    struct musaPitchedPtr pitchedDevPtr_B;
    checkMusaErrors(musaMalloc3D(&pitchedDevPtr_B, extent));
    struct musaPitchedPtr pitchedDevPtr_C;
    checkMusaErrors(musaMalloc3D(&pitchedDevPtr_C, extent));

    memset(hB, 0, bytes);
    memset(hC, 0, bytes);

    MUSA_MEMCPY3D copy3D;
    copy3D.srcXInBytes   = 0;
    copy3D.srcY          = 0;
    copy3D.srcZ          = 0;
    copy3D.srcLOD        = 0;
    copy3D.srcMemoryType = MU_MEMORYTYPE_HOST;
    copy3D.srcHost       = hA;
    copy3D.srcDevice     = 0;
    copy3D.srcArray      = 0;
    copy3D.reserved0     = nullptr;
    copy3D.srcPitch      = width * sizeof(int);
    copy3D.srcHeight     = height;

    copy3D.dstXInBytes   = 0;
    copy3D.dstY          = 0;
    copy3D.dstZ          = 0;
    copy3D.dstLOD        = 0;
    copy3D.dstMemoryType = MU_MEMORYTYPE_DEVICE;
    copy3D.dstHost       = nullptr;
    copy3D.dstDevice     = reinterpret_cast<MUdeviceptr>(pitchedDevPtr_B.ptr);
    copy3D.dstArray      = 0;
    copy3D.reserved1     = nullptr;
    copy3D.dstPitch      = pitchedDevPtr_B.pitch;
    copy3D.dstHeight     = height;

    copy3D.WidthInBytes = width * sizeof(int);
    copy3D.Height       = height;
    copy3D.Depth        = depth;

    timer.Restart();
    checkMuErrors(muMemcpy3DAsync(&copy3D, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_h2d->addValue(static_cast<double>(bytes) / (milliseconds * 1000));

    copy3D.srcMemoryType = MU_MEMORYTYPE_DEVICE;
    copy3D.srcPitch      = pitchedDevPtr_B.pitch;
    copy3D.srcHost       = nullptr;
    copy3D.srcDevice     = reinterpret_cast<MUdeviceptr>(pitchedDevPtr_B.ptr);
    copy3D.dstMemoryType = MU_MEMORYTYPE_HOST;
    copy3D.dstPitch      = width * sizeof(int);
    copy3D.dstHost       = hB;
    copy3D.dstDevice     = 0;

    checkMuErrors(muMemcpy3D(&copy3D));

    copy3D.srcMemoryType = MU_MEMORYTYPE_DEVICE;
    copy3D.srcPitch      = pitchedDevPtr_B.pitch;
    copy3D.srcHost       = nullptr;
    copy3D.srcDevice     = reinterpret_cast<MUdeviceptr>(pitchedDevPtr_B.ptr);
    copy3D.dstMemoryType = MU_MEMORYTYPE_DEVICE;
    copy3D.dstPitch      = pitchedDevPtr_C.pitch;
    copy3D.dstHost       = nullptr;
    copy3D.dstDevice     = reinterpret_cast<MUdeviceptr>(pitchedDevPtr_C.ptr);

    timer.Restart();
    checkMuErrors(muMemcpy3DAsync(&copy3D, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_d2d->addValue(2 * static_cast<double>(bytes) / (milliseconds * 1000));

    copy3D.srcMemoryType = MU_MEMORYTYPE_DEVICE;
    copy3D.srcPitch      = pitchedDevPtr_C.pitch;
    copy3D.srcHost       = nullptr;
    copy3D.srcDevice     = reinterpret_cast<MUdeviceptr>(pitchedDevPtr_C.ptr);
    copy3D.dstMemoryType = MU_MEMORYTYPE_HOST;
    copy3D.dstPitch      = width * sizeof(int);
    copy3D.dstHost       = hC;
    copy3D.dstDevice     = 0;

    timer.Restart();
    checkMuErrors(muMemcpy3DAsync(&copy3D, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_d2h->addValue(static_cast<double>(bytes) / (milliseconds * 1000));

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
    checkMusaErrors(musaFree(pitchedDevPtr_B.ptr));
    checkMusaErrors(musaFree(pitchedDevPtr_C.ptr));
}