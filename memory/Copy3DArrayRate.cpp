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
        for (int64_t i = 0; i <= 14; ++i) {
            ExperimentValue ev;
            ev.User_Data.push_back(1 << i);
            ev.User_Data.push_back(4);
            ev.User_Data.push_back(4);
            ev.Value = ev.User_Data[0] * ev.User_Data[1] * ev.User_Data[2] * sizeof(int);
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
        return {this->uband_h2t, this->uband_t2h, this->uband_t2t, this->uband_s2t, this->uband_t2s};
    }
    int64_t width;
    int64_t height;
    int64_t depth;
    musaEvent_t start;
    musaEvent_t stop;
    std::shared_ptr<UDMBandWidth> uband_h2t{new UDMBandWidth("*H2T")};
    std::shared_ptr<UDMBandWidth> uband_t2h{new UDMBandWidth("*T2H")};
    std::shared_ptr<UDMBandWidth> uband_t2t{new UDMBandWidth("*T2T")};
    std::shared_ptr<UDMBandWidth> uband_s2t{new UDMBandWidth("*S2T")};
    std::shared_ptr<UDMBandWidth> uband_t2s{new UDMBandWidth("*T2S")};
};

static const int SamplesCount    = 1;
static const int IterationsCount = 10;

BASELINE_F(musaCopy, 3DArray, CopyFixture, SamplesCount, IterationsCount) {
    float milliseconds = 0.f;
    CPerfCounter timer;
    MUresult status          = MUSA_SUCCESS;
    musaError_t err          = musaSuccess;
    const size_t numElements = width * height * depth;
    const size_t bytes       = numElements * sizeof(int);
    int* hA                  = static_cast<int*>(malloc(bytes));
    if (hA == nullptr) {
        printf("host A malloc FAILED!\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < numElements; ++i) {
        hA[i] = i;
    }

    int* hB = static_cast<int*>(malloc(bytes));
    if (hB == nullptr) {
        printf("host B malloc FAILED!\n");
        exit(EXIT_FAILURE);
    }
    memset(hB, 0, bytes);
    struct musaExtent extent = {width * sizeof(int), size_t(height), size_t(depth)};
    struct musaPitchedPtr pitchedDevPtr_A;
    checkMusaErrors(musaMalloc3D(&pitchedDevPtr_A, extent));

    checkMusaErrors(musaSetDevice(0));
    MUarray array_3d_a;
    MUarray array_3d_b;
    MUarray array_3d_c;
    MUSA_ARRAY3D_DESCRIPTOR desc = {size_t(width), size_t(height), size_t(depth), MU_AD_FORMAT_SIGNED_INT32, 1, 0};
    checkMuErrors(muArray3DCreate(&array_3d_a, &desc));
    checkMuErrors(muArray3DCreate(&array_3d_b, &desc));
    checkMuErrors(muArray3DCreate(&array_3d_c, &desc));

    /**********************Host to Array*****************************************************/
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
    copy3D.dstMemoryType = MU_MEMORYTYPE_ARRAY;
    copy3D.dstHost       = nullptr;
    copy3D.dstDevice     = 0;
    copy3D.dstArray      = array_3d_a;
    copy3D.reserved1     = nullptr;
    copy3D.dstPitch      = 0;
    copy3D.dstHeight     = height;

    copy3D.WidthInBytes = width * sizeof(int);
    copy3D.Height       = height;
    copy3D.Depth        = depth;

    // Host2Array warm up
    checkMuErrors(muMemcpy3D(&copy3D));
    checkMusaErrors(musaDeviceSynchronize());

    // Host2Array timing
    timer.Restart();
    checkMuErrors(muMemcpy3DAsync(&copy3D, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_h2t->addValue(static_cast<double>(bytes) / (milliseconds * 1000));

    /**********************Array to Array*****************************************************/
    copy3D.srcXInBytes   = 0;
    copy3D.srcY          = 0;
    copy3D.srcZ          = 0;
    copy3D.srcLOD        = 0;
    copy3D.srcMemoryType = MU_MEMORYTYPE_ARRAY;
    copy3D.srcHost       = nullptr;
    copy3D.srcDevice     = 0;
    copy3D.srcArray      = array_3d_a;
    copy3D.reserved0     = nullptr;
    copy3D.srcPitch      = 0;
    copy3D.srcHeight     = height;

    copy3D.dstXInBytes   = 0;
    copy3D.dstY          = 0;
    copy3D.dstZ          = 0;
    copy3D.dstLOD        = 0;
    copy3D.dstMemoryType = MU_MEMORYTYPE_ARRAY;
    copy3D.dstHost       = nullptr;
    copy3D.dstDevice     = 0;
    copy3D.dstArray      = array_3d_b;
    copy3D.reserved1     = nullptr;
    copy3D.dstPitch      = 0;
    copy3D.dstHeight     = height;

    copy3D.WidthInBytes = width * sizeof(int);
    copy3D.Height       = height;
    copy3D.Depth        = depth;

    // Array2Array warmup
    checkMuErrors(muMemcpy3D(&copy3D));
    checkMusaErrors(musaDeviceSynchronize());

    // Array2Array timing
    timer.Restart();
    checkMuErrors(muMemcpy3DAsync(&copy3D, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_t2t->addValue(2.0f * static_cast<double>(bytes) / (milliseconds * 1000));

    /**********************Array to Host*****************************************************/
    copy3D.srcXInBytes   = 0;
    copy3D.srcY          = 0;
    copy3D.srcZ          = 0;
    copy3D.srcLOD        = 0;
    copy3D.srcMemoryType = MU_MEMORYTYPE_ARRAY;
    copy3D.srcHost       = nullptr;
    copy3D.srcDevice     = 0;
    copy3D.srcArray      = array_3d_b;
    copy3D.reserved0     = nullptr;
    copy3D.srcPitch      = 0;
    copy3D.srcHeight     = height;

    copy3D.dstXInBytes   = 0;
    copy3D.dstY          = 0;
    copy3D.dstZ          = 0;
    copy3D.dstLOD        = 0;
    copy3D.dstMemoryType = MU_MEMORYTYPE_HOST;
    copy3D.dstHost       = hB;
    copy3D.dstDevice     = 0;
    copy3D.dstArray      = 0;
    copy3D.reserved1     = nullptr;
    copy3D.dstPitch      = width * sizeof(int);
    copy3D.dstHeight     = height;

    copy3D.WidthInBytes = width * sizeof(int);
    copy3D.Height       = height;
    copy3D.Depth        = depth;

    // Array2Host warmup
    checkMuErrors(muMemcpy3D(&copy3D));
    checkMusaErrors(musaDeviceSynchronize());

    // Array2Host timing
    timer.Restart();
    checkMuErrors(muMemcpy3DAsync(&copy3D, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_t2h->addValue(static_cast<double>(bytes) / (milliseconds * 1000));

    size_t errorCnt = 0;
    if (memcmp(hB, hA, bytes)) {
        for (size_t i = 0; i < numElements; ++i) {
            if (hB[i] != hA[i]) {
                errorCnt++;
                printf("Result check FAILED at hB[%lu]=%d\n", i, hB[i]);
                break;
            }
        }
    }
    if (errorCnt != 0) {
        printf("Failed in result verification!\n");
        exit(EXIT_FAILURE);
    }
    memset(hB, 0, bytes);

    /**********************Array to Device*****************************************************/
    copy3D.srcXInBytes   = 0;
    copy3D.srcY          = 0;
    copy3D.srcZ          = 0;
    copy3D.srcLOD        = 0;
    copy3D.srcMemoryType = MU_MEMORYTYPE_ARRAY;
    copy3D.srcHost       = nullptr;
    copy3D.srcDevice     = 0;
    copy3D.srcArray      = array_3d_a;
    copy3D.reserved0     = nullptr;
    copy3D.srcPitch      = 0;
    copy3D.srcHeight     = height;

    copy3D.dstXInBytes   = 0;
    copy3D.dstY          = 0;
    copy3D.dstZ          = 0;
    copy3D.dstLOD        = 0;
    copy3D.dstMemoryType = MU_MEMORYTYPE_DEVICE;
    copy3D.dstHost       = nullptr;
    copy3D.dstDevice     = reinterpret_cast<MUdeviceptr>(pitchedDevPtr_A.ptr);
    copy3D.dstArray      = 0;
    copy3D.reserved1     = nullptr;
    copy3D.dstPitch      = width * sizeof(int);
    copy3D.dstHeight     = height;

    copy3D.WidthInBytes = width * sizeof(int);
    copy3D.Height       = height;
    copy3D.Depth        = depth;

    // Array2Device warmup
    checkMuErrors(muMemcpy3D(&copy3D));
    checkMusaErrors(musaDeviceSynchronize());

    timer.Restart();
    checkMuErrors(muMemcpy3DAsync(&copy3D, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    checkMusaErrors(musaDeviceSynchronize());
    this->uband_t2s->addValue(static_cast<double>(bytes) / (milliseconds * 1000));

    /**********************Device to Array*****************************************************/
    copy3D.srcXInBytes   = 0;
    copy3D.srcY          = 0;
    copy3D.srcZ          = 0;
    copy3D.srcLOD        = 0;
    copy3D.srcMemoryType = MU_MEMORYTYPE_DEVICE;
    copy3D.srcHost       = nullptr;
    copy3D.srcDevice     = reinterpret_cast<MUdeviceptr>(pitchedDevPtr_A.ptr);
    copy3D.srcArray      = 0;
    copy3D.reserved0     = nullptr;
    copy3D.srcPitch      = width * sizeof(int);
    copy3D.srcHeight     = height;

    copy3D.dstXInBytes   = 0;
    copy3D.dstY          = 0;
    copy3D.dstZ          = 0;
    copy3D.dstLOD        = 0;
    copy3D.dstMemoryType = MU_MEMORYTYPE_ARRAY;
    copy3D.dstHost       = nullptr;
    copy3D.dstDevice     = 0;
    copy3D.dstArray      = array_3d_c;
    copy3D.reserved1     = nullptr;
    copy3D.dstPitch      = 0;
    copy3D.dstHeight     = height;

    copy3D.WidthInBytes = width * sizeof(int);
    copy3D.Height       = height;
    copy3D.Depth        = depth;

    // Device2Array warmup
    checkMuErrors(muMemcpy3D(&copy3D));
    checkMusaErrors(musaDeviceSynchronize());

    // Device2Array timing
    timer.Restart();
    checkMuErrors(muMemcpy3DAsync(&copy3D, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_s2t->addValue(static_cast<double>(bytes) / (milliseconds * 1000));

    /**********************Array to Host*****************************************************/
    copy3D.srcXInBytes   = 0;
    copy3D.srcY          = 0;
    copy3D.srcZ          = 0;
    copy3D.srcLOD        = 0;
    copy3D.srcMemoryType = MU_MEMORYTYPE_ARRAY;
    copy3D.srcHost       = nullptr;
    copy3D.srcDevice     = 0;
    copy3D.srcArray      = array_3d_c;
    copy3D.reserved0     = nullptr;
    copy3D.srcPitch      = 0;
    copy3D.srcHeight     = height;

    copy3D.dstXInBytes   = 0;
    copy3D.dstY          = 0;
    copy3D.dstZ          = 0;
    copy3D.dstLOD        = 0;
    copy3D.dstMemoryType = MU_MEMORYTYPE_HOST;
    copy3D.dstHost       = hB;
    copy3D.dstDevice     = 0;
    copy3D.dstArray      = 0;
    copy3D.reserved1     = nullptr;
    copy3D.dstPitch      = width * sizeof(int);
    copy3D.dstHeight     = height;

    copy3D.WidthInBytes = width * sizeof(int);
    copy3D.Height       = height;
    copy3D.Depth        = depth;

    // Array2Host
    checkMuErrors(muMemcpy3D(&copy3D));
    checkMusaErrors(musaDeviceSynchronize());
    errorCnt = 0;
    if (memcmp(hB, hA, bytes)) {
        for (size_t i = 0; i < numElements; ++i) {
            if (hB[i] != hA[i]) {
                errorCnt++;
                printf("Result check FAILED at hB[%lu]=%d\n", i, hB[i]);
                exit(EXIT_FAILURE);
            }
        }
    }

    free(hA);
    free(hB);
    status = muArrayDestroy(array_3d_a);
    status = muArrayDestroy(array_3d_b);
    status = muArrayDestroy(array_3d_c);
    checkMusaErrors(musaFree(pitchedDevPtr_A.ptr));
}