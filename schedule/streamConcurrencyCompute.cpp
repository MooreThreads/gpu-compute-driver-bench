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
#define KernelsNum 1024

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
        for (int64_t i = 0; i <= std::log2(64); ++i) {
            problemSpace.push_back(1 << i);
        }
        return problemSpace;
    }
    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        streamNum = experimentValue.Value;
#ifdef TEST_ON_NVIDIA
        codeObjectFile = std::string("./cdm_parallelism.ptx");
#else
        codeObjectFile = std::string("./cdm_parallelism.elf");
#endif
        functionName = std::string("_Z6kernelfPfS_");
        elementNum   = 16;
        checkMuErrors(muInit(0));
        MUcontext ctx = nullptr;
        checkMuErrors(muCtxCreate(&ctx, 0, 0));
        MUdevice device = -1;
        checkMuErrors(muCtxGetDevice(&device));
        unsigned flags = -1;
        checkMuErrors(muCtxGetFlags(&flags));
        MUfunc_cache cacheConfig = MU_FUNC_CACHE_PREFER_EQUAL;
        checkMuErrors(muCtxGetCacheConfig(&cacheConfig));
        checkMuErrors(muModuleLoad(&module, codeObjectFile.c_str()));
        checkMuErrors(muModuleGetFunction(&function, module, functionName.c_str()));
        for (uint32_t i = 0; i < streamNum; i++) {
            checkMuErrors(muStreamCreate(&stream[i], 0));
        }
    }

    void tearDown() override {
        if (streamNum == 1) {
            basicTime = this->utime1->getMean();
        }
        this->uratio->addValue(basicTime / this->utime1->getMean());
        checkMuErrors(muModuleUnload(module));
        for (uint32_t i = 0; i < streamNum; i++) {
            checkMuErrors(muStreamDestroy(stream[i]));
        }
        muCtxDestroy(ctx);
    }

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {
    }
    void onExperimentEnd() override {
    }

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->utime1, this->utp1, this->uratio};
    }
    float launchKernelByStreams();
    std::string codeObjectFile;
    std::string functionName;
    uint32_t elementNum;

    uint32_t streamNum;
    MUcontext ctx = nullptr;
    MUmodule module;
    MUfunction function;
    MUstream stream[KernelsNum] = {0};
    std::shared_ptr<UDMGPUTime> utime1{new UDMGPUTime("Ker(ms)", StatsView::MIN | StatsView::MAX)};
    std::shared_ptr<UDMThroughPut> utp1{new UDMThroughPut("TP(s^-1)")};
    std::shared_ptr<UDMRatio> uratio{new UDMRatio("*Ratio")};
    static float basicTime;
};

float LanchFixture::basicTime = 0.f;

float LanchFixture::launchKernelByStreams() {
    float *hA = nullptr, *hB = nullptr;
    MUdeviceptr dA = 0, dB = 0;
    const size_t sizeBytes = KernelsNum * sizeof(float);
    hA                     = reinterpret_cast<float*>(malloc(sizeBytes));
    memset(hA, 0, sizeBytes);
    hB = reinterpret_cast<float*>(malloc(sizeBytes));
    assert(hA != nullptr && hB != nullptr && "host malloc failed!");
    size_t freeMemSize = 0, totalMemSize = 0;
    checkMuErrors(muMemGetInfo(&freeMemSize, &totalMemSize));
    checkMuErrors(muMemAlloc(&dA, sizeBytes));
    checkMuErrors(muMemsetD8(dA, 0, sizeBytes));
    checkMuErrors(muMemAlloc(&dB, sizeBytes));
    assert(dA != 0 && dB != 0 && "device malloc failed!");
    checkMuErrors(muMemGetInfo(&freeMemSize, &totalMemSize));
    for (uint32_t i = 0; i < KernelsNum; ++i) {
        hB[i] = 2 * i;
    }
    MUevent startEvent, stopEvent;
    checkMuErrors(muEventCreate(&startEvent, 0));
    checkMuErrors(muEventCreate(&stopEvent, 0));
    checkMuErrors(muMemcpyHtoD(dB, hB, sizeBytes));
    int threadsPerBlock = 1;
    int blocksPerGrid   = 8;
    struct KernArg {
        float a;
        void *A, *B;
    };

    // warm up
    KernArg kernArg    = {0.123, reinterpret_cast<void*>(dB), reinterpret_cast<void*>(dA)};
    size_t kernArgSize = sizeof(kernArg);
    void* extra[]      = {
             MU_LAUNCH_PARAM_BUFFER_POINTER, &kernArg, MU_LAUNCH_PARAM_BUFFER_SIZE, &kernArgSize, MU_LAUNCH_PARAM_END};
    for (int sid = 0; sid < streamNum; ++sid) {
        checkMuErrors(muLaunchKernel(function, blocksPerGrid, 1, 1, /* grid dim */
            threadsPerBlock, 1, 1,                                  /* block dim */
            0,                                                      /* shared mem */
            stream[sid],                                            /* stream */
            nullptr,                                                /* params */
            extra                                                   /* extra */
            ));
    }

    checkMusaErrors(musaDeviceSynchronize());

    CPerfCounter timer;
    timer.Start();
    for (uint32_t i = 0; i < KernelsNum; i++) {
        KernArg kernArg = {
            0.123, reinterpret_cast<void*>(dB + i * sizeof(float)), reinterpret_cast<void*>(dA + i * sizeof(float))};
        size_t kernArgSize = sizeof(kernArg);

        void* extra[] = {
            MU_LAUNCH_PARAM_BUFFER_POINTER, &kernArg, MU_LAUNCH_PARAM_BUFFER_SIZE, &kernArgSize, MU_LAUNCH_PARAM_END};

        checkMuErrors(muLaunchKernel(function, blocksPerGrid, 1, 1, /* grid dim */
            threadsPerBlock, 1, 1,                                  /* block dim */
            0,                                                      /* shared mem */
            stream[i % streamNum],                                  /* stream */
            nullptr,                                                /* params */
            extra                                                   /* extra */
            ));
    }

    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    float timeElapsed = 0.0f;
    timeElapsed       = timer.GetElapsedSeconds() * 1000.f;
    checkMuErrors(muMemcpyDtoH(hA, dA, sizeBytes));
    bool resultIsCorrent = true;
    for (uint32_t i = 0; i < KernelsNum; ++i) {
        if (abs((hA[i] - (2.0 * i + 0.123))) > 0.0001) {
            std::cout << "Result check failed at A[" << i << "]=" << hA[i] << ", expert:" << 2.0 * i + 0.123 << std::endl;
            resultIsCorrent = false;
            exit(EXIT_FAILURE);
        }
    }
    checkMuErrors(muEventDestroy(startEvent));
    checkMuErrors(muEventDestroy(stopEvent));
    free(hA);
    free(hB);
    checkMuErrors(muMemFree(dA));
    checkMuErrors(muMemFree(dB));
    return timeElapsed;
}

BASELINE_F(launchKernel, MulStreams, LanchFixture, SamplesCount, IterationsCount) {
    float time = launchKernelByStreams();

    this->utime1->addValue(time);
    this->utp1->addValue(KernelsNum / (time / 1000));
}
