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

#include <assert.h>
#include <errno.h>
#include <unistd.h>

#include <iomanip>
#include <string>

#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "musa.h"
#include "musa_runtime.h"
#include "timer_his.h"
#include "helper_musa.h"
#include "helper_musa_drvapi.h"

static const int SamplesCount    = 1;
static const int IterationsCount = 10;
static const int N               = 30;

int main(int argc, char** argv) {
    int deviceCount;
    checkMusaErrors(musaGetDeviceCount(&deviceCount));
    musaDeviceProp prop;
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    Printer::get().TableSetPbName("counts");
    Run(argc, argv);
    return 0;
}

class LanchFixture : public TestFixture {
public:
    LanchFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t i = 0; i < 3; ++i) {
            problemSpace.push_back(N);
        }
        return problemSpace;
    }
    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
#ifdef TEST_ON_NVIDIA
        codeObjectFile = std::string("./VectorAdd.ptx");
        ;
#else
        codeObjectFile = std::string("./VectorAdd.elf");
#endif
        functionName = std::string("_Z9VectorAddPiS_");
        elementNum   = 1e7;
        checkMuErrors(muInit(0));
        checkMuErrors(muCtxCreate(&ctx, 0, 0));
        MUdevice device = -1;
        checkMuErrors(muCtxGetDevice(&device));
        unsigned flags = -1;
        checkMuErrors(muCtxGetFlags(&flags));
        MUfunc_cache cacheConfig = MU_FUNC_CACHE_PREFER_EQUAL;
        checkMuErrors(muCtxGetCacheConfig(&cacheConfig));
        checkMuErrors(muModuleLoad(&module, codeObjectFile.c_str()));
        checkMuErrors(muModuleGetFunction(&function, module, functionName.c_str()));
    }

    void tearDown() override {
        checkMuErrors(muModuleUnload(module));
        checkMuErrors(muCtxDestroy(ctx));
    }

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {
    }
    void onExperimentEnd() override {
    }

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->utime1, this->utime2};
    }
    float efficiencyBetweenKernelAndD2Dcopy();
    float efficiencyKernels();
    std::string codeObjectFile;
    std::string functionName;
    uint32_t elementNum;
    uint32_t loopNum = N;
    MUcontext ctx    = nullptr;
    MUmodule module;
    MUfunction function;
    std::shared_ptr<UDMGPUTime> utime1{new UDMGPUTime("Ker(ms)", StatsView::MIN | StatsView::MAX)};
    std::shared_ptr<UDMGPUTime> utime2{new UDMGPUTime("Ker&Cpy", StatsView::MIN | StatsView::MAX)};
};

float LanchFixture::efficiencyKernels() {
    const size_t sizeBytes = elementNum * sizeof(int);
    MUdeviceptr dA = 0, dB = 0;
    int* hA = reinterpret_cast<int*>(malloc(sizeBytes));
    int* hB = reinterpret_cast<int*>(malloc(sizeBytes));
    MUstream stream;
    checkMuErrors(muStreamCreate(&stream, 0));

    assert(hA != nullptr && hB != nullptr && "host malloc failed!");

    checkMuErrors(muMemAlloc(&dA, sizeBytes));
    checkMuErrors(muMemAlloc(&dB, sizeBytes));
    assert(dA != 0 && dB != 0 && "device malloc failed!");
    for (uint32_t i = 0; i < elementNum; ++i) {
        hB[i] = 1;
    }
    MUevent startEvent, stopEvent;
    checkMuErrors(muEventCreate(&startEvent, 0));
    checkMuErrors(muEventCreate(&stopEvent, 0));

    checkMuErrors(muMemcpyHtoD(dB, hB, sizeBytes));
    checkMuErrors(muMemsetD8(dA, 0, sizeBytes));
    int threadsPerBlock = 1024;
    int blocksPerGrid   = (elementNum + threadsPerBlock - 1) / threadsPerBlock;

    struct KernArg {
        void *A, *B;
    };

    /* Launch in default stream. */
    checkMuErrors(muEventRecord(startEvent, stream));
    for (uint32_t i = 0; i < loopNum; i++) {
        KernArg kernArg    = {reinterpret_cast<void*>(dA), reinterpret_cast<void*>(dB)};
        size_t kernArgSize = sizeof(kernArg);

        void* extra[] = {
            MU_LAUNCH_PARAM_BUFFER_POINTER, &kernArg, MU_LAUNCH_PARAM_BUFFER_SIZE, &kernArgSize, MU_LAUNCH_PARAM_END};

        checkMuErrors(muLaunchKernel(function, blocksPerGrid, 1, 1, /* grid dim */
            threadsPerBlock, 1, 1,                                  /* block dim */
            0,                                                      /* shared mem */
            stream,                                                 /* stream */
            nullptr,                                                /* params */
            extra                                                   /* extra */
            ));
    }
    checkMuErrors(muEventRecord(stopEvent, stream));
    checkMuErrors(muEventSynchronize(stopEvent));
    float timeElapsed = 0.0f;
    checkMuErrors(muEventElapsedTime(&timeElapsed, startEvent, stopEvent));
    checkMuErrors(muMemcpyDtoH(hA, dA, sizeBytes));
    bool resultIsCorrent = true;
    for (uint32_t i = 0; i < elementNum; ++i) {
        if (hA[i] != static_cast<int>(loopNum)) {
            std::cout << "Result check1 failed at A[" << i << "]=" << hA[i] << ", expert:" << static_cast<int>(loopNum)
                      << std::endl;
            resultIsCorrent = false;
            exit(EXIT_FAILURE);
        }
    }
    checkMuErrors(muEventSynchronize(stopEvent));
    checkMuErrors(muStreamDestroy(stream));
    checkMuErrors(muEventDestroy(startEvent));
    checkMuErrors(muEventDestroy(stopEvent));
    free(hA);
    free(hB);
    checkMuErrors(muMemFree(dA));
    checkMuErrors(muMemFree(dB));
    return timeElapsed;
}

float LanchFixture::efficiencyBetweenKernelAndD2Dcopy() {
    const size_t sizeBytes = elementNum * sizeof(int);
    MUdeviceptr dA = 0, dB = 0;
    int* hA = reinterpret_cast<int*>(malloc(sizeBytes));
    int* hB = reinterpret_cast<int*>(malloc(sizeBytes));
    MUstream stream;
    checkMuErrors(muStreamCreate(&stream, 0));
    assert(hA != nullptr && hB != nullptr && "host malloc failed!");
    checkMuErrors(muMemAlloc(&dA, sizeBytes));
    checkMuErrors(muMemAlloc(&dB, sizeBytes));
    assert(dA != 0 && dB != 0 && "device malloc failed!");
    for (uint32_t i = 0; i < elementNum; ++i) {
        hB[i] = 1;
    }
    MUevent startEvent, stopEvent;
    checkMuErrors(muEventCreate(&startEvent, 0));
    checkMuErrors(muEventCreate(&stopEvent, 0));
    checkMuErrors(muMemcpyHtoD(dB, hB, sizeBytes));
    checkMuErrors(muMemsetD8(dA, 0, sizeBytes));

    int threadsPerBlock = 1024;
    int blocksPerGrid   = (elementNum + threadsPerBlock - 1) / threadsPerBlock;

    struct KernArg {
        void *A, *B;
    };
    /* Launch in default stream. */
    checkMuErrors(muEventRecord(startEvent, stream));
    for (uint32_t i = 0; i < loopNum; i++) {
        KernArg kernArg    = {reinterpret_cast<void*>(dA), reinterpret_cast<void*>(dB)};
        size_t kernArgSize = sizeof(kernArg);
        void* extra[]      = {
                 MU_LAUNCH_PARAM_BUFFER_POINTER, &kernArg, MU_LAUNCH_PARAM_BUFFER_SIZE, &kernArgSize, MU_LAUNCH_PARAM_END};
        muLaunchKernel(function, blocksPerGrid, 1, 1, /* grid dim */
            threadsPerBlock, 1, 1,                    /* block dim */
            0,                                        /* shared mem */
            stream,                                   /* stream */
            nullptr,                                  /* params */
            extra                                     /* extra */
        );
        checkMuErrors(muMemcpyDtoDAsync(dB, dA, sizeBytes, stream));
    }
    checkMuErrors(muEventRecord(stopEvent, stream));
    checkMuErrors(muEventSynchronize(stopEvent));
    float timeElapsed = 0.0f;
    checkMuErrors(muEventElapsedTime(&timeElapsed, startEvent, stopEvent));
    checkMuErrors(muMemcpyDtoH(hA, dA, sizeBytes));
    bool resultIsCorrent = true;
    for (uint32_t i = 0; i < elementNum; ++i) {
        if (hA[i] != pow(2, loopNum - 1)) {
            std::cout << "Result check2 failed at A[" << i << "]=" << hA[i] << ", expert:" << pow(2, loopNum - 1)
                      << std::endl;
            resultIsCorrent = false;
            exit(EXIT_FAILURE);
        }
    }
    checkMuErrors(muEventSynchronize(stopEvent));
    checkMuErrors(muStreamDestroy(stream));
    checkMuErrors(muEventDestroy(startEvent));
    checkMuErrors(muEventDestroy(stopEvent));
    free(hA);
    free(hB);
    checkMuErrors(muMemFree(dA));
    checkMuErrors(muMemFree(dB));
    return timeElapsed;
}

BASELINE_F(Kernel, efficiency, LanchFixture, SamplesCount, IterationsCount) {
    float time1 = efficiencyKernels();
    float time2 = efficiencyBetweenKernelAndD2Dcopy();
    this->utime1->addValue(time1);
    this->utime2->addValue(time2);
}