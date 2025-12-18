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

static const int SamplesCount        = 1;
static const int IterationsCount     = 1;
static const int streamingSizeInByte = 128 * 1024 * 1024;
static int maxPersistentLlcSize      = 0;

__global__ void reset_data(int4* pDataStreaming, int4 const* pDataPersistent, size_t dataStreamingSize,
    size_t dataPersistentSize, size_t numRepeats) {
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    size_t const stride{blockDim.x * gridDim.x};
    for (size_t r = 0; r < numRepeats; ++r) {
        for (size_t i = idx; i < dataStreamingSize; i += stride) {
            pDataStreaming[i].x = pDataPersistent[i % dataPersistentSize].x;
            pDataStreaming[i].y = pDataPersistent[i % dataPersistentSize].y;
            pDataStreaming[i].z = pDataPersistent[i % dataPersistentSize].z;
            pDataStreaming[i].w = pDataPersistent[i % dataPersistentSize].w;
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
    Printer::get().TableSetPbName("Size(B)");
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    std::cout << "LLC Size: " << prop.l2CacheSize / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Max Persistetnt LLC Size: " << prop.persistingL2CacheMaxSize / 1024 / 1024 << " MB" << std::endl;
    maxPersistentLlcSize = prop.persistingL2CacheMaxSize;
    checkMusaErrors(musaDeviceSetLimit(musaLimitPersistingL2CacheSize, maxPersistentLlcSize));
    Run(argc, argv);
    return 0;
}

class LlcPersistLanchFixture : public TestFixture {
public:
    LlcPersistLanchFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t i = maxPersistentLlcSize - 4 * 1024 * 1024; i <= maxPersistentLlcSize; i += 4 * 1024 * 1024) {
            problemSpace.push_back(i);
        }
        return problemSpace;
    }
    void setUp(const TestFixture::ExperimentValue& experimentValue) override { persistentSize = experimentValue.Value; }

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    int testNStreams(int numStream, int n, float* t1, float* t2);

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->uband, this->uband2};
    }

    uint32_t persistentSize;
    std::shared_ptr<UDMBandWidth> uband{new UDMBandWidth("wo Persist")};
    std::shared_ptr<UDMBandWidth> uband2{new UDMBandWidth("Persist")};
};

int LlcPersistLanchFixture::testNStreams(int numStream, int persistSizeInByte, float* t1, float* t2) {
    CPerfCounter timer;
    musaStream_t streams[numStream];
    for (int i = 0; i < numStream; ++i) {
        checkMusaErrors(musaStreamCreate(&streams[i]));
    }

    // warm up
    for (uint64_t i = 0; i < 10; ++i) {
        emptyKernel<<<1, 1>>>();
    }

    int4* devicePersistent;
    int4* deviceStreaming;
    checkMusaErrors(musaMalloc(&devicePersistent, persistSizeInByte));
    checkMusaErrors(musaMalloc(&deviceStreaming, streamingSizeInByte));

    checkMusaErrors(musaCtxResetPersistingL2Cache());

    dim3 const threadsPerBlock{512};
    dim3 const blocksPerGrid{1024};
    int const numRepeats{1000};

    timer.Reset();
    timer.Start();

    for (int i = 0; i < numStream; ++i) {
        reset_data<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(deviceStreaming, devicePersistent,
            streamingSizeInByte / sizeof(int4), persistSizeInByte / sizeof(int4), numRepeats);
    }
    for (int i = 0; i < numStream; ++i) {
        checkMusaErrors(musaStreamSynchronize(streams[i]));
    }

    timer.Stop();
    float const result1 = 2 * streamingSizeInByte * uint64_t(numRepeats) / timer.GetElapsedSeconds() / 1000.f / 1000.f;

    for (int i = 0; i < numStream; ++i) {
        musaStreamAttrValue streamAttribute;
        streamAttribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(devicePersistent);
        streamAttribute.accessPolicyWindow.num_bytes = persistSizeInByte;
        streamAttribute.accessPolicyWindow.hitRatio =
            std::min(static_cast<double>(persistSizeInByte) / maxPersistentLlcSize, 1.0);
        streamAttribute.accessPolicyWindow.hitProp  = musaAccessPropertyPersisting;
        streamAttribute.accessPolicyWindow.missProp = musaAccessPropertyStreaming;

        checkMusaErrors(musaStreamSetAttribute(streams[i], musaStreamAttributeAccessPolicyWindow, &streamAttribute));
    }

    timer.Reset();
    timer.Start();

    for (int i = 0; i < numStream; ++i) {
        reset_data<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(deviceStreaming, devicePersistent,
            streamingSizeInByte / sizeof(int4), persistSizeInByte / sizeof(int4), numRepeats);
    }
    for (int i = 0; i < numStream; ++i) {
        checkMusaErrors(musaStreamSynchronize(streams[i]));
    }

    timer.Stop();
    float const result2 = 2 * streamingSizeInByte * uint64_t(numRepeats) / timer.GetElapsedSeconds() / 1000.f / 1000.f;

    checkMusaErrors(musaCtxResetPersistingL2Cache());
    checkMusaErrors(musaFree(devicePersistent));
    checkMusaErrors(musaFree(deviceStreaming));
    for (int i = 0; i < numStream; ++i) {
        checkMusaErrors(musaStreamDestroy(streams[i]));
    }

    *t1 = result1;
    *t2 = result2;

    return 0;
}

BASELINE_F(LlcPersistentTest, singleStream, LlcPersistLanchFixture, SamplesCount, IterationsCount) {
    float result1, result2;
    int numStream = 1;
    int ans       = testNStreams(numStream, persistentSize, &result1, &result2);
    this->uband->addValue(result1);
    this->uband2->addValue(result2);
}