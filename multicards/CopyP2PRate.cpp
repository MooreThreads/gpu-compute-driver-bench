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

#include "musa.h"
#include "musa_runtime.h"
#include "Celero.h"
#include "UserDefinedMeasurements.h"

#include "timer_his.h"
#include "helper_musa.h"
#include "helper_musa_drvapi.h"

int main(int argc, char** argv) {
    int deviceCount;
    checkMusaErrors(musaGetDeviceCount(&deviceCount));
    musaDeviceProp prop;
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    Printer::get().TableSetPbName("srcID");
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    std::cout << ":warning: If you get a bandwidth of -1, it indicates that P2P copying is not "
                 "supported between the two devices."
              << std::endl;
    Run(argc, argv);
    return 0;
}

class P2PFixture : public TestFixture {
public:
    P2PFixture() {
        int tempDeviceCount;
        checkMusaErrors(musaGetDeviceCount(&tempDeviceCount));
        deviceCount = tempDeviceCount;
        ubands.reserve(deviceCount);
        for (int i = 0; i < deviceCount; ++i) {
            ubands.push_back(std::make_shared<UDMBandWidth>("*DstID:" + std::to_string(i)));
        }
    }
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int i = 0; i < deviceCount; ++i) {
            problemSpace.push_back(i);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override { this->srcID = experimentValue.Value; }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        std::vector<std::shared_ptr<UserDefinedMeasurement>> umds(ubands.size());
        for (int i = 0; i < deviceCount; ++i) {
            umds.at(i) = ubands.at(i);
        }
        return umds;
    }

    double p2pTest(int srcId, int dstId, int sizeInBytes, bool enableP2p) {
        int numDevices;
        CPerfCounter timer;
        checkMusaErrors(musaGetDeviceCount(&numDevices));

        if (srcId < 0 || srcId >= numDevices || dstId < 0 || dstId >= numDevices) {
            // std::cerr << "Invalid device IDs. Please specify valid device IDs." << std::endl;
            return -1;
        }

        if (srcId == dstId) {
            // std::cerr << "Source and destination devices cannot be the same." << std::endl;
            return -1;
        }

        // set p2p copy
        if (enableP2p) {
            checkMusaErrors(musaSetDevice(srcId));
            checkMusaErrors(musaDeviceEnablePeerAccess(dstId, 0));
        }

        // init in src device
        int *d_srcData, *d_dstData;
        checkMusaErrors(musaMalloc(&d_srcData, sizeInBytes));
        checkMusaErrors(musaMemset(d_srcData, 0, sizeInBytes));

        // malloc in dst device
        checkMusaErrors(musaSetDevice(dstId));
        checkMusaErrors(musaMalloc(&d_dstData, sizeInBytes));

        // p2p copy and test bandwidth
        checkMusaErrors(musaSetDevice(srcId));
        timer.Restart();
        checkMusaErrors(musaMemcpyPeer(d_dstData, dstId, d_srcData, srcId, sizeInBytes));
        checkMusaErrors(musaDeviceSynchronize());
        timer.Stop();
        float milliseconds = 0;
        milliseconds       = timer.GetElapsedSeconds() * 1000.f;

        // clear
        checkMusaErrors(musaFree(d_srcData));
        checkMusaErrors(musaSetDevice(dstId));
        checkMusaErrors(musaFree(d_dstData));
        if (enableP2p) {
            checkMusaErrors(musaSetDevice(srcId));
            checkMusaErrors(musaDeviceDisablePeerAccess(dstId));
        }
        return sizeInBytes / (milliseconds * 1e-3 * 1e6);
    }

    int deviceCount;
    int64_t srcID;
    mutable std::vector<std::shared_ptr<UDMBandWidth>> ubands;
};

static const int SamplesCount    = 10;
static const int IterationsCount = 1;

BASELINE_F(memCopy, DISABLE_P2P_1B, P2PFixture, SamplesCount, IterationsCount) {
    for (int dstID = 0; dstID < deviceCount; ++dstID) {
        double band = p2pTest(srcID, dstID, 1, false);
        ubands[dstID]->addValue(band);
    }
}

#define MEM_COPY_BENCHMARK(name, srcID, deviceCount, size, enableP2p)       \
    BENCHMARK_F(memCopy, name, P2PFixture, SamplesCount, IterationsCount) { \
        for (int dstID = 0; dstID < deviceCount; ++dstID) {                 \
            double band = p2pTest(srcID, dstID, size, enableP2p);           \
            ubands[dstID]->addValue(band);                                  \
        }                                                                   \
    }

MEM_COPY_BENCHMARK(DISABLE_P2P_4B, srcID, deviceCount, 4, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_16B, srcID, deviceCount, 16, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_64B, srcID, deviceCount, 64, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_256B, srcID, deviceCount, 256, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_1K, srcID, deviceCount, 1 * 1024, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_4K, srcID, deviceCount, 4 * 1024, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_16K, srcID, deviceCount, 16 * 1024, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_64K, srcID, deviceCount, 64 * 1024, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_246K, srcID, deviceCount, 256 * 1024, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_1M, srcID, deviceCount, 1024 * 1024, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_4M, srcID, deviceCount, 4 * 1024 * 1024, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_16M, srcID, deviceCount, 16 * 1024 * 1024, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_64M, srcID, deviceCount, 64 * 1024 * 1024, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_256M, srcID, deviceCount, 256 * 1024 * 1024, false);
MEM_COPY_BENCHMARK(DISABLE_P2P_1G, srcID, deviceCount, 1024 * 1024 * 1024, false);

MEM_COPY_BENCHMARK(ENABLE_P2P_4B, srcID, deviceCount, 4, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_16B, srcID, deviceCount, 16, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_64B, srcID, deviceCount, 64, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_256B, srcID, deviceCount, 256, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_1K, srcID, deviceCount, 1 * 1024, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_4K, srcID, deviceCount, 4 * 1024, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_16K, srcID, deviceCount, 16 * 1024, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_64K, srcID, deviceCount, 64 * 1024, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_246K, srcID, deviceCount, 256 * 1024, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_1M, srcID, deviceCount, 1024 * 1024, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_4M, srcID, deviceCount, 4 * 1024 * 1024, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_16M, srcID, deviceCount, 16 * 1024 * 1024, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_64M, srcID, deviceCount, 64 * 1024 * 1024, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_256M, srcID, deviceCount, 256 * 1024 * 1024, true);
MEM_COPY_BENCHMARK(ENABLE_P2P_1G, srcID, deviceCount, 1024 * 1024 * 1024, true);
