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
    Printer::get().TableSetPbName("size");
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    std::cout << ":warning: If you get a bandwidth of -1, it indicates that P2P copying is not "
                 "supported between the two devices."
              << std::endl;
    Run(argc, argv);
    return 0;
}

class P2PFixture : public TestFixture {
public:
    P2PFixture() {}

    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int i = 2; i <= 30; ++i) {
            problemSpace.push_back(1 << i);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override { this->copySize = experimentValue.Value; }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->uband1, this->utime1};
    }

    double p2pTest(int setId, int srcId, int dstId, int sizeInbytes);

    int copySize;
    std::shared_ptr<UDMBandWidth> uband1{new UDMBandWidth("*band")};
    std::shared_ptr<UDMGPUTime> utime1{new UDMGPUTime("time")};
};

static const int SamplesCount    = 10;
static const int IterationsCount = 1;

double P2PFixture::p2pTest(int setId, int srcId, int dstId, int sizeInBytes) {
    int numDevices;
    checkMusaErrors(musaGetDeviceCount(&numDevices));

    if (srcId < 0 || srcId >= numDevices || dstId < 0 || dstId >= numDevices || setId < 0 || setId >= numDevices) {
        // std::cerr << "Invalid device IDs. Please specify valid device IDs." << std::endl;
        return -1;
    }

    if (srcId == dstId || srcId == setId || dstId == setId) {
        // std::cerr << "Source and destination devices cannot be the same." << std::endl;
        return -1;
    }

    checkMusaErrors(musaSetDevice(srcId));
    checkMusaErrors(musaDeviceEnablePeerAccess(dstId, 0));

    // init in src device
    int *d_srcData, *d_dstData;
    checkMusaErrors(musaMalloc(&d_srcData, sizeInBytes));
    checkMusaErrors(musaMemset(d_srcData, 0, sizeInBytes));

    // malloc in dst device
    checkMusaErrors(musaSetDevice(dstId));
    checkMusaErrors(musaMalloc(&d_dstData, sizeInBytes));

    // p2p copy and test bandwidth
    checkMusaErrors(musaSetDevice(setId));
    CPerfCounter timer;
    timer.Restart();
    checkMusaErrors(musaMemcpyPeer(d_dstData, dstId, d_srcData, srcId, sizeInBytes));
    checkMusaErrors(musaDeviceSynchronize());
    float milliseconds = 0;
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;

    // clear
    checkMusaErrors(musaFree(d_srcData));
    checkMusaErrors(musaSetDevice(dstId));
    checkMusaErrors(musaFree(d_dstData));
    checkMusaErrors(musaSetDevice(srcId));
    checkMusaErrors(musaDeviceDisablePeerAccess(dstId));

    return milliseconds;
}

BASELINE_F(memCopy, P2P_IJK, P2PFixture, SamplesCount, IterationsCount) {
    double time = p2pTest(0, 1, 2, copySize);
    utime1->addValue(time);
    if (time > 0) {
        uband1->addValue(static_cast<double>(copySize) / (time * 1000));
    } else {
        uband1->addValue(-1);
    }
}
