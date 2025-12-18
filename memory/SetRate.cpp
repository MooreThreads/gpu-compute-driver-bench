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
class SetFixture : public TestFixture {
public:
    SetFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t elements = 1LL << 0; elements <= (1LL << 32); elements *= 2) {
            problemSpace.push_back(elements);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override { this->setSize = experimentValue.Value; }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->ubandD1D8, this->ubandD2D8, this->ubandD3D8};
    }
    int64_t setSize;
    musaEvent_t start;
    musaEvent_t stop;

    std::shared_ptr<UDMBandWidth> ubandD1D8{new UDMBandWidth("*D1D8")};
    std::shared_ptr<UDMBandWidth> ubandD2D8{new UDMBandWidth("*D2D8")};
    std::shared_ptr<UDMBandWidth> ubandD3D8{new UDMBandWidth("*D3D8")};
};

static const int SamplesCount    = 3;
static const int IterationsCount = 1;

BASELINE_F(memSet, setVal, SetFixture, SamplesCount, IterationsCount) {
    float milliseconds = 0.f;
    CPerfCounter timer;
    void* d_A;
    uint8_t uc = 0x77;
    // test d1d8
    checkMusaErrors(musaMalloc(&d_A, this->setSize));

    timer.Restart();
    musaMemsetAsync(d_A, uc, setSize, nullptr);
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->ubandD1D8->addValue(static_cast<double>(setSize) / (milliseconds * 1000));
    checkMusaErrors(musaFree(d_A));

    // test d2d8
    void* d_B;
    size_t height       = (setSize >= 16) ? 16 : 1;
    size_t widthInBytes = setSize / height;
    size_t pitch        = 0;
    checkMusaErrors(musaMallocPitch((void**)&d_B, &pitch, widthInBytes, height));

    size_t d_B_bytes = pitch * height;

    timer.Restart();
    checkMusaErrors(musaMemset2D(d_B, pitch, uc, widthInBytes, height));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->ubandD2D8->addValue(static_cast<double>(d_B_bytes) / (milliseconds * 1000));
    checkMusaErrors(musaFree(d_B));

    // test d3d8
    struct musaPitchedPtr d_C;
    size_t height_3d         = (setSize >= 100) ? 100 : 1;
    size_t depth_3d          = (setSize >= 100 * 10) ? 10 : 1;
    size_t width_3d          = setSize / height_3d / depth_3d;
    struct musaExtent extent = {width_3d * sizeof(uint8_t), height_3d, depth_3d};
    checkMusaErrors(musaMalloc3D(&d_C, extent));

    timer.Restart();
    checkMusaErrors(musaMemset3D(d_C, uc, extent));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->ubandD3D8->addValue(static_cast<double>(setSize) / (milliseconds * 1000));
    checkMusaErrors(musaFree(d_C.ptr));
}
