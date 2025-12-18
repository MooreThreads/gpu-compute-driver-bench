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
        for (int64_t i = 0; i <= 29; ++i) {
            ExperimentValue ev;
            ev.Value = sizeof(int) * (1LL << i);
            problemSpace.push_back(ev);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override { bytes = experimentValue.Value; }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->uband_h2d, this->uband_d2h};
    }

    int64_t bytes;
    musaEvent_t start;
    musaEvent_t stop;
    std::shared_ptr<UDMBandWidth> uband_h2d{new UDMBandWidth("*H2D")};
    std::shared_ptr<UDMBandWidth> uband_d2h{new UDMBandWidth("*D2H")};
};

static const int SamplesCount    = 3;
static const int IterationsCount = 1;

BASELINE_F(musaCopy, copyRegisterRate, CopyFixture, SamplesCount, IterationsCount) {
    float milliseconds = 0.f;
    CPerfCounter timer;
    musaError_t err = musaSuccess;
    void* d_B;
    const size_t numElements = bytes / sizeof(int);

    int *hA, *hB;
    hA = (int*)malloc(bytes);
    checkMusaErrors(musaHostRegister(hA, bytes, musaHostRegisterDefault));

    for (size_t i = 0; i < numElements; ++i) {
        hA[i] = static_cast<int>(i);
    }

    hB = (int*)malloc(bytes);
    checkMusaErrors(musaHostRegister(hB, bytes, musaHostRegisterDefault));

    checkMusaErrors(musaSetDevice(0));

    checkMusaErrors(musaMalloc(&d_B, bytes));

    musaMemset(hB, 0, bytes);

    timer.Restart();
    checkMusaErrors(musaMemcpyAsync(d_B, hA, bytes, musaMemcpyHostToDevice, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_h2d->addValue(static_cast<double>(bytes) / (milliseconds * 1000));

    timer.Restart();
    checkMusaErrors(musaMemcpyAsync(hB, d_B, bytes, musaMemcpyDeviceToHost, nullptr));
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

    if (errorCnt != 0) {
        printf("Failed in result verification!\n");
        exit(EXIT_FAILURE);
    }

    checkMusaErrors(musaHostUnregister(hA));
    checkMusaErrors(musaHostUnregister(hB));
    free(hA);
    free(hB);
    checkMusaErrors(musaFree(d_B));
}
