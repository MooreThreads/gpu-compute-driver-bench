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
#define musa_ALIGNED 128ULL
#define musa_ALIGN(x) ((x + musa_ALIGNED - 1) & ~(musa_ALIGNED - 1))

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
class CopyFixture : public TestFixture {
public:
    CopyFixture() {}

    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int i = 2; i <= 32; ++i) {
            TestFixture::ExperimentValue ev;
            ev.Value = 1LL << i;
            problemSpace.push_back(ev);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override { this->copySize = experimentValue.Value; }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->uband_h2d, this->uband_d2h, this->uband_d2d, this->ucont_error};
    }

    int64_t copySize;
    musaEvent_t start;
    musaEvent_t stop;
    int *d_B, *d_C;
    std::shared_ptr<UDMBandWidth> uband_h2d{new UDMBandWidth("*H2D")};
    std::shared_ptr<UDMBandWidth> uband_d2d{new UDMBandWidth("*D2D")};
    std::shared_ptr<UDMBandWidth> uband_d2h{new UDMBandWidth("*D2H")};
    std::shared_ptr<UDMCount> ucont_error{new UDMCount("err")};
};

static const int SamplesCount    = 1;
static const int IterationsCount = 3;

size_t CheckResult(int* dst, int* src, size_t numElements, const char* dstDesc) {
    size_t errorCnt = 0;
    if (memcmp(dst, src, numElements * sizeof(int))) {
        for (size_t i = 0; i < numElements; ++i) {
            if (dst[i] != src[i]) {
                errorCnt++;
                printf("Result check FAILED at %s[%d]=%d\n", dstDesc, static_cast<int>(i), dst[i]);
                break;
            }
        }
    }
    return errorCnt;
}

BASELINE_F(musaCopy, copyRateUnAligned, CopyFixture, SamplesCount, IterationsCount) {
    // step1.prepare
    float milliseconds = 0.f;
    size_t numElements = copySize / sizeof(int);
    size_t errorCnt    = 0;
    int* hA            = static_cast<int*>(aligned_alloc(musa_ALIGNED, musa_ALIGN(copySize + 16)));
    if (hA == nullptr) {
        printf("host A malloc FAILED!\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < numElements + 4; ++i) {
        hA[i] = static_cast<int>(i);
    }
    int* hB = static_cast<int*>(aligned_alloc(musa_ALIGNED, musa_ALIGN(copySize + 16)));
    if (hB == nullptr) {
        printf("host B malloc FAILED!\n");
        exit(EXIT_FAILURE);
    }
    checkMusaErrors(musaMalloc(&d_B, copySize));
    checkMusaErrors(musaMalloc(&d_C, copySize));
    memset(hB, 0, musa_ALIGN(copySize + 16));
    CPerfCounter timer;

    // test h2d
    timer.Restart();
    checkMusaErrors(musaMemcpyAsync(d_B, hA + 4, copySize, musaMemcpyHostToDevice, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_h2d->addValue(static_cast<double>(copySize) / (milliseconds * 1000));
    checkMusaErrors(musaMemcpy(hB, d_B, copySize, musaMemcpyDeviceToHost));
    checkMusaErrors(musaDeviceSynchronize());
    errorCnt += CheckResult(hB, hA + 4, numElements, "hB");

    // test d2d
    timer.Restart();
    checkMusaErrors(musaMemcpyAsync(d_C, d_B, copySize, musaMemcpyDeviceToDevice, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_d2d->addValue(2.0 * static_cast<double>(copySize) / (milliseconds * 1000));

    // test d2h
    timer.Restart();
    checkMusaErrors(musaMemcpyAsync(hB + 4, d_C, copySize, musaMemcpyDeviceToHost, nullptr));
    checkMusaErrors(musaDeviceSynchronize());
    timer.Stop();
    milliseconds = timer.GetElapsedSeconds() * 1000.f;
    this->uband_d2h->addValue(static_cast<double>(copySize) / (milliseconds * 1000));
    checkMusaErrors(musaDeviceSynchronize());
    errorCnt += CheckResult(hB + 4, hA + 4, numElements, "hB");

    if (errorCnt > 0) {
        exit(EXIT_FAILURE);
    }

    free(hA);
    free(hB);
    checkMusaErrors(musaFree(d_B));
    checkMusaErrors(musaFree(d_C));
}