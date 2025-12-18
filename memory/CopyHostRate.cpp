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

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "helper_musa.h"
#include "helper_musa_drvapi.h"
#include "musa.h"
#include "musa_runtime.h"
#include "timer_his.h"
#define musa_ALIGNED 128U
#define musa_ALIGN(x) ((x + musa_ALIGNED - 1) & ~(musa_ALIGNED - 1))

int main(int argc, char** argv) {
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << std::endl;
    Run(argc, argv);
    return 0;
}

class CopyFixture : public TestFixture {
public:
    CopyFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        // 16B-32M
        for (int i = 4; i <= 25; ++i) {
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
        return {this->uband, this->utime};
    }
    int64_t copySize;
    std::shared_ptr<UDMBandWidth> uband{new UDMBandWidth("Band(Mps)")};
    std::shared_ptr<UDMCPUTime> utime{new UDMCPUTime("Time(us)")};
};

static const int SamplesCount    = 1;
static const int IterationsCount = 10;

BASELINE_F(memCopy, hostRate, CopyFixture, SamplesCount, IterationsCount) {
    char* hptrs = (char*)malloc(copySize);
    char* hptrd = (char*)malloc(copySize);
    if (!hptrd || !hptrs) {
        console::SetConsoleColor(console::ConsoleColor::Red);
        std::cout << "FAILED malloc host memory, test exit" << std::endl;
        exit(1);
    }
    char randChar = Random() % 254;
    for (int i = 0; i < copySize; ++i) {
        hptrs[i] = randChar;
        hptrd[i] = randChar + 1;
    }
    CPerfCounter timer;
    timer.Reset();
    timer.Start();
    memcpy(hptrd, hptrs, copySize);
    timer.Stop();
    double tcopy = timer.GetElapsedSeconds();
    this->utime->addValue(tcopy * 1000 * 1000);
    this->uband->addValue(copySize / (tcopy * (1 << 20))); // MByte / S
    for (int i = 0; i < copySize; ++i) {
        if (hptrd[i] != randChar) {
            console::SetConsoleColor(console::ConsoleColor::Red);
            printf("FAILED check copy Result at %d, test exit\n", i);
            exit(EXIT_FAILURE);
        }
    }
    free(hptrs);
    free(hptrd);
}