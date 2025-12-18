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

class MallocFixture : public TestFixture {
public:
    MallocFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t elements = 1LL << 0; elements <= (1LL << 32); elements *= 2) {
            problemSpace.push_back(elements);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        this->mallocSize = experimentValue.Value;
    }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->uband, this->utime, this->uband1, this->utime1};
    }
    int64_t mallocSize;
    musaEvent_t start;
    musaEvent_t stop;
    void* deviceMemory;
    std::shared_ptr<UDMBandWidth> uband{new UDMBandWidth("*Breg")};
    std::shared_ptr<UDMGPUTime> utime{new UDMGPUTime("treg")};
    std::shared_ptr<UDMBandWidth> uband1{new UDMBandWidth("*Bunreg")};
    std::shared_ptr<UDMGPUTime> utime1{new UDMGPUTime("tunreg")};
};

static const int SamplesCount    = 3;
static const int IterationsCount = 1;
BASELINE_F(hostReg, muHostReg, MallocFixture, SamplesCount, IterationsCount) {
    char* hostA;
    CPerfCounter timer;
    hostA = (char*)malloc(this->mallocSize);
    if (hostA == nullptr) {
        printf("malloc failed\n");
        exit(1);
    }
    for (int64_t i = 0; i < this->mallocSize; i++) {
        hostA[i] = 1;
    }

    char* hTemp;
    hTemp = (char*)malloc(this->mallocSize);
    checkMusaErrors(musaHostRegister(hTemp, this->mallocSize, 0));
    checkMusaErrors(musaHostUnregister(hTemp));
    free(hTemp);

    timer.Restart();
    checkMusaErrors(musaHostRegister(hostA, this->mallocSize, 0));
    timer.Stop();
    double milliseconds = timer.GetElapsedSeconds() * 1000;
    this->utime->addValue(milliseconds);
    this->uband->addValue(static_cast<double>(mallocSize) / (milliseconds * 1000));

    timer.Restart();
    checkMusaErrors(musaHostUnregister(hostA));
    timer.Stop();
    double time_s = timer.GetElapsedSeconds();
    this->utime1->addValue(time_s * 1000);
    this->uband1->addValue(static_cast<double>(this->mallocSize) / (time_s * 1000 * 1000));
    free(hostA);
}
