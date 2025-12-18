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

class ReUseageFixture : public TestFixture {
public:
    ReUseageFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t elements = 1LL << 0; elements <= (1LL << 26); elements *= 2) {
            problemSpace.push_back(elements * 1.1);
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
        return {this->useage};
    }
    int64_t mallocSize;
    std::shared_ptr<UDMUseage> useage{new UDMUseage("Reuse")};
};

static const int SamplesCount    = 3;
static const int IterationsCount = 1;

#include <cstdlib>
#include <ctime>

BASELINE_F(Reuseage, malloc, ReUseageFixture, SamplesCount, IterationsCount) {
    char* hp;
    char* rp;
    hp              = (char*)malloc(mallocSize);
    char randomChar = Random() % 255;
    for (int i = 0; i < mallocSize; ++i) {
        hp[i] = randomChar;
    }
    free(hp);

    rp      = (char*)malloc(mallocSize);
    int cnt = 0;
    for (int i = 0; i < mallocSize; ++i) {
        if (rp[i] == randomChar)
            ++cnt;
    }
    this->useage->addValue(double(cnt) / double(mallocSize));
    free(rp);
}

BENCHMARK_F(Reuseage, mumalloc, ReUseageFixture, SamplesCount, IterationsCount) {
    char* hp;
    char* hrp;
    char* dp;
    char* drp;
    char randomChar = Random() % 255;

    // step1. copy randow value to device
    hp = (char*)malloc(mallocSize);
    checkMusaErrors(musaMalloc(&dp, mallocSize));
    for (int i = 0; i < mallocSize; ++i) {
        hp[i] = randomChar;
    }
    checkMusaErrors(musaMemcpy(dp, hp, mallocSize, musaMemcpyHostToDevice));
    free(hp);

    // step2. musafree and musamalloc other memory
    checkMusaErrors(musaFree(dp));
    checkMusaErrors(musaMalloc(&drp, mallocSize));
    hrp = (char*)malloc(mallocSize);
    memset(hrp, randomChar + 1, mallocSize);

    // step3. copy drp memory and check is it reused?
    checkMusaErrors(musaMemcpy(hrp, drp, mallocSize, musaMemcpyDeviceToHost));
    checkMusaErrors(musaDeviceSynchronize());
    int cnt = 0;
    for (int i = 0; i < mallocSize; ++i) {
        if (hrp[i] == randomChar)
            ++cnt;
    }
    this->useage->addValue(double(cnt) / double(mallocSize));
    free(hrp);
    checkMusaErrors(musaFree(drp));
}