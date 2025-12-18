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
#include <iostream>
#include <unistd.h>
#include <errno.h>
#include <string>

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

class CreateStreamFixture : public TestFixture {
public:
    CreateStreamFixture() { checkMusaErrors(musaGetDeviceCount(&m_TotalCards)); }
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int i = 1; i <= 64; i *= 2) {
            problemSpace.push_back(i);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override { m_StreamNum = experimentValue.Value; }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}

    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->utimeCreate, this->utimeDestroy, this->uFreCreate, this->uFreDestroy};
    }

    int32_t m_TotalCards;
    uint64_t m_StreamNum;

    std::shared_ptr<UDMGPUTime> utimeCreate{new UDMGPUTime("Tc(ms)")};
    std::shared_ptr<UDMGPUTime> utimeDestroy{new UDMGPUTime("Td(ms)")};
    std::shared_ptr<UDMThroughPut> uFreCreate{new UDMThroughPut("Fc*")};
    std::shared_ptr<UDMThroughPut> uFreDestroy{new UDMThroughPut("Fd*")};
};

static const int SamplesCount    = 200;
static const int IterationsCount = 1;

BASELINE_F(resouce, stream, CreateStreamFixture, SamplesCount, IterationsCount) {
    CPerfCounter timer;
    musaStream_t* streams = new musaStream_t[m_StreamNum];

    void* dTemp;
    checkMusaErrors(musaMalloc(&dTemp, 4096));
    checkMusaErrors(musaFree(dTemp));

    timer.Restart();
    for (int i = 0; i < m_StreamNum; ++i) {
        checkMusaErrors(musaStreamCreate(&streams[i]));
    }
    timer.Stop();
    double ms = timer.GetElapsedSeconds() * 1000.0;
    utimeCreate->addValue(ms / m_StreamNum);
    uFreCreate->addValue(m_StreamNum * 1000 / ms);

    timer.Restart();
    for (int i = 0; i < m_StreamNum; ++i) {
        checkMusaErrors(musaStreamDestroy(streams[i]));
    }
    timer.Stop();
    ms = timer.GetElapsedSeconds() * 1000.0;
    utimeDestroy->addValue(ms / m_StreamNum);
    uFreDestroy->addValue(m_StreamNum * 1000 / ms);
    delete[] streams;
}
