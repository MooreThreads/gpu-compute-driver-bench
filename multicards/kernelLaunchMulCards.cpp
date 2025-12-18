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
    Printer::get().TableSetPbName("ThreadCnt");
    Run(argc, argv);
    return 0;
}

class MulCardsLanchFixture : public TestFixture {
public:
    MulCardsLanchFixture() { checkMusaErrors(musaGetDeviceCount(&m_TotalCards)); }
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (uint32_t i = 1; i <= m_TotalCards; i++) {
            problemSpace.push_back(i);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        functionName    = std::string("_Z11EmptyKernelv");
        m_ThreadsCounts = experimentValue.Value;
    }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}

    void onExperimentEnd() override {}

    void threadLaunchFunc(int32_t tid, int32_t cid, bool bRecord, int launchCount, float* pDurationMs);

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->uCount, this->utimeAvgPer, this->utimeTotal, this->utp};
    }
#ifdef TEST_ON_NVIDIA
    std::string codeObjectFile = std::string("./EmptyKernel.ptx");
#else
    std::string codeObjectFile = std::string("./EmptyKernel.elf");
#endif
    std::string functionName = std::string("_Z11EmptyKernelv");
    int32_t m_TotalCards;
    uint64_t m_ThreadsCounts;

    std::shared_ptr<UDMCount> uCount{new UDMCount("LaunchCnt")};
    std::shared_ptr<UDMGPUTime> utimeTotal{new UDMGPUTime("tAll(ms)", StatsView::MIN | StatsView::MAX)};
    std::shared_ptr<UDMGPUTime> utimeAvgPer{new UDMGPUTime("tPer(us)")};
    std::shared_ptr<UDMThroughPut> utp{new UDMThroughPut("*TP(s^-1)")};
};

void MulCardsLanchFixture::threadLaunchFunc(int32_t tid, int32_t cid, bool bRecord, int launchCount, float* pDurationMs) {
    checkMusaErrors(musaSetDevice(cid));

    musaEvent_t start;
    musaEvent_t stop;
    checkMusaErrors(musaEventCreate(&start));
    checkMusaErrors(musaEventCreate(&stop));

    musaStream_t stream = nullptr;

    MUmodule module;
    checkMuErrors(muModuleLoad(&module, codeObjectFile.c_str()));

    MUfunction function;
    checkMuErrors(muModuleGetFunction(&function, module, functionName.c_str()));

    musaEvent_t syncPoint;
    checkMusaErrors(musaEventCreate(&syncPoint));

    for (int i = 0; i < 100; ++i) {
        checkMuErrors(muLaunchKernel(function, 1, 1, 1, /* grid dim */
            1, 1, 1,                                    /* block dim */
            0,                                          /* shared mem */
            stream,                                     /* stream */
            nullptr,                                    /* params */
            nullptr                                     /* extra */
            ));
    }

    checkMusaErrors(musaDeviceSynchronize());
    checkMusaErrors(musaEventRecord(start, stream));
    for (int i = 0; i < launchCount; i++) {
        checkMuErrors(muLaunchKernel(function, 1, 1, 1, /* grid dim */
            1, 1, 1,                                    /* block dim */
            0,                                          /* shared mem */
            stream,                                     /* stream */
            nullptr,                                    /* params */
            nullptr                                     /* extra */
            ));
        if (bRecord) {
            checkMusaErrors(musaEventRecord(syncPoint, stream));
        }
    }
    checkMusaErrors(musaEventRecord(stop, stream));
    checkMusaErrors(musaEventSynchronize(stop));
    checkMusaErrors(musaEventElapsedTime(pDurationMs, start, stop));

    checkMusaErrors(musaEventDestroy(start));
    checkMusaErrors(musaEventDestroy(stop));
}

static const int SamplesCount    = 20;
static const int IterationsCount = 2;
static const int LaunchTimes     = 20;

BASELINE_F(mulCards, nullkernel, MulCardsLanchFixture, SamplesCount, IterationsCount) {
    std::thread* threads = new std::thread[m_ThreadsCounts];
    float* pDurationMs   = new float[m_ThreadsCounts];
    for (int i = 0; i < m_ThreadsCounts; i++) {
        threads[i] =
            std::thread(&MulCardsLanchFixture::threadLaunchFunc, this, i, i, false, LaunchTimes, &pDurationMs[i]);
    }
    for (int i = 0; i < m_ThreadsCounts; i++) {
        threads[i].join();
    }
    for (int i = 0; i < m_ThreadsCounts; i++) {
        this->utimeTotal->addValue(pDurationMs[i]);
        this->utimeAvgPer->addValue(pDurationMs[i] * 1000.0 / LaunchTimes);
        this->utp->addValue(LaunchTimes / (pDurationMs[i] / 1000));
        this->uCount->addValue(LaunchTimes);
    }

    delete[] threads;
    delete[] pDurationMs;
}

BENCHMARK_F(mulCards, nullAndRecord, MulCardsLanchFixture, SamplesCount, IterationsCount) {
    std::thread* threads = new std::thread[m_ThreadsCounts];
    float* pDurationMs   = new float[m_ThreadsCounts];
    for (int i = 0; i < m_ThreadsCounts; i++) {
        threads[i] = std::thread(&MulCardsLanchFixture::threadLaunchFunc, this, i, i, true, LaunchTimes, &pDurationMs[i]);
    }
    for (int i = 0; i < m_ThreadsCounts; i++) {
        threads[i].join();
    }
    for (int i = 0; i < m_ThreadsCounts; i++) {
        this->utimeTotal->addValue(pDurationMs[i]);
        this->utimeAvgPer->addValue(pDurationMs[i] * 1000.0 / LaunchTimes);
        this->utp->addValue(LaunchTimes / (pDurationMs[i] / 1000));
        this->uCount->addValue(LaunchTimes);
    }

    delete[] threads;
    delete[] pDurationMs;
}
