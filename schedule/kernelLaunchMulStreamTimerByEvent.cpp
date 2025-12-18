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
    Printer::get().TableSetPbName("StreamNumber");
    Run(argc, argv);
    return 0;
}

class MulStreamsLanchFixture : public TestFixture {
public:
    MulStreamsLanchFixture() { checkMusaErrors(musaGetDeviceCount(&m_TotalCards)); }
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (uint32_t i = 1; i <= 32; i <<= 1) {
            problemSpace.push_back(i);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
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

void MulStreamsLanchFixture::threadLaunchFunc(
    int32_t tid, int32_t cid, bool bRecord, int launchCount, float* pDurationMs) {
    checkMusaErrors(musaSetDevice(cid));

    MUmodule module;
    MUfunction function;

    musaEvent_t start;
    musaEvent_t stop;
    checkMusaErrors(musaEventCreate(&start));
    checkMusaErrors(musaEventCreate(&stop));

    musaStream_t stream = nullptr;
    checkMusaErrors(musaStreamCreate(&stream));

    musaEvent_t syncPoint;
    checkMusaErrors(musaEventCreate(&syncPoint));

    checkMuErrors(muModuleLoad(&module, codeObjectFile.c_str()));
    checkMuErrors(muModuleGetFunction(&function, module, functionName.c_str()));

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

    checkMusaErrors(musaStreamDestroy(stream));
    checkMusaErrors(musaEventDestroy(syncPoint));
}

static const int SamplesCount    = 20;
static const int IterationsCount = 1;
static const int launchTimes     = 32;

#define BENCH_NULL_KERNEL(times)                                                                             \
    BENCHMARK_F(mulStreams, nullkernelx##times, MulStreamsLanchFixture, SamplesCount, IterationsCount) {     \
        int launchTimes      = times;                                                                        \
        std::thread* threads = new std::thread[m_ThreadsCounts];                                             \
        float* pDurationMs   = new float[m_ThreadsCounts];                                                   \
        for (int i = 0; i < m_ThreadsCounts; i++) {                                                          \
            threads[i] = std::thread(                                                                        \
                &MulStreamsLanchFixture::threadLaunchFunc, this, i, 0, false, launchTimes, &pDurationMs[i]); \
        }                                                                                                    \
        for (int i = 0; i < m_ThreadsCounts; i++) {                                                          \
            threads[i].join();                                                                               \
        }                                                                                                    \
        for (int i = 0; i < m_ThreadsCounts; i++) {                                                          \
            this->utimeTotal->addValue(pDurationMs[i]);                                                      \
            this->utimeAvgPer->addValue(pDurationMs[i] * 1000.0 / launchTimes);                              \
            this->utp->addValue(launchTimes / (pDurationMs[i] / 1000));                                      \
            this->uCount->addValue(launchTimes);                                                             \
        }                                                                                                    \
                                                                                                             \
        delete[] threads;                                                                                    \
        delete[] pDurationMs;                                                                                \
    }

#define BENCH_NULL_RECORD(times)                                                                                      \
    BENCHMARK_F(mulStreams, nullAndRecordx##times, MulStreamsLanchFixture, SamplesCount, IterationsCount) {           \
        int launchTimes      = times;                                                                                 \
        std::thread* threads = new std::thread[m_ThreadsCounts];                                                      \
        float* pDurationMs   = new float[m_ThreadsCounts];                                                            \
        for (int i = 0; i < m_ThreadsCounts; i++) {                                                                   \
            threads[i] = std::thread(                                                                                 \
                &MulStreamsLanchFixture::threadLaunchFunc, this, launchTimes, 0, true, launchTimes, &pDurationMs[i]); \
        }                                                                                                             \
        for (int i = 0; i < m_ThreadsCounts; i++) {                                                                   \
            threads[i].join();                                                                                        \
        }                                                                                                             \
        for (int i = 0; i < m_ThreadsCounts; i++) {                                                                   \
            this->utimeTotal->addValue(pDurationMs[i]);                                                               \
            this->utimeAvgPer->addValue(pDurationMs[i] * 1000.0 / launchTimes);                                       \
            this->utp->addValue(launchTimes / (pDurationMs[i] / 1000));                                               \
            this->uCount->addValue(launchTimes);                                                                      \
        }                                                                                                             \
                                                                                                                      \
        delete[] threads;                                                                                             \
        delete[] pDurationMs;                                                                                         \
    }

BASELINE_F(mulStreams, nullkernelx1, MulStreamsLanchFixture, SamplesCount, IterationsCount) {
    int launchTimes      = 1;
    std::thread* threads = new std::thread[m_ThreadsCounts];
    float* pDurationMs   = new float[m_ThreadsCounts];
    for (int i = 0; i < m_ThreadsCounts; i++) {
        threads[i] =
            std::thread(&MulStreamsLanchFixture::threadLaunchFunc, this, i, 0, false, launchTimes, &pDurationMs[i]);
    }
    for (int i = 0; i < m_ThreadsCounts; i++) {
        threads[i].join();
    }
    for (int i = 0; i < m_ThreadsCounts; i++) {
        this->utimeTotal->addValue(pDurationMs[i]);
        this->utimeAvgPer->addValue(pDurationMs[i] * 1000.0 / launchTimes);
        this->utp->addValue(launchTimes / (pDurationMs[i] / 1000));
        this->uCount->addValue(launchTimes);
    }

    delete[] threads;
    delete[] pDurationMs;
}

BENCH_NULL_KERNEL(10);
BENCH_NULL_KERNEL(100);
BENCH_NULL_KERNEL(1000);

BENCH_NULL_RECORD(1);
BENCH_NULL_RECORD(10);
BENCH_NULL_RECORD(100);
BENCH_NULL_RECORD(1000);