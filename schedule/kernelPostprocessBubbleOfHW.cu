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

#include <vector>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <errno.h>
#include <string>
#include <Eigen/Dense>

#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "musa.h"
#include "musa_runtime.h"
#include "timer_his.h"
#include "helper_musa.h"
#include "helper_musa_drvapi.h"

__global__ void
delay(volatile int* flag, unsigned long long timeout_clocks = 100000000, long long int* duration = nullptr) {
    long long int start_clock, sample_clock;
    start_clock = clock64();
    while (*flag) {
        sample_clock = clock64();
        if (sample_clock - start_clock > timeout_clocks) {
            *duration = sample_clock - start_clock;
            break;
        }
    }
}

__global__ void nullkernel() {
}

int main(int argc, char** argv) {
    int deviceCount;
    checkMusaErrors(musaGetDeviceCount(&deviceCount));
    musaDeviceProp prop;
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    Printer::get().TableSetPbName("kernelRunningTime");
    Run(argc, argv);
    return 0;
}

class LanchFixture : public TestFixture {
public:
    LanchFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        problemSpace.push_back(1);
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {}

    void tearDown() override {}
    void onExperimentStart(const TestFixture::ExperimentValue& x) override {
        checkMusaErrors(musaEventCreate(&start));
        checkMusaErrors(musaEventCreate(&stop));
    }
    void onExperimentEnd() override {
        checkMusaErrors(musaEventDestroy(start));
        checkMusaErrors(musaEventDestroy(stop));
    }

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->utime};
    }

    musaEvent_t start;
    musaEvent_t stop;
    std::shared_ptr<UDMGPUTime> utime{new UDMGPUTime("t(us)", StatsView::MIN | StatsView::MAX | StatsView::STDDEV)};
};

BASELINE_F(launchKernel, delayKernel, LanchFixture, 10, 1) {
    CPerfCounter timer;
    int* flag;
    long long int* dduration;
    long long int hduration;

    musaMalloc(&flag, 4);
    musaMalloc(&dduration, sizeof(long long int));
    musaMemset((void*)flag, 1, 4);
    musaMemset((void*)flag, 1, 4);
    // const int tickNum = 1700000000; // 1700M in ph1

    const int tickNum = 1300000000;

    checkMusaErrors(musaEventRecord(start));
    delay<<<1, 1>>>(flag, tickNum / 1000, dduration); // warmup
    for (int it = 0; it < 500; ++it) {
        nullkernel<<<1, 1>>>();
    }
    checkMusaErrors(musaEventRecord(stop));
    musaDeviceSynchronize();

    float t1;
    checkMusaErrors(musaEventRecord(start));

    delay<<<1, 1>>>(flag, tickNum, dduration);

    checkMusaErrors(musaEventRecord(stop));
    checkMusaErrors(musaEventSynchronize(stop));
    checkMusaErrors(musaEventElapsedTime(&t1, start, stop));

    checkMusaErrors(musaMemcpy(&hduration, dduration, sizeof(long long int), musaMemcpyDeviceToHost));
    // printf("clock of t1: %f ms\n", t1);
    float t2 = (float(hduration) / tickNum) * 1000;
    // printf("clock of t2: %f ms\n", t2);
    // printf("clock of t3: %f us\n", 1000 * (t1-t2));
    utime->addValue(1000 * (t1 - t2));
}
