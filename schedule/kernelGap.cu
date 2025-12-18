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

// delay timeout_clocks's tick-tock 12.5M hz in S50?
__global__ void delay(volatile int* flag, unsigned long long timeout_clocks = 100000000) {
    long long int start_clock, sample_clock;
    start_clock = clock64();
    while (*flag) {
        sample_clock = clock64();
        if (sample_clock - start_clock > timeout_clocks) {
            break;
        }
    }
}

std::vector<double> xin;
std::vector<double> yout;

int main(int argc, char** argv) {
    int deviceCount;
    checkMusaErrors(musaGetDeviceCount(&deviceCount));
    musaDeviceProp prop;
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    Printer::get().TableSetPbName("lanchTimes:N");
    Run(argc, argv);
    Eigen::VectorXd x(xin.size());
    Eigen::VectorXd y(yout.size());
    for (int i = 0; i < xin.size(); ++i) {
        x(i) = xin[i];
        y(i) = yout[i];
    }

    Eigen::MatrixXd A(x.size(), 2);
    A.col(0)                     = x;
    A.col(1)                     = Eigen::VectorXd::Ones(x.size());
    Eigen::VectorXd coefficients = A.householderQr().solve(y);
    console::SetConsoleColor(console::ConsoleColor::Red);
    std::cout << "fit curve:t = " << coefficients(0) << "n + " << coefficients(1) << std::endl;
    printf("that means the gap between two kernels is %f us\n", coefficients(0));
    return 0;
}

class LanchFixture : public TestFixture {
public:
    LanchFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        int cnt1 = 1;
        int cnt5 = 5;
        for (int64_t i = 0; i < 4; ++i) {
            problemSpace.push_back(cnt1);
            problemSpace.push_back(cnt5);
            cnt1 *= 10;
            cnt5 *= 10;
        }
        problemSpace.push_back(cnt1);
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override { launchTimes = experimentValue.Value; }

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

    uint64_t launchTimes;
    musaEvent_t start;
    musaEvent_t stop;
    std::shared_ptr<UDMGPUTime> utime{new UDMGPUTime("t(us)", StatsView::MIN | StatsView::MAX)};
};

BASELINE_F(launchKernel, delayKernel, LanchFixture, 1, 1) {
    CPerfCounter timer;
    int* flag;
    musaMalloc(&flag, 4);
    musaMemset((void*)flag, 1, 4);
    const int tickNum = 12500000 / launchTimes;
    musaDeviceSynchronize();
    delay<<<1, 1>>>(flag, tickNum); // warmup

    float t;
    checkMusaErrors(musaEventRecord(start));
    for (uint64_t i = 0; i < launchTimes; ++i) {
        delay<<<1, 1>>>(flag, tickNum);
    }
    checkMusaErrors(musaEventRecord(stop));
    checkMusaErrors(musaEventSynchronize(stop));
    checkMusaErrors(musaEventElapsedTime(&t, start, stop));

    xin.push_back(static_cast<double>(launchTimes));
    yout.push_back(1000 * t);

    utime->addValue(1000 * t);
}
