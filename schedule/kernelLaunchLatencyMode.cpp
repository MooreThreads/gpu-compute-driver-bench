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

static const int SamplesCount    = 1;
static const int IterationsCount = 1;
static const int N               = 1000;

int main(int argc, char** argv) {
    int deviceCount;
    checkMusaErrors(musaGetDeviceCount(&deviceCount));
    musaDeviceProp prop;
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    Printer::get().TableSetPbName("counts");
    Run(argc, argv);
    return 0;
}

class LanchFixture : public TestFixture {
public:
    LanchFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        problemSpace.push_back(1 << 18);
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        functionName = std::string("_Z11EmptyKernelv");
        launchTime   = experimentValue.Value;
    }

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
        return {this->utime, this->utime1, this->utp};
    }
#ifdef TEST_ON_NVIDIA
    std::string codeObjectFile = std::string("./EmptyKernel.ptx");
#else
    std::string codeObjectFile = std::string("./EmptyKernel.elf");
#endif
    uint64_t launchTime;
    std::string functionName = std::string("_Z11EmptyKernelv");
    musaEvent_t start;
    musaEvent_t stop;
    std::shared_ptr<UDMGPUTime> utime{new UDMGPUTime("sum(ms)")};
    std::shared_ptr<UDMGPUTime> utime1{new UDMGPUTime("per(us)")};
    std::shared_ptr<UDMThroughPut> utp{new UDMThroughPut("*TP(s^-1)")};
};

BASELINE_F(launchKernel, nullkernel, LanchFixture, SamplesCount, IterationsCount) {
    musaStream_t streamIdx;
    checkMusaErrors(musaStreamCreate(&streamIdx));

    MUmodule module;
    MUfunction function;
    checkMuErrors(muModuleLoad(&module, codeObjectFile.c_str()));
    checkMuErrors(muModuleGetFunction(&function, module, functionName.c_str()));

    /*warm up*/
    for (int i = 0; i < 100; ++i) {
        checkMuErrors(muLaunchKernel(function, 1, 1, 1, /* grid dim */
            1, 1, 1,                                    /* block dim */
            0,                                          /* shared mem */
            streamIdx,                                  /* stream */
            nullptr,                                    /* params */
            nullptr                                     /* extra */
            ));
    }
    checkMusaErrors(musaDeviceSynchronize());
    float milliseconds       = 0.f;
    float total_milliseconds = 0.f;
    for (size_t i = 0; i < 1; ++i) {
        CPerfCounter timer;
        timer.Restart();
        for (int i = 0; i < launchTime; ++i) {
            checkMuErrors(muLaunchKernel(function, 1, 1, 1, /* grid dim */
                1, 1, 1,                                    /* block dim */
                0,                                          /* shared mem */
                streamIdx,                                  /* stream */
                nullptr,                                    /* params */
                nullptr                                     /* extra */
                ));
            checkMusaErrors(musaStreamSynchronize(streamIdx));
        }
        timer.Stop();
        milliseconds = timer.GetElapsedSeconds() * 1000.f;
        this->utime->addValue(milliseconds);
        this->utime1->addValue(milliseconds * 1000.f / launchTime);
        this->utp->addValue(launchTime / (milliseconds / 1000));
    }
    checkMusaErrors(musaStreamDestroy(streamIdx));
}
