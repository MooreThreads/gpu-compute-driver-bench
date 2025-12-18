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

static const int SamplesCount    = 3;
static const int IterationsCount = 1;
static const int GraphNodesCount = 1024;
static const int launchCount     = 1000;

static __global__ void emptyKernel() {
}

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
        for (int i = 1; i <= GraphNodesCount; i *= 2) {
            problemSpace.push_back(i);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override { m_M = experimentValue.Value; }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->utimeTotal, this->utp};
    }

    musaError_t graphLaunchLatency(uint64_t m, uint64_t n, uint64_t launchCount, float* totalTime);

    musaEvent_t start;
    musaEvent_t stop;
    uint64_t m_M;

    std::shared_ptr<UDMGPUTime> utimeTotal{new UDMGPUTime("tAll(ms)", StatsView::MIN | StatsView::MAX)};
    std::shared_ptr<UDMThroughPut> utp{new UDMThroughPut("*TP(s^-1)")};
};

musaError_t LanchFixture::graphLaunchLatency(uint64_t m, uint64_t n, uint64_t launchCount, float* totalTime) {
    const int M = m;
    const int N = n;
    const int K = launchCount;

    musaStream_t stream;
    checkMusaErrors(musaStreamCreate(&stream));

    musaGraph_t graph;
    checkMusaErrors(musaGraphCreate(&graph, 0));

    std::vector<musaGraphNode_t> nodes(M * N);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            std::vector<musaGraphNode_t> deps;
            if (m > 0) {
                for (int k = 0; k < N; k++) {
                    deps.push_back(nodes[(m - 1) * N + k]);
                }
            }

            musaKernelNodeParams params{};
            params.func           = (void*)emptyKernel;
            params.gridDim        = dim3(1);
            params.blockDim       = dim3(1);
            params.sharedMemBytes = 0;
            params.kernelParams   = nullptr;
            params.extra          = nullptr;

            checkMusaErrors(musaGraphAddKernelNode(
                &nodes[m * N + n], graph, deps.empty() ? nullptr : deps.data(), deps.size(), &params));
        }
    }

    // for debug topo of graph
    // checkMusaErrors(musaGraphDebugDotPrint(
    //     graph,
    //     "graph.dot",
    //     musaGraphDebugDotFlagsVerbose));

    // std::cout << "DOT exported to graph.dot\n";

    musaGraphExec_t graphExec;
    checkMusaErrors(musaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    for (int i = 0; i < 10; i++)
        checkMusaErrors(musaGraphLaunch(graphExec, stream));

    checkMusaErrors(musaStreamSynchronize(stream));

    CPerfCounter timer;
    timer.Start();
    for (int i = 0; i < K; i++) {
        checkMusaErrors(musaGraphLaunch(graphExec, stream));
    }
    checkMusaErrors(musaStreamSynchronize(stream));
    timer.Stop();

    double ms            = timer.GetElapsedSeconds() * 1000;
    double us_per_launch = (ms * 1000.0) / K;

    // std::cout << "Launch latency per graph: " << us_per_launch << " us" << std::endl;
    *totalTime = static_cast<float>(ms);

    checkMusaErrors(musaGraphDestroy(graph));
    checkMusaErrors(musaGraphExecDestroy(graphExec));
    checkMusaErrors(musaStreamDestroy(stream));

    return musaSuccess;
}

BASELINE_F(launchGraph, nodes1024, LanchFixture, SamplesCount, IterationsCount) {
    float totalTime    = 0.0f;
    musaError_t status = this->graphLaunchLatency(this->m_M, GraphNodesCount / m_M, launchCount, &totalTime);

    utimeTotal->addValue(totalTime / launchCount);
    utp->addValue(1000 * launchCount / totalTime);
}
