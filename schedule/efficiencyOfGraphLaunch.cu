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

#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "helper_musa.h"
#include "helper_musa_drvapi.h"
#include "musa.h"
#include "musa_runtime.h"
#include "timer_his.h"

__global__ void emptyKernel() {
}

__global__ void copyKernel(int* b, const int* a, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < n)
        b[gtid] = a[gtid];
}

__global__ void addOneKernel(int* a, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < n)
        a[gtid] = a[gtid] + 1;
}

#define BLOCK_SIZE 256
unsigned int getGridSize(unsigned int size) {
    return (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

int main(int argc, char** argv) {
    int deviceCount;
    checkMusaErrors(musaGetDeviceCount(&deviceCount));
    musaDeviceProp prop;
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    Printer::get().TableSetPbName("size/1");
    Run(argc, argv);
    return 0;
}

class GraphLaunchFixture : public TestFixture {
public:
    GraphLaunchFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int i = 6; i < 16; i += 2) {
            problemSpace.push_back(1 << i);
        }
        return problemSpace;
    }
    void setUp(const TestFixture::ExperimentValue& experimentValue) override { m_Size = experimentValue.Value; }
    void tearDown() override {}
    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->kernelCount, this->exeTime, this->exeBand};
    }

    musaError_t graphLaunch(unsigned int size, int kernelCount, long long* tElapsedTime);
    musaError_t kernelLaunch(unsigned int size, int kernelCount, long long* tElapsedTime);
    musaError_t graphApiLaunch(unsigned int size, int kernelCount, long long* tElapsedTime);
    musaError_t emptyKernelGraphLaunch(unsigned int size, int kernelCount, long long* tElapsedTime);
    musaError_t graphAndKernel(unsigned int size, int kernelCount, long long* tElapsedTime);
    musaError_t graphAfterKernel(unsigned int size, int kernelCount, long long* tElapsedTime);
    musaError_t graphBeforeKernel(unsigned int size, int kernelCount, long long* tElapsedTime);

    int getErrorSize(const int* const c, const int* const a, const int* const b, unsigned int size) {
        int errorSize = 0;
        for (int i = 0; i < size; ++i) {
            if (c[i] != a[i] + b[i]) {
                errorSize++;
                printf("c[%d] = %d, a[%d] = %d, b[%d] = %d\n", i, c[i], i, a[i], i, b[i]);
            }
        }
        return errorSize;
    }
    size_t m_Size;
    std::shared_ptr<UDMCount> kernelCount{new UDMCount("kernelCount")};
    std::shared_ptr<UDMGPUTime> exeTime{new UDMGPUTime("gr(us)")};
    std::shared_ptr<UDMBandWidth> exeBand{new UDMBandWidth("*gr(MB/s)")};
};

musaError_t GraphLaunchFixture::graphLaunch(unsigned int size, int kernelCount, long long* tElapsedTime) {
    int* host_a = (int*)malloc(size * sizeof(int));
    int* host_b = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        host_a[i] = Random() % 255;
        host_b[i] = 0;
    }

    int* dev_a = 0;
    int* dev_b = 0;
    checkMusaErrors(musaMalloc((void**)&dev_a, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_b, size * sizeof(int)));

    musaStream_t captureStream, runStream;
    checkMusaErrors(musaStreamCreate(&captureStream));
    checkMusaErrors(musaStreamCreate(&runStream));

    checkMusaErrors(musaStreamBeginCapture(captureStream, musaStreamCaptureModeGlobal));
    uint32_t threads = BLOCK_SIZE;
    uint32_t blocks  = getGridSize(size);

    copyKernel<<<blocks, threads, 0, captureStream>>>(dev_b, dev_a, size);
    for (int i = 0; i < kernelCount; i++) {
        addOneKernel<<<blocks, threads, 0, captureStream>>>(dev_b, size);
    }

    musaGraph_t graph = NULL;
    checkMusaErrors(musaStreamEndCapture(captureStream, &graph));

    musaGraphExec_t graphExec;
    checkMusaErrors(musaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    checkMusaErrors(musaMemcpyAsync(dev_a, host_a, size * sizeof(int), musaMemcpyHostToDevice, runStream));
    musaStreamSynchronize(runStream);

    int loopCount = 1000;
    CPerfCounter timer;
    timer.Start();
    for (int i = 0; i < loopCount; ++i) {
        // graph
        checkMusaErrors(musaGraphLaunch(graphExec, runStream));
    }
    checkMusaErrors(musaStreamSynchronize(runStream));
    timer.Stop();

    *tElapsedTime = timer.GetElapsedSeconds() * 1000 / loopCount;

    checkMusaErrors(musaMemcpyAsync(host_b, dev_b, size * sizeof(int), musaMemcpyDeviceToHost, runStream));
    musaStreamSynchronize(runStream);

    for (int i = 0; i < size; i++) {
        if (host_b[i] != (host_a[i] + kernelCount)) {
            printf("graphLaunch check failed, i %d, hb=%d, ha=%d\n", i, host_b[i], host_a[i]);
            exit(EXIT_FAILURE);
        }
    }

    checkMusaErrors(musaGraphExecDestroy(graphExec));
    checkMusaErrors(musaGraphDestroy(graph));

    checkMusaErrors(musaStreamDestroy(captureStream));
    checkMusaErrors(musaStreamDestroy(runStream));

    checkMusaErrors(musaFree(dev_a));
    checkMusaErrors(musaFree(dev_b));
    free(host_a);
    free(host_b);
    return musaSuccess;
}

musaError_t GraphLaunchFixture::kernelLaunch(unsigned int size, int kernelCount, long long* tElapsedTime) {
    int* host_a = (int*)malloc(size * sizeof(int));
    int* host_b = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        host_a[i] = Random() % 255;
        host_b[i] = 0;
    }

    int* dev_a = 0;
    int* dev_b = 0;
    checkMusaErrors(musaMalloc((void**)&dev_a, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_b, size * sizeof(int)));

    musaStream_t runStream;
    checkMusaErrors(musaStreamCreate(&runStream));

    checkMusaErrors(musaMemcpyAsync(dev_a, host_a, size * sizeof(int), musaMemcpyHostToDevice, runStream));
    musaStreamSynchronize(runStream);

    uint32_t threads = BLOCK_SIZE;
    uint32_t blocks  = getGridSize(size);
    int loopCount    = 1000;
    CPerfCounter timer;
    timer.Start();
    for (int i = 0; i < loopCount; ++i) {
        copyKernel<<<blocks, threads, 0, runStream>>>(dev_b, dev_a, size);

        for (int i = 0; i < kernelCount; i++) {
            addOneKernel<<<blocks, threads, 0, runStream>>>(dev_b, size);
        }
    }
    checkMusaErrors(musaStreamSynchronize(runStream));
    timer.Stop();

    *tElapsedTime = timer.GetElapsedSeconds() * 1000 / loopCount;

    checkMusaErrors(musaMemcpyAsync(host_b, dev_b, size * sizeof(int), musaMemcpyDeviceToHost, runStream));
    musaStreamSynchronize(runStream);

    for (int i = 0; i < size; i++) {
        if (host_b[i] != (host_a[i] + kernelCount)) {
            printf("kernelLaunch check failed, i %d, hb=%d, ha=%d\n", i, host_b[i], host_a[i]);
            exit(EXIT_FAILURE);
        }
    }

    checkMusaErrors(musaStreamDestroy(runStream));

    checkMusaErrors(musaFree(dev_a));
    checkMusaErrors(musaFree(dev_b));
    free(host_a);
    free(host_b);
    return musaSuccess;
}

musaError_t GraphLaunchFixture::graphApiLaunch(unsigned int size, int kernelCount, long long* tElapsedTime) {
    int* host_a = (int*)malloc(size * sizeof(int));
    int* host_b = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        host_a[i] = Random() % 255;
        host_b[i] = 0;
    }

    int* dev_a = 0;
    int* dev_b = 0;
    checkMusaErrors(musaMalloc((void**)&dev_a, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_b, size * sizeof(int)));

    musaStream_t captureStream, runStream;
    checkMusaErrors(musaStreamCreate(&captureStream));
    checkMusaErrors(musaStreamCreate(&runStream));

    checkMusaErrors(musaStreamBeginCapture(captureStream, musaStreamCaptureModeGlobal));
    uint32_t threads = BLOCK_SIZE;
    uint32_t blocks  = getGridSize(size);

    copyKernel<<<blocks, threads, 0, captureStream>>>(dev_b, dev_a, size);
    for (int i = 0; i < kernelCount; i++) {
        addOneKernel<<<blocks, threads, 0, captureStream>>>(dev_b, size);
    }

    musaGraph_t graph = NULL;
    checkMusaErrors(musaStreamEndCapture(captureStream, &graph));

    musaGraphExec_t graphExec;
    checkMusaErrors(musaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    checkMusaErrors(musaMemcpyAsync(dev_a, host_a, size * sizeof(int), musaMemcpyHostToDevice, runStream));
    musaStreamSynchronize(runStream);

    int loopCount = 1000;
    CPerfCounter timer;
    timer.Start();
    for (int i = 0; i < loopCount; ++i) {
        checkMusaErrors(musaGraphLaunch(graphExec, runStream));
    }
    timer.Stop();
    checkMusaErrors(musaStreamSynchronize(runStream));

    *tElapsedTime = timer.GetElapsedSeconds() * 1000 * 1000 / loopCount;

    checkMusaErrors(musaMemcpyAsync(host_b, dev_b, size * sizeof(int), musaMemcpyDeviceToHost, runStream));
    musaStreamSynchronize(runStream);

    for (int i = 0; i < size; i++) {
        if (host_b[i] != (host_a[i] + kernelCount)) {
            printf("graphApiLaunch check failed, i %d, hb=%d, ha=%d\n", i, host_b[i], host_a[i]);
            exit(EXIT_FAILURE);
        }
    }

    checkMusaErrors(musaGraphExecDestroy(graphExec));
    checkMusaErrors(musaGraphDestroy(graph));

    checkMusaErrors(musaStreamDestroy(captureStream));
    checkMusaErrors(musaStreamDestroy(runStream));

    checkMusaErrors(musaFree(dev_a));
    checkMusaErrors(musaFree(dev_b));
    free(host_a);
    free(host_b);
    return musaSuccess;
}

musaError_t GraphLaunchFixture::emptyKernelGraphLaunch(unsigned int size, int kernelCount, long long* tElapsedTime) {
    musaStream_t captureStream, runStream;
    checkMusaErrors(musaStreamCreate(&captureStream));
    checkMusaErrors(musaStreamCreate(&runStream));

    checkMusaErrors(musaStreamBeginCapture(captureStream, musaStreamCaptureModeGlobal));
    uint32_t threads = BLOCK_SIZE;
    uint32_t blocks  = getGridSize(size);

    for (int i = 0; i < kernelCount; i++) {
        emptyKernel<<<blocks, threads, 0, captureStream>>>();
    }

    musaGraph_t graph = NULL;
    checkMusaErrors(musaStreamEndCapture(captureStream, &graph));

    musaGraphExec_t graphExec;
    checkMusaErrors(musaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    int loopCount = 1000;
    CPerfCounter timer;
    timer.Start();
    for (int i = 0; i < loopCount; ++i) {
        checkMusaErrors(musaGraphLaunch(graphExec, runStream));
    }
    checkMusaErrors(musaStreamSynchronize(runStream));
    timer.Stop();

    *tElapsedTime = timer.GetElapsedSeconds() * 1000 / loopCount;

    checkMusaErrors(musaGraphExecDestroy(graphExec));
    checkMusaErrors(musaGraphDestroy(graph));

    checkMusaErrors(musaStreamDestroy(captureStream));
    checkMusaErrors(musaStreamDestroy(runStream));
    return musaSuccess;
}

musaError_t GraphLaunchFixture::graphAndKernel(unsigned int size, int kernelCount, long long* tElapsedTime) {
    int* host_a = (int*)malloc(size * sizeof(int));
    int* host_b = (int*)malloc(size * sizeof(int));
    int* host_c = (int*)malloc(size * sizeof(int));
    int* host_d = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        host_a[i] = Random() % 255;
        host_b[i] = 0;
        host_c[i] = Random() % 255;
        host_d[i] = 0;
    }

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    int* dev_d = 0;
    checkMusaErrors(musaMalloc((void**)&dev_a, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_b, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_c, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_d, size * sizeof(int)));

    musaStream_t captureStream, runStream1, runStream2;
    checkMusaErrors(musaStreamCreate(&captureStream));
    checkMusaErrors(musaStreamCreate(&runStream1));
    checkMusaErrors(musaStreamCreate(&runStream2));

    checkMusaErrors(musaStreamBeginCapture(captureStream, musaStreamCaptureModeGlobal));
    uint32_t threads = BLOCK_SIZE;
    uint32_t blocks  = getGridSize(size);

    copyKernel<<<blocks, threads, 0, captureStream>>>(dev_b, dev_a, size);
    for (int i = 0; i < kernelCount; i++) {
        addOneKernel<<<blocks, threads, 0, captureStream>>>(dev_b, size);
    }

    musaGraph_t graph = NULL;
    checkMusaErrors(musaStreamEndCapture(captureStream, &graph));

    musaGraphExec_t graphExec;
    checkMusaErrors(musaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    checkMusaErrors(musaMemcpyAsync(dev_a, host_a, size * sizeof(int), musaMemcpyHostToDevice, runStream1));
    checkMusaErrors(musaMemcpyAsync(dev_c, host_c, size * sizeof(int), musaMemcpyHostToDevice, runStream2));
    checkMusaErrors(musaStreamSynchronize(runStream1));
    checkMusaErrors(musaStreamSynchronize(runStream2));

    int loopCount = 1000;
    CPerfCounter timer;
    timer.Start();
    for (int i = 0; i < loopCount; ++i) {
        checkMusaErrors(musaGraphLaunch(graphExec, runStream1));

        copyKernel<<<blocks, threads, 0, runStream2>>>(dev_d, dev_c, size);
        for (int i = 0; i < kernelCount; i++) {
            addOneKernel<<<blocks, threads, 0, runStream2>>>(dev_d, size);
        }
    }
    checkMusaErrors(musaStreamSynchronize(runStream1));
    timer.Stop();
    checkMusaErrors(musaStreamSynchronize(runStream2));

    *tElapsedTime = timer.GetElapsedSeconds() * 1000 / loopCount;

    checkMusaErrors(musaMemcpyAsync(host_b, dev_b, size * sizeof(int), musaMemcpyDeviceToHost, runStream1));
    musaStreamSynchronize(runStream1);
    checkMusaErrors(musaMemcpyAsync(host_d, dev_d, size * sizeof(int), musaMemcpyDeviceToHost, runStream2));
    musaStreamSynchronize(runStream2);

    for (int i = 0; i < size; i++) {
        if (host_b[i] != (host_a[i] + kernelCount)) {
            printf("graphAndKernel graph check failed, i %d, hb=%d, ha=%d\n", i, host_b[i], host_a[i]);
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < size; i++) {
        if (host_d[i] != (host_c[i] + kernelCount)) {
            printf("graphAndKernel kernel check failed, i %d, hb=%d, ha=%d\n", i, host_d[i], host_c[i]);
            exit(EXIT_FAILURE);
        }
    }

    checkMusaErrors(musaGraphExecDestroy(graphExec));
    checkMusaErrors(musaGraphDestroy(graph));

    checkMusaErrors(musaStreamDestroy(captureStream));
    checkMusaErrors(musaStreamDestroy(runStream1));
    checkMusaErrors(musaStreamDestroy(runStream2));

    checkMusaErrors(musaFree(dev_a));
    checkMusaErrors(musaFree(dev_b));
    checkMusaErrors(musaFree(dev_c));
    checkMusaErrors(musaFree(dev_d));
    free(host_a);
    free(host_b);
    free(host_c);
    free(host_d);
    return musaSuccess;
}

musaError_t GraphLaunchFixture::graphAfterKernel(unsigned int size, int kernelCount, long long* tElapsedTime) {
    int* host_a = (int*)malloc(size * sizeof(int));
    int* host_b = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        host_a[i] = Random() % 255;
        host_b[i] = 0;
    }

    int* dev_a = 0;
    int* dev_b = 0;
    checkMusaErrors(musaMalloc((void**)&dev_a, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_b, size * sizeof(int)));

    musaStream_t captureStream, runStream;
    checkMusaErrors(musaStreamCreate(&captureStream));
    checkMusaErrors(musaStreamCreate(&runStream));

    checkMusaErrors(musaStreamBeginCapture(captureStream, musaStreamCaptureModeGlobal));
    uint32_t threads = BLOCK_SIZE;
    uint32_t blocks  = getGridSize(size);

    for (int i = 0; i < (kernelCount / 2); i++) {
        addOneKernel<<<blocks, threads, 0, captureStream>>>(dev_b, size);
    }

    musaGraph_t graph = NULL;
    checkMusaErrors(musaStreamEndCapture(captureStream, &graph));

    musaGraphExec_t graphExec;
    checkMusaErrors(musaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    checkMusaErrors(musaMemcpyAsync(dev_a, host_a, size * sizeof(int), musaMemcpyHostToDevice, runStream));

    int loopCount = 1000;
    CPerfCounter timer;
    timer.Start();
    for (int i = 0; i < loopCount; ++i) {
        copyKernel<<<blocks, threads, 0, runStream>>>(dev_b, dev_a, size);

        for (int i = 0; i < (kernelCount / 2); i++) {
            addOneKernel<<<blocks, threads, 0, runStream>>>(dev_b, size);
        }

        checkMusaErrors(musaGraphLaunch(graphExec, runStream));
    }
    checkMusaErrors(musaStreamSynchronize(runStream));
    timer.Stop();

    *tElapsedTime = timer.GetElapsedSeconds() * 1000 / loopCount;

    checkMusaErrors(musaMemcpyAsync(host_b, dev_b, size * sizeof(int), musaMemcpyDeviceToHost, runStream));
    musaStreamSynchronize(runStream);

    for (int i = 0; i < size; i++) {
        if (host_b[i] != (host_a[i] + kernelCount)) {
            printf("graphAfterKernel check failed, i %d, hb=%d, ha=%d\n", i, host_b[i], host_a[i]);
            exit(EXIT_FAILURE);
        }
    }

    checkMusaErrors(musaGraphExecDestroy(graphExec));
    checkMusaErrors(musaGraphDestroy(graph));

    checkMusaErrors(musaStreamDestroy(captureStream));
    checkMusaErrors(musaStreamDestroy(runStream));

    checkMusaErrors(musaFree(dev_a));
    checkMusaErrors(musaFree(dev_b));
    free(host_a);
    free(host_b);
    return musaSuccess;
}

musaError_t GraphLaunchFixture::graphBeforeKernel(unsigned int size, int kernelCount, long long* tElapsedTime) {
    int* host_a = (int*)malloc(size * sizeof(int));
    int* host_b = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        host_a[i] = Random() % 255;
        host_b[i] = 0;
    }

    int* dev_a = 0;
    int* dev_b = 0;
    checkMusaErrors(musaMalloc((void**)&dev_a, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_b, size * sizeof(int)));

    musaStream_t captureStream, runStream;
    checkMusaErrors(musaStreamCreate(&captureStream));
    checkMusaErrors(musaStreamCreate(&runStream));

    checkMusaErrors(musaStreamBeginCapture(captureStream, musaStreamCaptureModeGlobal));
    uint32_t threads = BLOCK_SIZE;
    uint32_t blocks  = getGridSize(size);

    copyKernel<<<blocks, threads, 0, captureStream>>>(dev_b, dev_a, size);
    for (int i = 0; i < (kernelCount / 2); i++) {
        addOneKernel<<<blocks, threads, 0, captureStream>>>(dev_b, size);
    }

    musaGraph_t graph = NULL;
    checkMusaErrors(musaStreamEndCapture(captureStream, &graph));

    musaGraphExec_t graphExec;
    checkMusaErrors(musaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    checkMusaErrors(musaMemcpyAsync(dev_a, host_a, size * sizeof(int), musaMemcpyHostToDevice, runStream));

    int loopCount = 1000;
    CPerfCounter timer;
    timer.Start();
    for (int i = 0; i < loopCount; ++i) {
        // graph
        checkMusaErrors(musaGraphLaunch(graphExec, runStream));

        for (int i = 0; i < (kernelCount / 2); i++) {
            addOneKernel<<<blocks, threads, 0, runStream>>>(dev_b, size);
        }
    }
    musaStreamSynchronize(runStream);
    timer.Stop();

    *tElapsedTime = timer.GetElapsedSeconds() * 1000 / loopCount;

    checkMusaErrors(musaMemcpyAsync(host_b, dev_b, size * sizeof(int), musaMemcpyDeviceToHost, runStream));
    musaStreamSynchronize(runStream);

    for (int i = 0; i < size; i++) {
        if (host_b[i] != (host_a[i] + kernelCount)) {
            printf("graphBeforeKernel check failed, i %d, hb=%d, ha=%d\n", i, host_b[i], host_a[i]);
            exit(EXIT_FAILURE);
        }
    }

    checkMusaErrors(musaGraphExecDestroy(graphExec));
    checkMusaErrors(musaGraphDestroy(graph));

    checkMusaErrors(musaStreamDestroy(captureStream));
    checkMusaErrors(musaStreamDestroy(runStream));

    checkMusaErrors(musaFree(dev_a));
    checkMusaErrors(musaFree(dev_b));
    free(host_a);
    free(host_b);
    return musaSuccess;
}

BASELINE_F(kernelLaunch, kernelLaunch, GraphLaunchFixture, 5, 1) {
    long long tElapsedTime = 0;
    int count              = 1024;
    kernelLaunch(m_Size, count, &tElapsedTime);

    kernelCount->addValue(count);
    exeTime->addValue(tElapsedTime);
    exeBand->addValue(((float)(m_Size * sizeof(int)) / tElapsedTime));
}

BASELINE_F(graphLaunch, graphLaunch, GraphLaunchFixture, 5, 1) {
    long long tElapsedTime = 0;
    int count              = 1024;
    graphLaunch(m_Size, count, &tElapsedTime);

    kernelCount->addValue(count);
    exeTime->addValue(tElapsedTime);
    exeBand->addValue(((float)(m_Size * sizeof(int)) / tElapsedTime));
}

BENCHMARK_F(graphLaunch, graphApiLaunch, GraphLaunchFixture, 5, 1) {
    long long tElapsedTime = 0;
    int count              = 1024;
    graphApiLaunch(m_Size, count, &tElapsedTime);

    kernelCount->addValue(count);
    exeTime->addValue(tElapsedTime);
    exeBand->addValue(((float)(m_Size * sizeof(int)) / tElapsedTime));
}

BENCHMARK_F(graphLaunch, emptyKernelGraphLaunch, GraphLaunchFixture, 5, 1) {
    long long tElapsedTime = 0;
    int count              = 1024;
    emptyKernelGraphLaunch(m_Size, count, &tElapsedTime);

    kernelCount->addValue(count);
    exeTime->addValue(tElapsedTime);
    exeBand->addValue(((float)(m_Size * sizeof(int)) / tElapsedTime));
}

BENCHMARK_F(graphLaunch, graphAndKernel, GraphLaunchFixture, 5, 1) {
    long long tElapsedTime = 0;
    int count              = 1024;
    graphAndKernel(m_Size, count, &tElapsedTime);

    kernelCount->addValue(count);
    exeTime->addValue(tElapsedTime);
    exeBand->addValue(((float)(m_Size * sizeof(int)) / tElapsedTime));
}

BENCHMARK_F(graphLaunch, graphAfterKernel, GraphLaunchFixture, 5, 1) {
    long long tElapsedTime = 0;
    int count              = 1024;
    graphAfterKernel(m_Size, count, &tElapsedTime);

    kernelCount->addValue(count);
    exeTime->addValue(tElapsedTime);
    exeBand->addValue(((float)(m_Size * sizeof(int)) / tElapsedTime));
}

BENCHMARK_F(graphLaunch, graphBeforeKernel, GraphLaunchFixture, 5, 1) {
    long long tElapsedTime = 0;
    int count              = 1024;
    graphBeforeKernel(m_Size, count, &tElapsedTime);

    kernelCount->addValue(count);
    exeTime->addValue(tElapsedTime);
    exeBand->addValue(((float)(m_Size * sizeof(int)) / tElapsedTime));
}
