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

typedef std::chrono::high_resolution_clock Clock;
__global__ void addKernel(int* c, const int* a, const int* b, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (gtid == 1)
    //     printf("running addKernel, a[1] = %d, b[1] = %d\n", a[1], b[1]);
    if (gtid < n)
        c[gtid] = a[gtid] + b[gtid];
}

__global__ void addOneKernel(int* c, const int* a, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (gtid == 1)
    //     printf("running addOneKernel, a[1] = %d\n", a[1]);
    if (gtid < n)
        c[gtid] = a[gtid] + 1;
}

__global__ void mulTwoKernel(int* c, const int* a, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (gtid == 1)
    //     printf("running mulTwoKernel, a[1] = %d\n", a[1]);
    if (gtid < n)
        c[gtid] = a[gtid] * 2;
}

struct CallbackPara {
    int* res;
    int* inp;
    int size;
};

void testCallback(CallbackPara* para) {
    // printf("testCallback....\n");
    int* res = para->res;
    int* inp = para->inp;
    int size = para->size;
    for (int i = 0; i < size; i++) {
        if (res[i] != 2 * inp[i] + 2) {
            printf("res[%d] = %d, inp[%d] = %d\n", i, res[i], i, inp[i]);
        }
    }
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

class GraphFixture : public TestFixture {
public:
    GraphFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int i = 4; i < 18; i += 2) {
            problemSpace.push_back(1 << i);
        }
        return problemSpace;
    }
    void setUp(const TestFixture::ExperimentValue& experimentValue) override { m_Size = experimentValue.Value; }
    void tearDown() override {}
    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}
    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->strGraphTime, this->strGraphBand, this->exeTimeWithoutGraph, this->exeBandWithoutGraph,
            //                this->errWithoutGraph,
            this->exeTimeWithGraph, this->exeBandWithGraph,
            //                this->errWithGraph,
            this->uratio};
    }
    musaError_t naiveGraphWithoutGraph(
        int* c, const int* a, const int* b, unsigned int size, int loopCount, long long* tElapsedExec);
    musaError_t naiveGraphWithGraph(int* c, const int* a, const int* b, unsigned int size, int loopCount,
        long long* tElapsedExec, long long* tElapsedGraph);
    musaError_t typicalGraphWithoutGraph(unsigned int size, int loopCount, long long* tElapsedExec);
    musaError_t
    typicalGraphWithGraph(unsigned int size, int loopCount, long long* tElapsedExec, long long* tElapsedGraph);
    musaError_t complexGraphWithoutGraph(unsigned int size, int loopCount, long long* tElapsedExec);
    musaError_t
    complexGraphWithGraph(unsigned int size, int loopCount, long long* tElapsedExec, long long* tElapsedGraph);
    musaError_t pureDeviceGraphWithoutGraph(unsigned int size, int loopCount, long long* tElapsedExec);
    musaError_t
    pureDeviceGraphWithGraph(unsigned int size, int loopCount, long long* tElapsedExec, long long* tElapsedGraph);
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
    std::shared_ptr<UDMGPUTime> exeTimeWithoutGraph{new UDMGPUTime("no(ms)")};
    std::shared_ptr<UDMBandWidth> exeBandWithoutGraph{new UDMBandWidth("no(MB/s)")};
    std::shared_ptr<UDMGPUTime> exeTimeWithGraph{new UDMGPUTime("gr(ms)")};
    std::shared_ptr<UDMBandWidth> exeBandWithGraph{new UDMBandWidth("gr(MB/s)")};
    std::shared_ptr<UDMGPUTime> strGraphTime{new UDMGPUTime("str(us)")};
    std::shared_ptr<UDMBandWidth> strGraphBand{new UDMBandWidth("str(M/s)")};
    std::shared_ptr<UDMUseage> errWithoutGraph{new UDMUseage("er1(%)")};
    std::shared_ptr<UDMUseage> errWithGraph{new UDMUseage("er2(%)")};
    std::shared_ptr<UDMRatio> uratio{new UDMRatio("*ratio")};
};

musaError_t GraphFixture::naiveGraphWithoutGraph(
    int* c, const int* a, const int* b, unsigned int size, int loopCount, long long* tElapsedExec) {
    int* dev_a          = 0;
    int* dev_b          = 0;
    int* dev_c          = 0;
    int threads         = 256;
    int blocks          = (size + threads - 1) / threads;
    long long usElapsed = 0;

    checkMusaErrors(musaMalloc((void**)&dev_a, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_b, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_c, size * sizeof(int)));
    CPerfCounter timer;
    timer.Start();
    for (int i = 0; i < loopCount; ++i) {
        checkMusaErrors(musaMemcpy(dev_a, a, size * sizeof(int), musaMemcpyHostToDevice));
        checkMusaErrors(musaMemcpy(dev_b, b, size * sizeof(int), musaMemcpyHostToDevice));
        addKernel<<<blocks, threads>>>(dev_c, dev_a, dev_b, size);
        checkMusaErrors(musaGetLastError());
        checkMusaErrors(musaDeviceSynchronize());
        checkMusaErrors(musaMemcpy(c, dev_c, size * sizeof(int), musaMemcpyDeviceToHost));
    }
    timer.Stop();
    usElapsed     = timer.GetElapsedSeconds() * 1000 * 1000;
    *tElapsedExec = usElapsed;
    checkMusaErrors(musaFree(dev_a));
    checkMusaErrors(musaFree(dev_b));
    checkMusaErrors(musaFree(dev_c));
    return musaSuccess;
}

musaError_t GraphFixture::naiveGraphWithGraph(int* c, const int* a, const int* b, unsigned int size, int loopCount,
    long long* tElapsedExec, long long* tElapsedGraph) {
    int* dev_a  = 0;
    int* dev_b  = 0;
    int* dev_c  = 0;
    int threads = 256;
    int blocks  = (size + threads - 1) / threads;

    musaStream_t streamForGraph;
    musaGraph_t graph;
    std::vector<musaGraphNode_t> nodeDependencies;
    musaGraphNode_t kernelNode, memcpyNode1, memcpyNode2, memcpyNode3;
    musaKernelNodeParams kernelNodeParams = {0};

    checkMusaErrors(musaMalloc((void**)&dev_a, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_b, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_c, size * sizeof(int)));
    checkMusaErrors(musaGraphCreate(&graph, 0));
    checkMusaErrors(musaStreamCreateWithFlags(&streamForGraph, musaStreamNonBlocking));

    CPerfCounter timer;
    timer.Start();
    checkMusaErrors(
        musaGraphAddMemcpyNode1D(&memcpyNode1, graph, NULL, 0, dev_a, a, size * sizeof(int), musaMemcpyHostToDevice));
    nodeDependencies.push_back(memcpyNode1);
    checkMusaErrors(
        musaGraphAddMemcpyNode1D(&memcpyNode2, graph, NULL, 0, dev_b, b, size * sizeof(int), musaMemcpyHostToDevice));
    nodeDependencies.push_back(memcpyNode2);
    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));

    kernelNodeParams.func           = (void*)addKernel;
    kernelNodeParams.gridDim        = dim3(blocks, 1, 1);
    kernelNodeParams.blockDim       = dim3(threads, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    void* kernelArgs[4]             = {&dev_c, &dev_a, &dev_b, &size};
    kernelNodeParams.kernelParams   = (void**)kernelArgs;
    kernelNodeParams.extra          = NULL;

    checkMusaErrors(
        musaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams));
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);
    checkMusaErrors(musaGraphAddMemcpyNode1D(&memcpyNode3, graph, nodeDependencies.data(), nodeDependencies.size(), c,
        dev_c, size * sizeof(int), musaMemcpyDeviceToHost));
    musaGraphExec_t graphExec;
    checkMusaErrors(musaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    timer.Stop();

    long long usElapsedGraph = timer.GetElapsedSeconds() * 1000 * 1000;
    *tElapsedGraph           = usElapsedGraph;

    timer.Restart();
    for (int i = 0; i < loopCount; ++i) {
        checkMusaErrors(musaGraphLaunch(graphExec, streamForGraph));
        checkMusaErrors(musaStreamSynchronize(streamForGraph));
    }
    timer.Stop();
    long long usElapsed = timer.GetElapsedSeconds() * 1000 * 1000;
    *tElapsedExec       = usElapsed;
    checkMusaErrors(musaGraphExecDestroy(graphExec));
    checkMusaErrors(musaGraphDestroy(graph));
    checkMusaErrors(musaStreamDestroy(streamForGraph));
    checkMusaErrors(musaFree(dev_a));
    checkMusaErrors(musaFree(dev_b));
    checkMusaErrors(musaFree(dev_c));
    return musaSuccess;
}

musaError_t GraphFixture::typicalGraphWithGraph(
    unsigned int size, int loopCount, long long* tElapsedExec, long long* tElapsedGraph) {
    // 01. Step1. Initialize the hsot/device memorys
    int* ha            = (int*)malloc(size * sizeof(int));
    int* hy            = (int*)malloc(size * sizeof(int));
    int* hf            = (int*)malloc(size * sizeof(int));
    int* da            = 0;
    int* db            = 0;
    int* dc            = 0;
    int* dd            = 0;
    int* de            = 0;
    int* dx            = 0;
    int theadsPerBlock = 256;
    int blocksPerGrid  = (size + theadsPerBlock - 1) / theadsPerBlock;
    for (int i = 0; i < size; i++) {
        ha[i] = i;
    }
    checkMusaErrors(musaMalloc((void**)&da, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&db, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dc, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dd, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&de, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dx, size * sizeof(int)));

    // 02. Step2. Create the stream and graph
    musaStream_t streamForGraph;
    musaGraph_t graph;
    musaGraphNode_t nodeA, nodeB, nodeC, nodeD, nodeE, nodeF, nodeX, nodeY;
    musaKernelNodeParams kernelNodeParams  = {0};
    musaKernelNodeParams kernelNodeParams0 = {0};
    musaKernelNodeParams kernelNodeParams1 = {0};
    musaKernelNodeParams kernelNodeParams2 = {0};
    musaKernelNodeParams kernelNodeParams3 = {0};
    checkMusaErrors(musaStreamCreateWithFlags(&streamForGraph, musaStreamNonBlocking));
    checkMusaErrors(musaGraphCreate(&graph, 0));
    // 0201. nodeA depend on no node, copy ha to da
    CPerfCounter timer;
    timer.Start();
    checkMusaErrors(musaGraphAddMemcpyNode1D(&nodeA, graph, NULL, 0, da, ha, size * sizeof(int), musaMemcpyHostToDevice));
    // 0202. nodeB depend on nodeA, copy da to db,
    checkMusaErrors(
        musaGraphAddMemcpyNode1D(&nodeB, graph, &nodeA, 1, db, da, size * sizeof(int), musaMemcpyDeviceToDevice));
    // 0203. nodeX depend on nodeA, kernel dx[] = da[] + 1
    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
    kernelNodeParams0.func           = (void*)addOneKernel;
    kernelNodeParams0.gridDim        = dim3(blocksPerGrid, 1, 1);
    kernelNodeParams0.blockDim       = dim3(theadsPerBlock, 1, 1);
    kernelNodeParams0.sharedMemBytes = 0;
    void* kernelArgs[3]              = {&dx, &da, &size};
    kernelNodeParams0.kernelParams   = (void**)kernelArgs;
    kernelNodeParams0.extra          = nullptr;
    checkMusaErrors(musaGraphAddKernelNode(&nodeX, graph, &nodeA, 1, &kernelNodeParams0));
    // 0204. nodeC depend on nodeB, kernel dc[] = db[] * 2
    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
    kernelNodeParams1.func           = (void*)mulTwoKernel;
    kernelNodeParams1.gridDim        = dim3(blocksPerGrid, 1, 1);
    kernelNodeParams1.blockDim       = dim3(theadsPerBlock, 1, 1);
    kernelNodeParams1.sharedMemBytes = 0;
    void* kernelArgs1[3]             = {&dc, &db, &size};
    kernelNodeParams1.kernelParams   = (void**)kernelArgs1;
    kernelNodeParams1.extra          = nullptr;
    checkMusaErrors(musaGraphAddKernelNode(&nodeC, graph, &nodeB, 1, &kernelNodeParams1));
    // 0205. nodeD depend on nodeB, kernel dd[] = db[] + 1
    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
    kernelNodeParams2.func           = (void*)addOneKernel;
    kernelNodeParams2.gridDim        = dim3(blocksPerGrid, 1, 1);
    kernelNodeParams2.blockDim       = dim3(theadsPerBlock, 1, 1);
    kernelNodeParams2.sharedMemBytes = 0;
    void* kernelArgs2[3]             = {&dd, &db, &size};
    kernelNodeParams2.kernelParams   = (void**)kernelArgs2;
    kernelNodeParams2.extra          = nullptr;
    checkMusaErrors(musaGraphAddKernelNode(&nodeD, graph, &nodeB, 1, &kernelNodeParams2));
    // 02.06. nodeE depend on nodeC,nodeE, kernel de[] = dc[] + dd[]
    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
    kernelNodeParams3.func              = (void*)addKernel;
    kernelNodeParams3.gridDim           = dim3(blocksPerGrid, 1, 1);
    kernelNodeParams3.blockDim          = dim3(theadsPerBlock, 1, 1);
    kernelNodeParams3.sharedMemBytes    = 0;
    void* kernelArgs3[4]                = {&de, &dc, &dd, &size};
    kernelNodeParams3.kernelParams      = (void**)kernelArgs3;
    kernelNodeParams3.extra             = nullptr;
    musaGraphNode_t nodeDependencies[2] = {nodeC, nodeD};
    checkMusaErrors(musaGraphAddKernelNode(&nodeE, graph, &nodeDependencies[0], 2, &kernelNodeParams3));
    // 02.07. nodeY depend on nodeX, memcpy dx->hy
    checkMusaErrors(
        musaGraphAddMemcpyNode1D(&nodeY, graph, &nodeX, 1, hy, dx, size * sizeof(int), musaMemcpyDeviceToHost));
    // 02.08. nodeF depend on nodeE, memcpy de->hf
    checkMusaErrors(
        musaGraphAddMemcpyNode1D(&nodeF, graph, &nodeE, 1, hf, de, size * sizeof(int), musaMemcpyDeviceToHost));
    musaGraphExec_t graphExec;
    checkMusaErrors(musaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    timer.Stop();
    long long usElapsedGraph = timer.GetElapsedSeconds() * 1000 * 1000;
    *tElapsedGraph           = usElapsedGraph;

    // 03. Step3. Run the graph
    timer.Restart();
    for (int i = 0; i < loopCount; ++i) {
        checkMusaErrors(musaGraphLaunch(graphExec, streamForGraph));
        checkMusaErrors(musaStreamSynchronize(streamForGraph));
    }
    timer.Stop();
    long long usElapsed = timer.GetElapsedSeconds() * 1000 * 1000;
    // 04. Step4. Check the result
    for (int i = 0; i < size; i++) {
        if (hy[i] != ha[i] + 1) {
            printf("hy[%d] = %d, ha[%d] = %d\n", i, hy[i], i, ha[i]);
            exit(EXIT_FAILURE);
        }
        if (hf[i] != ha[i] * 3 + 1) {
            printf("hf[%d] = %d, ha[%d] = %d\n", i, hf[i], i, ha[i]);
            exit(EXIT_FAILURE);
        }
    }
    *tElapsedExec = usElapsed;
    // 05. Step5. Clean the memory and resources
    checkMusaErrors(musaGraphExecDestroy(graphExec));
    checkMusaErrors(musaGraphDestroy(graph));
    checkMusaErrors(musaStreamDestroy(streamForGraph));
    checkMusaErrors(musaFree(da));
    checkMusaErrors(musaFree(db));
    checkMusaErrors(musaFree(dc));
    checkMusaErrors(musaFree(dd));
    checkMusaErrors(musaFree(de));
    checkMusaErrors(musaFree(dx));
    free(ha);
    free(hy);
    free(hf);
    return musaSuccess;
}

musaError_t GraphFixture::typicalGraphWithoutGraph(unsigned int size, int loopCount, long long* tElapsedExec) {
    // 01. Step1. Initialize the hsot/device memorys
    int* ha            = (int*)malloc(size * sizeof(int));
    int* hy            = (int*)malloc(size * sizeof(int));
    int* hf            = (int*)malloc(size * sizeof(int));
    int* da            = 0;
    int* db            = 0;
    int* dc            = 0;
    int* dd            = 0;
    int* de            = 0;
    int* dx            = 0;
    int* dy            = 0;
    int theadsPerBlock = 256;
    int blocksPerGrid  = (size + theadsPerBlock - 1) / theadsPerBlock;
    for (int i = 0; i < size; i++) {
        ha[i] = i;
    }
    checkMusaErrors(musaMalloc((void**)&da, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&db, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dc, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dd, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&de, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dx, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dy, size * sizeof(int)));

    musaStream_t streamEdge1, streamEdge2, streamEdge3;
    musaEvent_t eventA, eventB, eventC, eventD, eventE, eventF, eventX, eventY;
    checkMusaErrors(musaEventCreate(&eventA));
    checkMusaErrors(musaEventCreate(&eventB));
    checkMusaErrors(musaEventCreate(&eventC));
    checkMusaErrors(musaEventCreate(&eventD));
    checkMusaErrors(musaEventCreate(&eventE));
    checkMusaErrors(musaEventCreate(&eventF));
    checkMusaErrors(musaEventCreate(&eventX));
    checkMusaErrors(musaEventCreate(&eventY));
    checkMusaErrors(musaStreamCreateWithFlags(&streamEdge1, musaStreamNonBlocking));
    checkMusaErrors(musaStreamCreateWithFlags(&streamEdge2, musaStreamNonBlocking));
    checkMusaErrors(musaStreamCreateWithFlags(&streamEdge3, musaStreamNonBlocking));
    *tElapsedExec = 0.f;
    for (int i = 0; i < loopCount; ++i) {
        CPerfCounter timer;
        timer.Start();
        checkMusaErrors(musaMemcpyAsync(da, ha, size * sizeof(int), musaMemcpyHostToDevice, streamEdge1)); // nodeA
        checkMusaErrors(musaEventRecord(eventA, streamEdge1));

        checkMusaErrors(musaMemcpyAsync(db, da, size * sizeof(int), musaMemcpyDeviceToDevice, streamEdge1)); // nodeB
        checkMusaErrors(musaEventRecord(eventB, streamEdge1));

        checkMusaErrors(musaStreamWaitEvent(streamEdge2, eventA, 0));
        addOneKernel<<<blocksPerGrid, theadsPerBlock, 0, streamEdge2>>>(dx, da, size); // nodeX
        checkMusaErrors(musaEventRecord(eventX, streamEdge2));

        mulTwoKernel<<<blocksPerGrid, theadsPerBlock, 0, streamEdge1>>>(dc, db, size); // nodeC

        checkMusaErrors(musaStreamWaitEvent(streamEdge3, eventB, 0));
        addOneKernel<<<blocksPerGrid, theadsPerBlock, 0, streamEdge3>>>(dd, db, size); // nodeD
        checkMusaErrors(musaEventRecord(eventD, streamEdge3));

        checkMusaErrors(musaMemcpyAsync(hy, dx, size * sizeof(int), musaMemcpyDeviceToHost, streamEdge2)); // nodeY

        checkMusaErrors(musaStreamWaitEvent(streamEdge1, eventD, 0));
        addKernel<<<blocksPerGrid, theadsPerBlock, 0, streamEdge1>>>(de, dc, dd, size); // nodeE

        checkMusaErrors(musaMemcpyAsync(hf, de, size * sizeof(int), musaMemcpyDeviceToHost, streamEdge1)); // nodeF

        checkMusaErrors(musaStreamSynchronize(streamEdge1));
        checkMusaErrors(musaStreamSynchronize(streamEdge2));
        checkMusaErrors(musaStreamSynchronize(streamEdge3));
        timer.Stop();
        long long usElapsed = timer.GetElapsedSeconds() * 1000 * 1000;
        *tElapsedExec       = (*tElapsedExec) + usElapsed;
        for (int i = 0; i < size; ++i) {
            if (hy[i] != ha[i] + 1) {
                printf("hy[%d] = %d, ha[%d] = %d\n", i, hy[i], i, ha[i]);
                exit(EXIT_FAILURE);
            }
            if (hf[i] != ha[i] * 3 + 1) {
                printf("hf[%d] = %d, ha[%d] = %d\n", i, hf[i], i, ha[i]);
                exit(EXIT_FAILURE);
            }
        }
    }
    checkMusaErrors(musaEventDestroy(eventA));
    checkMusaErrors(musaEventDestroy(eventB));
    checkMusaErrors(musaEventDestroy(eventC));
    checkMusaErrors(musaEventDestroy(eventD));
    checkMusaErrors(musaEventDestroy(eventE));
    checkMusaErrors(musaEventDestroy(eventF));
    checkMusaErrors(musaEventDestroy(eventX));
    checkMusaErrors(musaEventDestroy(eventY));
    checkMusaErrors(musaStreamDestroy(streamEdge1));
    checkMusaErrors(musaStreamDestroy(streamEdge2));
    checkMusaErrors(musaStreamDestroy(streamEdge3));
    checkMusaErrors(musaFree(da));
    checkMusaErrors(musaFree(db));
    checkMusaErrors(musaFree(dc));
    checkMusaErrors(musaFree(dd));
    checkMusaErrors(musaFree(de));
    checkMusaErrors(musaFree(dx));
    checkMusaErrors(musaFree(dy));
    free(ha);
    free(hy);
    free(hf);
    return musaSuccess;
}

musaError_t GraphFixture::complexGraphWithGraph(
    unsigned int size, int loopCount, long long* tElapsedExec, long long* tElapsedGraph) {
    // 01. Step1. Initialize the hsot/device memorys
    int** pph = (int**)malloc(10 * sizeof(int*));
    for (int i = 0; i < 10; i++) {
        pph[i] = (int*)malloc(size * sizeof(int));
        if (i == 0) {
            for (int j = 0; j < size; j++) {
                pph[i][j] = j;
            }
        }
    }
    int** res = (int**)malloc(10 * sizeof(int*));
    for (int i = 0; i < 10; i++) {
        res[i] = (int*)malloc(size * sizeof(int));
    }
    int** ppd = (int**)malloc(30 * sizeof(int*));
    for (int i = 0; i < 30; i++) {
        checkMusaErrors(musaMalloc((void**)&ppd[i], size * sizeof(int)));
    }
    int theadsPerBlock = 256;
    int blocksPerGrid  = (size + theadsPerBlock - 1) / theadsPerBlock;
    // 02. Step2. Create the stream and graph
    musaStream_t streamForGraph, streamForGraph1;
    musaGraph_t graph;
    musaGraphNode_t nodes[30];                       /// for nodes[i], the device data will be ppd[i]
    musaKernelNodeParams kernelNodeParams[30] = {0}; /// for nodes[i], if it is a kernel node, the kernelNodeParams[i]
                                                     /// will be used
    checkMusaErrors(musaStreamCreateWithFlags(&streamForGraph, musaStreamNonBlocking));
    checkMusaErrors(musaStreamCreateWithFlags(&streamForGraph1, musaStreamNonBlocking));
    checkMusaErrors(musaGraphCreate(&graph, 0));

    CPerfCounter timer;
    timer.Start();
    /* 0201.edge1, node0 -> node1 -> node2 -> node4
                               \-> node3 /
       node0: memcpy pph[0] -> ppd[0], no dependency
       node1: memcpy ppd[0] -> ppd[1], depend on node0
       node2: kernel ppd[1] =  ppd[0] + 1, depend on node1
       node3: kernel ppd[2] =  ppd[0] * 2, depend on node1
       node4: kernel ppd[3] =  ppd[1] + ppd[2], depend on node2 and node3
       node5: memcpy ppd[3] -> res[0], depend on node4, res[i] = 3 * pph[0] + 1
    **/
    checkMusaErrors(musaGraphAddMemcpyNode1D(
        &nodes[0], graph, NULL, 0, ppd[0], pph[0], size * sizeof(int), musaMemcpyHostToDevice)); // node0.

    checkMusaErrors(musaGraphAddMemcpyNode1D(
        &nodes[1], graph, &nodes[0], 1, ppd[1], ppd[0], size * sizeof(int), musaMemcpyDeviceToDevice)); // node1.

    kernelNodeParams[2].func           = (void*)addOneKernel;
    kernelNodeParams[2].gridDim        = dim3(blocksPerGrid, 1, 1);
    kernelNodeParams[2].blockDim       = dim3(theadsPerBlock, 1, 1);
    kernelNodeParams[2].sharedMemBytes = 0;
    void* kernelArgs1[3]               = {&ppd[2], &ppd[1], &size};
    kernelNodeParams[2].kernelParams   = (void**)kernelArgs1;
    kernelNodeParams[2].extra          = nullptr;
    checkMusaErrors(musaGraphAddKernelNode(&nodes[2], graph, &nodes[1], 1, &kernelNodeParams[2]));
    kernelNodeParams[3].func           = (void*)mulTwoKernel;
    kernelNodeParams[3].gridDim        = dim3(blocksPerGrid, 1, 1);
    kernelNodeParams[3].blockDim       = dim3(theadsPerBlock, 1, 1);
    kernelNodeParams[3].sharedMemBytes = 0;
    void* kernelArgs2[3]               = {&ppd[3], &ppd[1], &size};
    kernelNodeParams[3].kernelParams   = (void**)kernelArgs2;
    kernelNodeParams[3].extra          = nullptr;
    checkMusaErrors(musaGraphAddKernelNode(&nodes[3], graph, &nodes[1], 1, &kernelNodeParams[3]));
    kernelNodeParams[4].func             = (void*)addKernel;
    kernelNodeParams[4].gridDim          = dim3(blocksPerGrid, 1, 1);
    kernelNodeParams[4].blockDim         = dim3(theadsPerBlock, 1, 1);
    kernelNodeParams[4].sharedMemBytes   = 0;
    void* kernelArgs3[4]                 = {&ppd[4], &ppd[2], &ppd[3], &size};
    kernelNodeParams[4].kernelParams     = (void**)kernelArgs3;
    kernelNodeParams[4].extra            = nullptr;
    musaGraphNode_t nodeDependencies4[2] = {nodes[2], nodes[3]};
    checkMusaErrors(musaGraphAddKernelNode(&nodes[4], graph, &nodeDependencies4[0], 2, &kernelNodeParams[4]));
    checkMusaErrors(musaGraphAddMemcpyNode1D(
        &nodes[5], graph, &nodes[4], 1, res[0], ppd[4], size * sizeof(int), musaMemcpyDeviceToHost));

    /*02.02 edge2 node0 -> node6 -> node7 -> node8 -> node9
         node6: kernel ppd[6] =  ppd[0] + 1, depend on node0
         node7: kernel ppd[7] =  ppd[6] * 2, depend on node6
         node8: memcpy ppd[7] -> res[1], depend on node7, res[1] = 2 * pph[0] + 2
         node9: callback, check result in host, depend on node8
    */
    kernelNodeParams[6].func           = (void*)addOneKernel;
    kernelNodeParams[6].gridDim        = dim3(blocksPerGrid, 1, 1);
    kernelNodeParams[6].blockDim       = dim3(theadsPerBlock, 1, 1);
    kernelNodeParams[6].sharedMemBytes = 0;
    void* kernelArgs4[3]               = {&ppd[6], &ppd[0], &size};
    kernelNodeParams[6].kernelParams   = (void**)kernelArgs4;
    kernelNodeParams[6].extra          = nullptr;
    checkMusaErrors(musaGraphAddKernelNode(&nodes[6], graph, &nodes[0], 1, &kernelNodeParams[6]));
    kernelNodeParams[7].func           = (void*)mulTwoKernel;
    kernelNodeParams[7].gridDim        = dim3(blocksPerGrid, 1, 1);
    kernelNodeParams[7].blockDim       = dim3(theadsPerBlock, 1, 1);
    kernelNodeParams[7].sharedMemBytes = 0;
    void* kernelArgs5[3]               = {&ppd[7], &ppd[6], &size};
    kernelNodeParams[7].kernelParams   = (void**)kernelArgs5;
    kernelNodeParams[7].extra          = nullptr;
    checkMusaErrors(musaGraphAddKernelNode(&nodes[7], graph, &nodes[6], 1, &kernelNodeParams[7]));
    checkMusaErrors(musaGraphAddMemcpyNode1D(
        &nodes[8], graph, &nodes[7], 1, res[1], ppd[7], size * sizeof(int), musaMemcpyDeviceToHost));
    musaHostNodeParams hostNodeParams = {0};
    hostNodeParams.fn                 = (musaHostFn_t)testCallback;
    CallbackPara pa1                  = {res[1], pph[0], static_cast<int>(size)};
    hostNodeParams.userData           = &pa1;
    checkMusaErrors(musaGraphAddHostNode(&nodes[9], graph, &nodes[8], 1, &hostNodeParams));

    auto addEdge = [&](int idx, int* result) {
        kernelNodeParams[idx].func           = (void*)addOneKernel;
        kernelNodeParams[idx].gridDim        = dim3(blocksPerGrid, 1, 1);
        kernelNodeParams[idx].blockDim       = dim3(theadsPerBlock, 1, 1);
        kernelNodeParams[idx].sharedMemBytes = 0;
        void* kernelArgs_add[3]              = {&ppd[idx], &ppd[0], &size};
        kernelNodeParams[idx].kernelParams   = (void**)kernelArgs_add;
        kernelNodeParams[idx].extra          = nullptr;
        checkMusaErrors(musaGraphAddKernelNode(&nodes[idx], graph, &nodes[0], 1, &kernelNodeParams[idx]));

        kernelNodeParams[idx + 1].func           = (void*)mulTwoKernel;
        kernelNodeParams[idx + 1].gridDim        = dim3(blocksPerGrid, 1, 1);
        kernelNodeParams[idx + 1].blockDim       = dim3(theadsPerBlock, 1, 1);
        kernelNodeParams[idx + 1].sharedMemBytes = 0;
        void* kernelArgs_mul[3]                  = {&ppd[idx + 1], &ppd[idx], &size};
        kernelNodeParams[idx + 1].kernelParams   = (void**)kernelArgs_mul;
        kernelNodeParams[idx + 1].extra          = nullptr;
        checkMusaErrors(musaGraphAddKernelNode(&nodes[idx + 1], graph, &nodes[idx], 1, &kernelNodeParams[idx + 1]));

        checkMusaErrors(musaGraphAddMemcpyNode1D(&nodes[idx + 2], graph, &nodes[idx + 1], 1, result, ppd[idx + 1],
            size * sizeof(int), musaMemcpyDeviceToHost));
    };
    // 02.03
    addEdge(10, res[2]);
    addEdge(13, res[3]);
    addEdge(16, res[4]);
    addEdge(19, res[5]);
    addEdge(22, res[6]);
    addEdge(22, res[7]);
    musaEvent_t beginEvent, endEvent;
    checkMusaErrors(musaEventCreate(&beginEvent));
    checkMusaErrors(musaEventCreate(&endEvent));
    checkMusaErrors(musaGraphAddEventRecordNode(&nodes[25], graph, &nodes[0], 1, beginEvent));
    checkMusaErrors(musaGraphAddMemcpyNode1D(
        &nodes[26], graph, &nodes[25], 1, ppd[26], ppd[0], size * sizeof(int), musaMemcpyDeviceToDevice));
    checkMusaErrors(musaGraphAddMemcpyNode1D(
        &nodes[27], graph, &nodes[26], 1, res[8], ppd[26], size * sizeof(int), musaMemcpyDeviceToHost));
    checkMusaErrors(musaGraphAddEventRecordNode(&nodes[28], graph, &nodes[27], 1, endEvent));
    checkMusaErrors(musaGraphAddEventWaitNode(&nodes[29], graph, &nodes[28], 1, endEvent));
    // 03. Step3. Run the graph
    musaGraphExec_t graphExec;
    checkMusaErrors(musaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    timer.Stop();
    long long usElapsedGraph = timer.GetElapsedSeconds() * 1000 * 1000;
    *tElapsedGraph           = usElapsedGraph;

    timer.Restart();
    for (int i = 0; i < loopCount; ++i) {
        checkMusaErrors(musaGraphLaunch(graphExec, streamForGraph));
        checkMusaErrors(musaStreamSynchronize(streamForGraph));
    }
    timer.Stop();
    long long usElapsed = timer.GetElapsedSeconds() * 1000 * 1000;
    *tElapsedExec       = usElapsed;
    // 04. Step4. Check the result
    for (int i = 0; i < size; ++i) {
        if (res[0][i] != 3 * pph[0][i] + 1) {
            printf("res[0][%d] = %d, pph[0][%d] = %d\n", i, res[0][i], i, pph[0][i]);
            exit(EXIT_FAILURE);
        }

        for (int j = 1; j < 7; j++) {
            if (res[j][i] != 2 * pph[0][i] + 2) {
                printf("res[%d][%d] = %d, pph[0][%d] = %d\n", j, i, res[j][i], i, pph[0][i]);
                exit(EXIT_FAILURE);
            }
        }
        if (res[8][i] != pph[0][i]) {
            printf("res[7][%d] = %d, pph[0][%d] = %d\n", i, res[8][i], i, pph[0][i]);
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < 10; i++) {
        free(pph[i]);
        free(res[i]);
    }
    for (int i = 0; i < 30; i++) {
        checkMusaErrors(musaFree(ppd[i]));
    }
    free(pph);
    free(res);
    free(ppd);
    // 05. Step5. Clean the memory and resources
    //  checkMusaErrors(musaEventDestroy(beginEvent));
    //  checkMusaErrors(musaEventDestroy(endEvent));
    checkMusaErrors(musaGraphExecDestroy(graphExec));
    checkMusaErrors(musaGraphDestroy(graph));
    checkMusaErrors(musaStreamDestroy(streamForGraph));
    checkMusaErrors(musaStreamDestroy(streamForGraph1));
    return musaSuccess;
}

musaError_t GraphFixture::complexGraphWithoutGraph(unsigned int size, int loopCount, long long* tElapsedExecNoGraph) {
    // 01. Step1. Initialize the hsot/device memorys
    int** pph = (int**)malloc(10 * sizeof(int*));
    for (int i = 0; i < 10; i++) {
        pph[i] = (int*)malloc(size * sizeof(int));
        if (i == 0) {
            for (int j = 0; j < size; j++) {
                pph[i][j] = j;
            }
        }
    }
    int** res = (int**)malloc(10 * sizeof(int*));
    for (int i = 0; i < 10; i++) {
        res[i] = (int*)malloc(size * sizeof(int));
    }
    int** ppd = (int**)malloc(30 * sizeof(int*));
    for (int i = 0; i < 30; i++) {
        checkMusaErrors(musaMalloc((void**)&ppd[i], size * sizeof(int)));
    }
    int theadsPerBlock = 256;
    int blocksPerGrid  = (size + theadsPerBlock - 1) / theadsPerBlock;
    musaStream_t streams[10];
    musaEvent_t events[30];
    musaEvent_t startEvent, endEvent;
    for (int i = 0; i < 10; ++i) {
        checkMusaErrors(musaStreamCreateWithFlags(&streams[i], musaStreamNonBlocking));
    }
    for (int i = 0; i < 30; ++i) {
        checkMusaErrors(musaEventCreate(&events[i]));
    }
    checkMusaErrors(musaEventCreate(&startEvent));
    checkMusaErrors(musaEventCreate(&endEvent));

    *tElapsedExecNoGraph = 0.f;
    for (int runCounter = 0; runCounter < loopCount; ++runCounter) {
        CPerfCounter timer;
        timer.Start();
        checkMusaErrors(musaMemcpyAsync(ppd[0], pph[0], size * sizeof(int), musaMemcpyHostToDevice, streams[0])); // node0.
        checkMusaErrors(musaEventRecord(events[0], streams[0]));

        checkMusaErrors(musaMemcpyAsync(ppd[1], ppd[0], size * sizeof(int), musaMemcpyDeviceToDevice, streams[0])); // node1.
        checkMusaErrors(musaEventRecord(events[1], streams[0]));

        checkMusaErrors(musaEventSynchronize(events[0]));
        addOneKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[2]>>>(ppd[6], ppd[0], size);  // node6.
        addOneKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[3]>>>(ppd[10], ppd[0], size); // node10.
        addOneKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[4]>>>(ppd[13], ppd[0], size); // node13.
        addOneKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[5]>>>(ppd[16], ppd[0], size); // node16.
        addOneKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[6]>>>(ppd[19], ppd[0], size); // node19.
        addOneKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[7]>>>(ppd[22], ppd[0], size); // node22.

        mulTwoKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[2]>>>(ppd[7], ppd[6], size);   // node7.
        mulTwoKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[3]>>>(ppd[11], ppd[10], size); // node11.
        mulTwoKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[4]>>>(ppd[14], ppd[13], size); // node14.
        mulTwoKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[5]>>>(ppd[17], ppd[16], size); // node17.
        mulTwoKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[6]>>>(ppd[20], ppd[19], size); // node20.
        mulTwoKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[7]>>>(ppd[23], ppd[22], size); // node23.

        checkMusaErrors(musaMemcpyAsync(res[1], ppd[7], size * sizeof(int), musaMemcpyDeviceToHost, streams[2])); // node8.
        checkMusaErrors(musaMemcpyAsync(res[2], ppd[11], size * sizeof(int), musaMemcpyDeviceToHost, streams[3])); // node11.
        checkMusaErrors(musaMemcpyAsync(res[3], ppd[14], size * sizeof(int), musaMemcpyDeviceToHost, streams[4])); // node14.
        checkMusaErrors(musaMemcpyAsync(res[4], ppd[17], size * sizeof(int), musaMemcpyDeviceToHost, streams[5])); // node17.
        checkMusaErrors(musaMemcpyAsync(res[5], ppd[20], size * sizeof(int), musaMemcpyDeviceToHost, streams[6])); // node20.
        checkMusaErrors(musaMemcpyAsync(res[6], ppd[23], size * sizeof(int), musaMemcpyDeviceToHost, streams[7])); // node23.

        addOneKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[0]>>>(ppd[2], ppd[1], size); // node2.
        checkMusaErrors(musaEventRecord(events[2], streams[0]));

        checkMusaErrors(musaEventSynchronize(events[1]));
        mulTwoKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[1]>>>(ppd[3], ppd[1], size); // node3.
        checkMusaErrors(musaEventRecord(events[3], streams[1]));

        checkMusaErrors(musaEventSynchronize(events[3]));
        addKernel<<<blocksPerGrid, theadsPerBlock, 0, streams[0]>>>(ppd[4], ppd[2], ppd[3], size); // node4.
        checkMusaErrors(musaEventRecord(events[4], streams[0]));
        checkMusaErrors(musaMemcpyAsync(res[0], ppd[4], size * sizeof(int), musaMemcpyDeviceToHost, streams[0])); // node5.

        checkMusaErrors(musaEventRecord(startEvent, streams[8])); // node25.
        checkMusaErrors(
            musaMemcpyAsync(ppd[26], ppd[0], size * sizeof(int), musaMemcpyDeviceToDevice, streams[0])); // node26.
        checkMusaErrors(musaMemcpyAsync(res[8], ppd[26], size * sizeof(int), musaMemcpyDeviceToHost, streams[0])); // node27.
        checkMusaErrors(musaEventRecord(endEvent, streams[8])); // node28.
        checkMusaErrors(musaEventSynchronize(endEvent));        // node29.
        float ms;
        checkMusaErrors(musaEventElapsedTime(&ms, startEvent, endEvent));

        for (int i = 0; i < 8; ++i) {
            checkMusaErrors(musaStreamSynchronize(streams[i]));
        }
        CallbackPara pa1 = {res[1], pph[0], static_cast<int>(size)};
        testCallback(&pa1);
        timer.Stop();
        long long usElapsed  = timer.GetElapsedSeconds() * 1000 * 1000;
        *tElapsedExecNoGraph = (*tElapsedExecNoGraph) + usElapsed;
        for (int i = 0; i < size; ++i) {
            if (res[0][i] != 3 * pph[0][i] + 1) {
                printf("res[0][%d] = %d, pph[0][%d] = %d\n", i, res[0][i], i, pph[0][i]);
                exit(EXIT_FAILURE);
            }
            for (int j = 1; j < 7; j++) {
                if (res[j][i] != 2 * pph[0][i] + 2) {
                    printf("res[%d][%d] = %d, pph[0][%d] = %d\n", j, i, res[j][i], i, pph[0][i]);
                    exit(EXIT_FAILURE);
                }
            }
            if (res[8][i] != pph[0][i]) {
                printf("res[7][%d] = %d, pph[0][%d] = %d\n", i, res[8][i], i, pph[0][i]);
                exit(EXIT_FAILURE);
            }
        }
    }
    for (int i = 0; i < 10; ++i) {
        free(pph[i]);
        free(res[i]);
    }
    for (int i = 0; i < 30; ++i) {
        checkMusaErrors(musaFree(ppd[i]));
    }
    free(pph);
    free(res);
    free(ppd);
    checkMusaErrors(musaEventDestroy(startEvent));
    checkMusaErrors(musaEventDestroy(endEvent));
    for (int i = 0; i < 10; ++i) {
        checkMusaErrors(musaStreamDestroy(streams[i]));
    }
    for (int i = 0; i < 30; ++i) {
        checkMusaErrors(musaEventDestroy(events[i]));
    }
    return musaSuccess;
}

musaError_t GraphFixture::pureDeviceGraphWithGraph(
    unsigned int size, int loopCount, long long* tElapsedExec, long long* tElapsedGraph) {
    int* host_a = (int*)malloc(size * sizeof(int));
    int* host_b = (int*)malloc(size * sizeof(int));
    int* host_c = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        host_a[i] = Random() % 255;
        host_b[i] = Random() % 255;
        host_c[i] = 0;
    }

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    checkMusaErrors(musaMalloc((void**)&dev_a, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_b, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_c, size * sizeof(int)));

    constexpr int stageNum = 8;
    musaStream_t streamForGraph;
    musaGraph_t graph;
    musaGraphNode_t nodes[3 * stageNum + 3];
    checkMusaErrors(musaStreamCreateWithFlags(&streamForGraph, musaStreamNonBlocking));
    checkMusaErrors(musaGraphCreate(&graph, 0));

    CPerfCounter timer;
    timer.Start();
    checkMusaErrors(
        musaGraphAddMemcpyNode1D(&nodes[0], graph, NULL, 0, dev_a, host_a, size * sizeof(int), musaMemcpyHostToDevice));
    checkMusaErrors(
        musaGraphAddMemcpyNode1D(&nodes[1], graph, NULL, 0, dev_b, host_b, size * sizeof(int), musaMemcpyHostToDevice));
    checkMusaErrors(
        musaGraphAddMemcpyNode1D(&nodes[2], graph, NULL, 0, dev_c, host_c, size * sizeof(int), musaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (size + threads - 1) / threads;
    for (int j = 0; j < stageNum; ++j) {
        std::vector<musaGraphNode_t> nodeDependencies;
        int current = 3 + j * 3;
        if (j == 0) {
            nodeDependencies.push_back(nodes[0]);
            nodeDependencies.push_back(nodes[1]);
            nodeDependencies.push_back(nodes[2]);
        } else {
            nodeDependencies.push_back(nodes[current - 2]);
            nodeDependencies.push_back(nodes[current - 1]);
        }

        musaKernelNodeParams kernelNodeParams;
        kernelNodeParams.func           = (void*)addKernel;
        kernelNodeParams.gridDim        = dim3(blocks, 1, 1);
        kernelNodeParams.blockDim       = dim3(threads, 1, 1);
        kernelNodeParams.sharedMemBytes = 0;
        void* kernelArgs[4]             = {&dev_c, &dev_a, &dev_b, &size};
        kernelNodeParams.kernelParams   = (void**)kernelArgs;
        kernelNodeParams.extra          = nullptr;
        checkMusaErrors(musaGraphAddKernelNode(
            &nodes[current], graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams));

        checkMusaErrors(musaGraphAddMemcpyNode1D(&nodes[current + 1], graph, &nodes[current], 1, dev_a, dev_c,
            size * sizeof(int), musaMemcpyDeviceToDevice));

        checkMusaErrors(musaGraphAddMemcpyNode1D(&nodes[current + 2], graph, &nodes[current], 1, dev_b, dev_c,
            size * sizeof(int), musaMemcpyDeviceToDevice));
    }
    musaGraphExec_t graphExec;
    checkMusaErrors(musaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    timer.Stop();
    long long usElapsedGraph = timer.GetElapsedSeconds() * 1000 * 1000;
    *tElapsedGraph           = usElapsedGraph;

    timer.Restart();
    for (int i = 0; i < loopCount; ++i) {
        checkMusaErrors(musaGraphLaunch(graphExec, streamForGraph));
        checkMusaErrors(musaStreamSynchronize(streamForGraph));
    }
    timer.Stop();
    long long usElapsed = timer.GetElapsedSeconds() * 1000 * 1000;
    *tElapsedExec       = usElapsed;

    checkMusaErrors(musaMemcpy(host_c, dev_c, size * sizeof(int), musaMemcpyDeviceToHost));

    for (int i = 0; i < size; i++) {
        if (host_c[i] != pow(2, stageNum - 1) * (host_a[i] + host_b[i])) {
            printf("pureDeviceWithGraph check failed, i %d, ha=%d, hb=%d, hc=%d\n", i, host_a[i], host_b[i], host_c[i]);
            exit(EXIT_FAILURE);
        }
    }

    free(host_a);
    free(host_b);
    free(host_c);
    checkMusaErrors(musaFree(dev_a));
    checkMusaErrors(musaFree(dev_b));
    checkMusaErrors(musaFree(dev_c));
    return musaSuccess;
}

musaError_t GraphFixture::pureDeviceGraphWithoutGraph(unsigned int size, int loopCount, long long* tElapsedExec) {
    int* host_a = (int*)malloc(size * sizeof(int));
    int* host_b = (int*)malloc(size * sizeof(int));
    int* host_c = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        host_a[i] = Random() % 255;
        host_b[i] = Random() % 255;
        host_c[i] = 0;
    }

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    checkMusaErrors(musaMalloc((void**)&dev_a, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_b, size * sizeof(int)));
    checkMusaErrors(musaMalloc((void**)&dev_c, size * sizeof(int)));

    CPerfCounter timer;
    timer.Start();
    constexpr int stageNum = 8;
    int threads            = 256;
    int blocks             = (size + threads - 1) / threads;
    for (int i = 0; i < loopCount; ++i) {
        checkMusaErrors(musaMemcpy(dev_a, host_a, size * sizeof(int), musaMemcpyHostToDevice));
        checkMusaErrors(musaMemcpy(dev_b, host_b, size * sizeof(int), musaMemcpyHostToDevice));
        checkMusaErrors(musaMemcpy(dev_c, host_c, size * sizeof(int), musaMemcpyHostToDevice));

        for (int j = 0; j < stageNum; ++j) {
            addKernel<<<blocks, threads>>>(dev_c, dev_a, dev_b, size);

            checkMusaErrors(musaMemcpy(dev_a, dev_c, size * sizeof(int), musaMemcpyDeviceToDevice));
            checkMusaErrors(musaMemcpy(dev_b, dev_c, size * sizeof(int), musaMemcpyDeviceToDevice));
        }
    }

    timer.Stop();
    long long usElapsed = timer.GetElapsedSeconds() * 1000 * 1000;
    *tElapsedExec       = usElapsed;

    checkMusaErrors(musaMemcpy(host_c, dev_c, size * sizeof(int), musaMemcpyDeviceToHost));

    for (int i = 0; i < size; i++) {
        if (host_c[i] != pow(2, stageNum - 1) * (host_a[i] + host_b[i])) {
            printf(
                "pureDeviceWithoutGraph check failed, i %d, ha=%d, hb=%d, hc=%d\n", i, host_a[i], host_b[i], host_c[i]);
            exit(EXIT_FAILURE);
        }
    }

    free(host_a);
    free(host_b);
    free(host_c);
    checkMusaErrors(musaFree(dev_a));
    checkMusaErrors(musaFree(dev_b));
    checkMusaErrors(musaFree(dev_c));
    return musaSuccess;
}

BASELINE_F(testGraph, naiveGraph, GraphFixture, 5, 1) {
    int* host_a = (int*)malloc(m_Size * sizeof(int));
    int* host_b = (int*)malloc(m_Size * sizeof(int));
    int* host_c = (int*)malloc(m_Size * sizeof(int));
    for (int i = 0; i < m_Size; ++i) {
        host_a[i] = Random();
        host_b[i] = Random();
        host_c[i] = 0;
    }
    long long tElapsedExec  = 0;
    long long tElapsedGraph = 0;
    naiveGraphWithoutGraph(host_c, host_a, host_b, m_Size, 1000, &tElapsedExec);
    exeTimeWithoutGraph->addValue(tElapsedExec / 1000.0);
    exeBandWithoutGraph->addValue(m_Size * sizeof(int) / (tElapsedExec / 1000.0));
    errWithoutGraph->addValue(getErrorSize(host_c, host_a, host_b, m_Size) * 100.0 / m_Size);
    for (int i = 0; i < m_Size; ++i) {
        host_c[i] = 0;
    }
    naiveGraphWithGraph(host_c, host_a, host_b, m_Size, 1000, &tElapsedExec, &tElapsedGraph);
    exeTimeWithGraph->addValue(tElapsedExec / 1000.0);
    exeBandWithGraph->addValue(m_Size * sizeof(int) / (tElapsedExec / 1000.0));
    strGraphTime->addValue(tElapsedGraph);
    strGraphBand->addValue(4.f / tElapsedGraph);

    errWithGraph->addValue(getErrorSize(host_c, host_a, host_b, m_Size) * 100.0 / m_Size);
    uratio->addValue(exeTimeWithoutGraph->getMean() / exeTimeWithGraph->getMean());
    free(host_a);
    free(host_b);
    free(host_c);
}

BENCHMARK_F(testGraph, typicalGraph, GraphFixture, 5, 1) {
    long long tElapsedExec  = 0;
    long long tElapsedGraph = 0;
    typicalGraphWithGraph(m_Size, 1000, &tElapsedExec, &tElapsedGraph);
    exeTimeWithGraph->addValue(tElapsedExec / 1000);
    exeBandWithGraph->addValue(m_Size * sizeof(int) / (tElapsedExec / 1000.0));
    typicalGraphWithoutGraph(m_Size, 1000, &tElapsedExec);
    exeTimeWithoutGraph->addValue(tElapsedExec / 1000);
    exeBandWithoutGraph->addValue(m_Size * sizeof(int) / (tElapsedExec / 1000.0));
    strGraphTime->addValue(tElapsedGraph);
    strGraphBand->addValue(8.f / tElapsedGraph);
    uratio->addValue(exeTimeWithoutGraph->getMean() / exeTimeWithGraph->getMean());
}

BENCHMARK_F(testGraph, complexGraph, GraphFixture, 5, 1) {
    long long tElapsedExec  = 0;
    long long tElapsedGraph = 0;
    complexGraphWithGraph(m_Size, 1000, &tElapsedExec, &tElapsedGraph);
    exeTimeWithGraph->addValue(tElapsedExec / 1000);
    exeBandWithGraph->addValue(m_Size * sizeof(int) / (tElapsedExec / 1000.0));
    complexGraphWithoutGraph(m_Size, 1000, &tElapsedExec);
    exeTimeWithoutGraph->addValue(tElapsedExec / 1000);
    exeBandWithoutGraph->addValue(m_Size * sizeof(int) / (tElapsedExec / 1000.0));
    strGraphTime->addValue(tElapsedGraph);
    strGraphBand->addValue(30.f / tElapsedGraph);
    uratio->addValue(exeTimeWithoutGraph->getMean() / exeTimeWithGraph->getMean());
}

BENCHMARK_F(testGraph, pureDeviceGraph, GraphFixture, 5, 1) {
    long long tElapsedExec  = 0;
    long long tElapsedGraph = 0;
    pureDeviceGraphWithGraph(m_Size, 1000, &tElapsedExec, &tElapsedGraph);
    exeTimeWithGraph->addValue(tElapsedExec / 1000);
    exeBandWithGraph->addValue(m_Size * sizeof(int) / (tElapsedExec / 1000.0));
    pureDeviceGraphWithoutGraph(m_Size, 1000, &tElapsedExec);
    exeTimeWithoutGraph->addValue(tElapsedExec / 1000);
    exeBandWithoutGraph->addValue(m_Size * sizeof(int) / (tElapsedExec / 1000.0));
    strGraphTime->addValue(tElapsedGraph);
    strGraphBand->addValue(30.f / tElapsedGraph);
    uratio->addValue(exeTimeWithoutGraph->getMean() / exeTimeWithGraph->getMean());
}
