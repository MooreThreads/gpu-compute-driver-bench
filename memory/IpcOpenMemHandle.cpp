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

#include <chrono>
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/shm.h>
#include <filesystem>
#include "stdio.h"

#include "musa.h"
#include "musa_runtime.h"
#include "helper_musa.h"
#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "ShareMemroyDefine.h"

namespace fs = std::filesystem;

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

static const int SamplesCount    = 1;
static const int IterationsCount = 4;

class IpcMemoryFixture : public TestFixture {
public:
    IpcMemoryFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t elements = 1LL << 5; elements <= (1LL << 30); elements *= 2) {
            problemSpace.push_back(elements);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        this->mallocSize = experimentValue.Value;
    }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}

    void onExperimentEnd() override {}

    int testGetAndImportedMemory(float* getTime, float* importTime);

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->getBand, this->getTime, this->openBand, this->openTime};
    }

    int64_t mallocSize;

    void* deviceMemory;

    std::shared_ptr<UDMBandWidth> getBand{new UDMBandWidth("*B(Get)")};

    std::shared_ptr<UDMGPUTime> getTime{new UDMGPUTime("t(Get)")};

    std::shared_ptr<UDMBandWidth> openBand{new UDMBandWidth("*B(Open)")};

    std::shared_ptr<UDMGPUTime> openTime{new UDMGPUTime("t(Open)")};
};

std::string getCurrentExecutorDir() {
    char buf[PATH_MAX] = {0};
    ssize_t len        = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len == -1) {
        perror("readlink");
        return "";
    }
    buf[len] = '\0';
    std::string fullPath(buf);
    auto lastSlash = fullPath.find_last_of('/');
    return fullPath.substr(0, lastSlash);
}

int IpcMemoryFixture::testGetAndImportedMemory(float* getTime, float* openTime) {
    key_t key;
    int shmid;
    if ((key = ftok(".", 'a')) == -1) {
        std::cerr << "ftok fail" << std::endl;
        return -1;
    }

    int shareMemSize = sizeof(ShareMemHandleInfo);
    if ((shmid = shmget(key, shareMemSize, IPC_CREAT | 0666)) == -1) {
        std::cerr << "shmget fail" << std::endl;
        return -1;
    }

    void* sm = nullptr;
    if ((sm = shmat(shmid, nullptr, 0)) == (char*)-1) {
        std::cerr << "shmat fail" << std::endl;
        return -1;
    }
    ShareMemHandleInfo* shareMem = static_cast<ShareMemHandleInfo*>(sm);
    shareMem->child_wake         = false;
    shareMem->parent_wake        = false;
    shareMem->mallocSize         = mallocSize;
    pid_t pid                    = fork();
    if (pid == -1) {
        std::cerr << "test ipc memory but fork failed!" << std::endl;
        return -1;
    } else if (pid == 0) {
        // open handler process
        auto execDir = getCurrentExecutorDir();
        if (execDir.empty()) {
            std::cerr << "getCurrentExecutorDir fail" << std::endl;
        }

        auto childPorcess = execDir + "/IpcOpenMemHandleChild";
        auto ret          = execv(childPorcess.c_str(), nullptr);
        if (ret < 0) {
            std::cerr << "exec failed" << std::endl;
        }
        exit(0);
    } else {
        // get handler process
        musaEvent_t start;
        musaEvent_t stop;
        checkMusaErrors(musaEventCreate(&start));
        checkMusaErrors(musaEventCreate(&stop));
        checkMusaErrors(musaMalloc(&deviceMemory, mallocSize));
        checkMusaErrors(musaMemset(deviceMemory, 11, mallocSize));
        musaIpcMemHandle_t ipcHandle;
        checkMusaErrors(musaEventRecord(start, 0));
        checkMusaErrors(musaIpcGetMemHandle(&(shareMem->handler), deviceMemory));
        checkMusaErrors(musaEventRecord(stop, 0));
        checkMusaErrors(musaEventSynchronize(stop));
        checkMusaErrors(musaEventElapsedTime(getTime, start, stop));

        shareMem->child_wake.store(true, std::memory_order_seq_cst);
        while (!shareMem->parent_wake.load(std::memory_order_seq_cst)) {}

        checkMusaErrors(musaFree(deviceMemory));
        int status;
        if (wait(&status) < 0) {
            std::cerr << "wait err" << std::endl;
            return -1;
        }
        if (!WIFEXITED(status)) {
            std::cerr << "child exit with err" << std::endl;
            return -1;
        }
        *openTime = shareMem->openTime;
        if (shmdt(sm) == -1) {
            std::cerr << "shmdt failed" << std::endl;
            return -1;
        }

        if (shmctl(shmid, IPC_RMID, nullptr) == -1) {
            std::cerr << "shmctl failed" << std::endl;
            return -1;
        }
    }
    return 0;
}

BASELINE_F(ipc, ipcMemory, IpcMemoryFixture, SamplesCount, IterationsCount) {
    float getTime  = 0;
    float openTime = 0;

    if (testGetAndImportedMemory(&getTime, &openTime) < 0) {
        std::cerr << "test fail" << std::endl;
        exit(EXIT_FAILURE);
    }

    this->getBand->addValue(this->mallocSize / (getTime / 1000.0) / 1024.0 / 1024.0);
    this->getTime->addValue(getTime);
    this->openBand->addValue(this->mallocSize / (openTime / 1000.0) / 1024.0 / 1024.0);
    this->openTime->addValue(openTime);
    return;
}
