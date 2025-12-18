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

#include "ShareMemroyDefine.h"
#include "helper_musa.h"

#include <iostream>
#include <sys/shm.h>

int main() {
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
    while (!shareMem->child_wake.load(std::memory_order_seq_cst)) {}
    void* slavePtr = nullptr;
    musaEvent_t start;
    musaEvent_t stop;
    checkMusaErrors(musaEventCreate(&start));

    checkMusaErrors(musaEventCreate(&stop));

    checkMusaErrors(musaEventRecord(start, 0));

    checkMusaErrors(musaIpcOpenMemHandle(&slavePtr, shareMem->handler, musaIpcMemLazyEnablePeerAccess));
    checkMusaErrors(musaEventRecord(stop, 0));
    checkMusaErrors(musaEventSynchronize(stop));

    auto mallocSize = shareMem->mallocSize;
    char* slaveData = reinterpret_cast<char*>(malloc(mallocSize));
    checkMusaErrors(musaMemcpy(slaveData, reinterpret_cast<char*>(slavePtr), mallocSize, musaMemcpyDeviceToHost));

    for (int i = 0; i < mallocSize; i++) {
        if (reinterpret_cast<char*>(slaveData)[i] != 11) {
            std::cerr << "test ipc memory but data not equal!" << std::endl;
            return -1;
        }
    }
    free(slaveData);
    checkMusaErrors(musaEventElapsedTime(&(shareMem->openTime), start, stop));
    checkMusaErrors(musaIpcCloseMemHandle(slavePtr));
    // sem_post(sem_ob2);
    shareMem->parent_wake.store(true, std::memory_order_seq_cst);
    if (shmdt(sm) == -1) {
        std::cerr << "shmdt fail" << std::endl;
        return -1;
    }
    return 0;
}