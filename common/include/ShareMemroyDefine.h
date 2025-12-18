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

#ifndef COMMON_SHARE_MEMORY_DEFINE_H
#define COMMON_SHARE_MEMORY_DEFINE_H
#include "musa.h"
#include "musa_runtime.h"

#include <atomic>
struct ShareMemHandleInfo {
    musaIpcMemHandle_t handler;
    std::atomic_bool child_wake;
    std::atomic_bool parent_wake;
    int64_t mallocSize;
    float openTime;
};

#endif