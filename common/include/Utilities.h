#ifndef COMMON_UTILITIES_H
#define COMMON_UTILITIES_H
///
/// \author	John Farrier
///
/// \copyright Copyright 2015-2023 John Farrier
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
/// http://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.
///
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>

template<class T> void DoNotOptimizeAway(T&& x) {
    static auto ttid = std::this_thread::get_id();
    if (ttid == std::thread::id()) {
        const auto* p = &x;
        putchar(*reinterpret_cast<const char*>(p));
        std::abort();
    }
}

template<class T> void DoNotOptimizeAway(std::function<T(void)>&& x) {
    volatile auto foo = x();
    static auto ttid  = std::this_thread::get_id();
    if (ttid == std::thread::id()) {
        const auto* p = &foo + &x;
        putchar(*reinterpret_cast<const char*>(p));

        // If we do get here, kick out because something has gone wrong.
        std::abort();
    }
}

template<> void DoNotOptimizeAway(std::function<void(void)>&& x);

constexpr uint64_t UsPerSec(1000000);
constexpr double UsToSec{1.0e-6};
int Random();
void GenerateRandomNumbers(float* ptr, float a, float b, size_t size);

#endif