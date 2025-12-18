#ifndef COMMON_TIMER_H
#define COMMON_TIMER_H

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

#include <stdint.h>

#include "./Utilities.h"
namespace timer {

uint64_t GetSystemTime();
constexpr double ConvertSystemTime(const uint64_t x) {
    return static_cast<double>(x) * UsToSec;
}
double CachePerformanceFrequency(bool quiet);
} // namespace timer

#endif