#ifndef COMMON_EXCEPTIONS_H
#define COMMON_EXCEPTIONS_H
///
/// \author	Peter Azmanov
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
#include <utility>

#include "./TestFixture.h"

///
/// A Singleton storing exception settings (currently only one flag)
///
class ExceptionSettings {
public:
    static bool GetCatchExceptions();
    static void SetCatchExceptions(bool catchExceptions);

private:
    static ExceptionSettings& instance();

private:
    bool catchExceptions{true};
};
std::pair<bool, uint64_t> RunAndCatchSEHExc(
    TestFixture& test, uint64_t threads, uint64_t calls, const TestFixture::ExperimentValue& experimentValue);

std::pair<bool, uint64_t>
RunAndCatchExc(TestFixture& test, uint64_t threads, uint64_t calls, const TestFixture::ExperimentValue& experimentValue);

#endif
