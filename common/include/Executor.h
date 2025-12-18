#ifndef COMMON_EXECUTOR_H
#define COMMON_EXECUTOR_H

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

#include <memory>
#include <string>

#include "./Benchmark.h"
namespace executor {

void RunAll();
void RunAllBaselines();
bool RunBaseline(std::shared_ptr<Benchmark> x);
void RunAllExperiments();
void RunExperiments(std::shared_ptr<Benchmark> x);
void Run(std::shared_ptr<Benchmark> x);
void Run(std::shared_ptr<Experiment> x);
void Run(const std::string& group);
void Run(const std::string& group, const std::string& experiment);
} // namespace executor

#endif
