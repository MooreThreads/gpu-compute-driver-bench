#ifndef COMMON_BENCHMARK_H
#define COMMON_BENCHMARK_H
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

#include <functional>
#include <memory>
#include <string>

#include "./Experiment.h"
class Benchmark {
public:
    Benchmark();
    explicit Benchmark(const std::string& name);
    explicit Benchmark(const Benchmark& other);
    ~Benchmark();
    Benchmark& operator=(const Benchmark& other);
    std::string getName() const;
    void setBaseline(std::shared_ptr<Experiment> x);
    std::shared_ptr<Experiment> getBaseline() const;
    void addExperiment(std::shared_ptr<Experiment> x);
    std::shared_ptr<Experiment> getExperiment(size_t experimentIndex);
    std::shared_ptr<Experiment> getExperiment(const std::string& experimentName);
    size_t getExperimentSize() const;

private:
    class Impl;
    Pimpl<Impl> pimpl;
};

#endif