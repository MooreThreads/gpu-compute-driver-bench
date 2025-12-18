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

#include "../include/Celero.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <list>
#include <set>
#include <unordered_set>

#include "../include/Benchmark.h"
#include "../include/Callbacks.h"
#include "../include/CommandLine.h"
#include "../include/Console.h"

#include "../include/Exceptions.h"
#include "../include/Executor.h"

#include "../include/Print.h"
#include "../include/ResultTable.h"
#include "../include/TestVector.h"
#include "../include/UserDefinedMeasurement.h"
#include "../include/Utilities.h"

std::shared_ptr<Benchmark> RegisterTest(const char* groupName, const char* benchmarkName, const uint64_t samples,
    const uint64_t iterations, const uint64_t threads, std::shared_ptr<Factory> experimentFactory, const double target) {
    if (groupName != nullptr && benchmarkName != nullptr) {
        auto bm = TestVector::Instance()[groupName];

        if (bm == nullptr) {
            bm = std::make_shared<Benchmark>(groupName);
            TestVector::Instance().push_back(bm);
        }

        auto p = std::make_shared<Experiment>(bm);
        p->setIsBaselineCase(false);
        p->setName(benchmarkName);
        p->setSamples(samples);
        p->setIterations(iterations);
        p->setThreads(threads);
        p->setFactory(experimentFactory);
        p->setBaselineTarget(target);

        bm->addExperiment(p);

        return bm;
    }

    return nullptr;
}

std::shared_ptr<Benchmark> RegisterBaseline(const char* groupName, const char* benchmarkName, const uint64_t samples,
    const uint64_t iterations, const uint64_t threads, std::shared_ptr<Factory> experimentFactory) {
    if (groupName != nullptr && benchmarkName != nullptr) {
        auto bm = TestVector::Instance()[groupName];

        if (bm == nullptr) {
            bm = std::make_shared<Benchmark>(groupName);
            TestVector::Instance().push_back(bm);
        }

        auto p = std::make_shared<Experiment>(bm);
        p->setIsBaselineCase(true);
        p->setName(benchmarkName);
        p->setSamples(samples);
        p->setIterations(iterations);
        p->setThreads(threads);
        p->setFactory(experimentFactory);
        p->setBaselineTarget(1.0);

        bm->setBaseline(p);

        return bm;
    }

    return nullptr;
}

void Run(int argc, char** argv) {
#ifdef _DEBUG
    std::cout << "Celero is running in Debug.  Results are for debugging only as any measurements "
                 "made while in Debug are likely not representative "
                 "of non-debug results."
              << std::endl
              << std::endl;
#endif

    cmdline::parser args;
    args.add("list", 'l', "Prints a list of all available benchmarks.");
    args.add<std::string>("group", 'g', "Runs a specific group of benchmarks.", false, "");
    args.add<std::string>("outputTable", 't', "Saves a results table to the named file.", false, "");
    args.add<bool>("basicInfosView", 'b', "Prints basic info before your test", false, true);
    args.add<bool>("catchExceptions", 'e',
        "Allows Celero to catch exceptions and continue processing following benchmarks.", false, true);

    args.parse_check(argc, argv);

    if (args.exist("list") == true) {
        auto& tests = TestVector::Instance();
        std::vector<std::string> testNames;

        for (auto i = size_t(0); i < tests.size(); i++) {
            auto bm = TestVector::Instance()[i];

            if (bm != nullptr) {
                testNames.push_back(bm->getName());
            }
        }

        std::sort(std::begin(testNames), std::end(testNames));

        std::cout << "Avaliable tests:" << std::endl;
        for (auto i : testNames) {
            std::cout << "\t" << i << std::endl;
        }

        return;
    }

    // Disable dynamic CPU frequency scaling
    timer::CachePerformanceFrequency(false);

    // Show basic infos before running?
    if (args.others_exist("b") == true) {
        BasicInfos().showBasicInfos();
    }
    // Has a result output file been specified?
    auto mustCloseFile = false;
    auto argument      = args.get<std::string>("outputTable");
    if (argument.empty() == false) {
        std::cout << "- Writing results to: " << argument << std::endl;
        ResultTable::Instance().setFileName(argument);

        AddExperimentResultCompleteFunction(
            [](std::shared_ptr<ExperimentResult> p) { ResultTable::Instance().addUserOnly(p); });
        mustCloseFile = true;
    }

    // Has a flag to catch exceptions or not been specified?
    if (args.exist("catchExceptions") == true) {
        ExceptionSettings::SetCatchExceptions(args.get<bool>("catchExceptions"));
    }

    // Has a run group been specified?
    argument = args.get<std::string>("group");

    // Collect all user-defined fields
    std::vector<std::string> userDefinedFields;
    auto collectFromBenchmark = [&](std::shared_ptr<Benchmark> bmark) {
        // Collect from baseline
        auto baselineExperiment = bmark->getBaseline();
        if (baselineExperiment != nullptr) {
            auto test = baselineExperiment->getFactory()->Create();
            UserDefinedMeasurementCollector udmCollector(test);
            for (const auto& fieldName : udmCollector.getFields(test)) {
                userDefinedFields.push_back(fieldName);
            }
        }

        // Collect from all experiments
        const auto experimentSize = bmark->getExperimentSize();

        for (size_t i = 0; i < experimentSize; i++) {
            auto e = bmark->getExperiment(i);
            assert(e != nullptr);

            auto test = baselineExperiment->getFactory()->Create();
            UserDefinedMeasurementCollector udmCollector(test);
            for (const auto& fieldName : udmCollector.getFields(test)) {
                userDefinedFields.push_back(fieldName);
            }
        }
    };

    if (argument.empty() == false) {
        if (TestVector::Instance().containsGroup(argument)) {
            auto bmark = TestVector::Instance()[argument];

            if (bmark != nullptr) {
                collectFromBenchmark(bmark);
            }
        } else {
            std::cerr << "Error. The specified group \"" << argument << "\" does not exist.\n";
            return;
        }
    } else {
        for (size_t i = 0; i < TestVector::Instance().size(); i++) {
            auto bmark = TestVector::Instance()[i];

            if (bmark != nullptr) {
                collectFromBenchmark(bmark);
            }
        }
    }

    std::vector<std::string> userDefinedFieldsOrder;
    for (const std::string& field : userDefinedFields) {
        if (std::find(userDefinedFieldsOrder.begin(), userDefinedFieldsOrder.end(), field) ==
            userDefinedFieldsOrder.end()) {
            userDefinedFieldsOrder.push_back(field);
        }
    }

    Printer::get().initialize(userDefinedFieldsOrder);
    Printer::get().TableBanner(1);

    const auto startTime = timer::GetSystemTime();

    if (argument.empty() == false) {
        executor::Run(argument);
    } else {
        executor::RunAll();
    }

    const auto totalTime = timer::ConvertSystemTime(timer::GetSystemTime() - startTime);

    if (mustCloseFile == true) {
        ResultTable::Instance().closeFile();
    }

    // Final output.
    auto hours   = std::to_string(static_cast<int>(totalTime) / 3600);
    auto minutes = std::to_string((static_cast<int>(totalTime) % 3600) / 60);
    auto seconds = std::to_string(fmod(totalTime, 60.0));

    if (hours.length() < 2) {
        hours = std::string("0") + hours;
    }

    if (minutes.length() < 2) {
        minutes = std::string("0") + minutes;
    }

    if (fmod(totalTime, 60.0) < 10.0) {
        seconds = std::string("0") + seconds;
    }

    std::cout << std::endl << "Completed in " << hours << ":" << minutes << ":" << seconds << std::endl;
}
