#ifndef COMMON_RESULTTABLE_H
#define COMMON_RESULTTABLE_H

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

#include <string>

#include "./Experiment.h"
#include "./Pimpl.h"

///
/// \class ResultTable
///
/// \author	John Farrier
///
class ResultTable {
public:
    static ResultTable& Instance();
    void setFileName(const std::string& x);
    void closeFile();
    void add(std::shared_ptr<ExperimentResult> x);
    void addUserOnly(std::shared_ptr<ExperimentResult> x);
    void save();

private:
    ResultTable();
    ~ResultTable();

    ResultTable(const ResultTable&)                  = delete;
    ResultTable& operator=(ResultTable const& other) = delete;
    class Impl;
    Pimpl<Impl> pimpl;
};

#endif
