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
#include "../include/Print.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "../include/Benchmark.h"
#include "../include/Console.h"
#include "../include/UserDefinedMeasurementCollector.h"

enum PrintConstants : size_t { ColumnSeperatorWidth = 3, DoubleDecimals = 5, NumberOfColumns = 9, ColumnWidth = 18 };

std::string PrintCenter(const std::string& s, const size_t w = PrintConstants::ColumnWidth) {
    std::stringstream ss;
    std::stringstream spaces;

    // count excess room to pad
    auto padding = w - s.size();

    for (size_t i = 0; i < padding / 2; ++i) {
        spaces << " ";
    }
    // format with padding
    ss << spaces.str() << s << spaces.str();

    // if odd #, add 1 space
    if ((padding > 0) && (padding % 2 != 0)) {
        ss << " ";
    }
    console::SetConsoleColor(console::ConsoleColor::Default);
    ss << " | ";
    return ss.str();
}

std::string PrintColumn(const double x, const size_t decDigits = PrintConstants::DoubleDecimals,
    const size_t width = PrintConstants::ColumnWidth) {
    std::stringstream ss;
    ss << std::fixed << std::right;

    // fill space around displayed #
    ss.fill(' ');

    // set  width around displayed #
    ss.width(width);

    // set # places after decimal
    ss.precision(decDigits);
    ss << x;

    console::SetConsoleColor(console::ConsoleColor::Default);
    ss << " | ";
    return ss.str();
}

///
/// http://stackoverflow.com/questions/14765155/how-can-i-easily-format-my-data-table-in-c
/// Convert double to string with specified number of places after the decimal.
///
std::string PrintColumn(const int64_t x, const size_t width = PrintConstants::ColumnWidth) {
    std::stringstream ss;
    ss << std::fixed;

    // fill space around displayed #
    ss.fill(' ');

    // set  width around displayed #
    ss.width(width);
    ss << x;

    console::SetConsoleColor(console::ConsoleColor::Default);
    ss << " | ";
    return ss.str();
}

std::string PrintColumn(const uint64_t x, const size_t width = PrintConstants::ColumnWidth) {
    std::stringstream ss;
    ss << std::fixed;

    // fill space around displayed #
    ss.fill(' ');

    // set  width around displayed #
    ss.width(width);
    ss << x;

    console::SetConsoleColor(console::ConsoleColor::Default);
    ss << " | ";
    return ss.str();
}

///
/// http://stackoverflow.com/questions/14765155/how-can-i-easily-format-my-data-table-in-c
/// Convert double to string with specified number of places after the decimal.
///
std::string
PrintStrColumnAligned(const std::string& x, const size_t width = PrintConstants::ColumnWidth, bool alignLeft = true) {
    std::stringstream ss;
    ss << std::fixed << (alignLeft ? std::left : std::right);

    // fill space around displayed #
    ss.fill(' ');

    // set width around displayed #
    ss.width(width);

    if (x.length() > width) {
        // Truncate
        std::string xTrunc = x;
        xTrunc             = xTrunc.substr(0, width);
        ss << xTrunc;
    } else {
        ss << x;
    }

    console::SetConsoleColor(console::ConsoleColor::Default);
    ss << " | ";
    return ss.str();
}

std::string PrintColumn(const std::string& x, const size_t width = PrintConstants::ColumnWidth) {
    return PrintStrColumnAligned(x, width);
}

std::string PrintColumnRight(const std::string& x, const size_t width = PrintConstants::ColumnWidth) {
    return PrintStrColumnAligned(x, width, false);
}

std::string PrintHRule(const size_t additionalColumns = 0) {
    console::SetConsoleColor(console::ConsoleColor::Default);

    std::stringstream ss;
    std::string column{":"};

    while (column.length() < PrintConstants::ColumnWidth) {
        column += "-";
    }

    std::string firstColumn = column + ":|";

    column += "-:|";

    ss << "|" << firstColumn;

    for (size_t i = 0; i < PrintConstants::NumberOfColumns + additionalColumns - 1; ++i) {
        ss << column;
    }

    ss << std::endl;

    return ss.str();
}

void Printer::Console(const std::string& x) {
    std::cout << "Celero: " << x << std::endl;
}

void Printer::TableBanner() {
    console::SetConsoleColor(console::ConsoleColor::Default);

    std::cout << "|" << PrintCenter("Group") << PrintCenter("Experiment") << PrintCenter("Size(B)")
              << PrintCenter("Samples") << PrintCenter("Iterations") << PrintCenter("Baseline")
              << PrintCenter("us/Iteration") << PrintCenter("Iterations/sec") << PrintCenter("RAM (bytes)");

    for (size_t i = PrintConstants::NumberOfColumns; i < this->columnWidths.size(); ++i) {
        std::cout << PrintCenter(this->userDefinedColumns[i - PrintConstants::NumberOfColumns], this->columnWidths[i]);
    }

    std::cout << "\n";
    std::cout << PrintHRule(this->userDefinedColumns.size());
}

void Printer::TableBanner(int num) {
    console::SetConsoleColor(console::ConsoleColor::Default);

    std::cout << "|" << PrintCenter("Group") << PrintCenter("Experiment") << PrintCenter(pbName);
    // << PrintCenter("Samples") << PrintCenter("Iterations");
    //            << PrintCenter("Baseline") << PrintCenter("us/Iteration") <<
    //            PrintCenter("Iterations/sec")
    //           << PrintCenter("RAM (bytes)");

    for (size_t i = PrintConstants::NumberOfColumns; i < this->columnWidths.size(); ++i) {
        std::cout << PrintCenter(this->userDefinedColumns[i - PrintConstants::NumberOfColumns], this->columnWidths[i]);
    }

    std::cout << "\n";
    std::cout << PrintHRule(this->userDefinedColumns.size() - 6);
}

void Printer::TableBanner(const std::vector<std::string>& strs) {
    console::SetConsoleColor(console::ConsoleColor::Default);
    for (const std::string& str : strs) {
        std::cout << PrintCenter(str);
    }
    std::cout << "\n";
}

void Printer::TableRowExperimentHeader(Experiment* x) {
    console::SetConsoleColor(console::ConsoleColor::Default);
    std::cout << "|" << PrintColumn(x->getBenchmark()->getName()) << PrintColumn(x->getName());
}

void Printer::TableRowFailure(const std::string& msg) {
    std::cout << PrintColumnRight("-") << PrintColumnRight("-") << PrintColumnRight("-");

    for (size_t i = PrintConstants::NumberOfColumns; i < this->columnWidths.size(); ++i) {
        std::cout << PrintColumnRight("-", this->columnWidths[i]);
    }

    console::SetConsoleColor(console::ConsoleColor::Red);
    std::cout << msg << std::endl;
    console::SetConsoleColor(console::ConsoleColor::Default);
}

void Printer::TableRowProblemSpaceHeader(std::shared_ptr<ExperimentResult> x) {
    console::SetConsoleColor(console::ConsoleColor::Default);

    if (x->getProblemSpaceValue() == static_cast<int64_t>(TestFixture::Constants::NoProblemSpaceValue)) {
        std::cout << PrintColumnRight("Null");
    } else {
        std::cout << PrintColumn(x->getProblemSpaceValue());
    }

    // std::cout << PrintColumn(x->getExperiment()->getSamples()) <<
    // PrintColumn(x->getProblemSpaceIterations());
}

void Printer::TableRowHeader(std::shared_ptr<ExperimentResult> x) {
    TableRowExperimentHeader(x->getExperiment());
    TableRowProblemSpaceHeader(x);
}

void Printer::TableResult(std::shared_ptr<ExperimentResult> x) {
    console::SetConsoleColor(console::ConsoleColor::Default);

    std::unordered_map<std::string, double> udmValues;

    auto udmCollector = x->getUserDefinedMeasurements();
    for (const auto& entry : udmCollector->getAggregateValues()) {
        udmValues[entry.first] = entry.second;
    }

    for (size_t i = 0; i < this->userDefinedColumns.size(); ++i) {
        const auto& fieldName = this->userDefinedColumns[i];

        if (udmValues.find(fieldName) == udmValues.end()) {
            std::cout << PrintCenter("---", this->columnWidths[i + PrintConstants::NumberOfColumns]);
        } else {
            std::cout << PrintColumn(udmValues.at(fieldName), 2, this->columnWidths[i + PrintConstants::NumberOfColumns]);
        }
    }

    std::cout << "\n";
}

void Printer::initialize(std::vector<std::string> userDefinedColumnsIn) {
    this->userDefinedColumns = userDefinedColumnsIn;

    this->columnWidths.clear();
    this->columnWidths.resize(PrintConstants::NumberOfColumns, PrintConstants::ColumnWidth);

    for (const auto& name : this->userDefinedColumns) {
        this->columnWidths.push_back(std::max(name.size() + 2, (size_t)PrintConstants::ColumnWidth));
    }
}
