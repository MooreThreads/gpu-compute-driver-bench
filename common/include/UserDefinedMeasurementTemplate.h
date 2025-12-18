#ifndef USERDEFINEDMEASUREMENTTEMPLATE_H
#define USERDEFINEDMEASUREMENTTEMPLATE_H
///
/// \author	Lukas Barth, John Farrier
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
#include <numeric>
#include <type_traits>

#include "./Statistics.h"
#include "./UserDefinedMeasurement.h"

///
/// \class UserDefinedMeasurementTemplate
///
/// Base class that the user must derive user-defined measurements from.
///
/// \author	Lukas Barth, John Farrier
///

enum StatsView {
    SIZE     = 0X0001,
    MEAN     = 0X0002,
    VARIANCE = 0X0004,
    STDDEV   = 0X0008,
    SKEWNESS = 0X0010,
    KURTOSIS = 0X0020,
    ZSCORE   = 0X0040,
    MIN      = 0X0080,
    MAX      = 0X0100,
};
template<typename T> class UserDefinedMeasurementTemplate : public UserDefinedMeasurement {
    static_assert(std::is_arithmetic<T>::value, "UserDefinedMeasurementTemplate requres an arithmetic type.");

public:
    UserDefinedMeasurementTemplate(int opt)
        : viewOption(opt) {}
    virtual ~UserDefinedMeasurementTemplate() = default;

    ///
    /// \brief Must be implemented by the user. Must return a specification which aggregations the
    /// user wants to be computed.
    ///
    UDMAggregationTable getAggregationInfo() const override {
        UDMAggregationTable table;

        if (this->reportSize() == true) {
            table.push_back({"# Samp", [this]() { return static_cast<double>(this->getStatistics().getSize()); }});
        }

        if (this->reportMean() == true) {
            table.push_back({"Mean", [this]() { return this->getStatistics().getMean(); }});
        }

        if (this->reportVariance() == true) {
            table.push_back({"Var", [this]() { return this->getStatistics().getVariance(); }});
        }

        if (this->reportStandardDeviation() == true) {
            table.push_back({"StdDev", [this]() { return this->getStatistics().getStandardDeviation(); }});
        }

        if (this->reportSkewness() == true) {
            table.push_back({"Skew", [this]() { return this->getStatistics().getSkewness(); }});
        }

        if (this->reportKurtosis() == true) {
            table.push_back({"Kurtosis", [this]() { return this->getStatistics().getKurtosis(); }});
        }

        if (this->reportZScore() == true) {
            table.push_back({"ZScore", [this]() { return this->getStatistics().getZScore(); }});
        }

        if (this->reportMin() == true) {
            table.push_back({"Min", [this]() { return static_cast<double>(this->getStatistics().getMin()); }});
        }

        if (this->reportMax() == true) {
            table.push_back({"Max", [this]() { return static_cast<double>(this->getStatistics().getMax()); }});
        }

        return table;
    }

    ///
    /// \brief You must call this method from your fixture to add a measurement
    ///
    void addValue(T x) { this->stats.addSample(x); }

    ///
    /// Preserve measurements within a group/experiment/problem space set.
    ///
    void merge(const UserDefinedMeasurement* const x) override {
        const auto toMerge = dynamic_cast<const UserDefinedMeasurementTemplate<T>*>(x);
        this->stats += toMerge->stats;
    }

    T getMean() const { return this->stats.getMean(); }

    void clear() override { this->stats.reset(); }

protected:
    virtual bool reportSize() const { return StatsView::SIZE & viewOption; }
    virtual bool reportMean() const { return StatsView::MEAN & viewOption; }
    virtual bool reportVariance() const { return StatsView::VARIANCE & viewOption; }
    virtual bool reportStandardDeviation() const { return StatsView::STDDEV & viewOption; }
    virtual bool reportSkewness() const { return StatsView::SKEWNESS & viewOption; }
    virtual bool reportKurtosis() const { return StatsView::KURTOSIS & viewOption; }
    virtual bool reportZScore() const { return StatsView::ZSCORE & viewOption; }
    virtual bool reportMin() const { return StatsView::MIN & viewOption; }
    virtual bool reportMax() const { return StatsView::MAX & viewOption; }
    const Statistics<T>& getStatistics() const { return this->stats; }

private:
    /// Continuously gathers statistics without having to retain data history.
    Statistics<T> stats;
    int viewOption = StatsView::MEAN;
};

#endif