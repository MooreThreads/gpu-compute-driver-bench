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

#ifndef COMMON_USERDEFINEDMEASUREMENTS
#define COMMON_USERDEFINEDMEASUREMENTS

#include "./Statistics.h"
#include "./UserDefinedMeasurement.h"
#include "./UserDefinedMeasurementTemplate.h"

class UDMGPUTime : public UserDefinedMeasurementTemplate<double> {
public:
    UDMGPUTime(const char* name = "T(ms)", int opt = StatsView::MEAN)
        : UserDefinedMeasurementTemplate(opt)
        , m_name(std::move(name)) {}
    virtual std::string getName() const override { return m_name; }

    virtual bool reportMean() const override { return true; }

private:
    std::string m_name;
};

class UDMCPUTime : public UserDefinedMeasurementTemplate<double> {
public:
    UDMCPUTime(const char* name = "T(ms)", int opt = StatsView::MEAN)
        : UserDefinedMeasurementTemplate(opt)
        , m_name(std::move(name)) {}
    virtual std::string getName() const override { return m_name; }

    virtual bool reportMean() const override { return true; }

private:
    std::string m_name;
};

class UDMThroughPut : public UserDefinedMeasurementTemplate<double> {
public:
    UDMThroughPut(const char* name = "TP(s^-1)", int opt = StatsView::MEAN)
        : UserDefinedMeasurementTemplate(opt)
        , m_name(std::move(name)) {}
    virtual std::string getName() const override { return m_name; }

    virtual bool reportMean() const override { return true; }

private:
    std::string m_name;
};

class UDMTFLOPS : public UserDefinedMeasurementTemplate<double> {
public:
    UDMTFLOPS(const char* name = "(TF/s)", int opt = StatsView::MEAN)
        : UserDefinedMeasurementTemplate(opt)
        , m_name(std::move(name)) {}

    UDMTFLOPS(std::string name = "(TF/s)", int opt = StatsView::MEAN)
        : UserDefinedMeasurementTemplate(opt)
        , m_name(std::move(name)) {}

    virtual std::string getName() const override { return m_name; }
    virtual bool reportMean() const override { return true; }

private:
    std::string m_name;
};

class UDMBandWidth : public UserDefinedMeasurementTemplate<double> {
public:
    UDMBandWidth(const char* name = "B(M/s)", int opt = StatsView::MEAN)
        : UserDefinedMeasurementTemplate(opt)
        , m_name(std::move(name)) {}

    UDMBandWidth(std::string name = "B(M/s)", int opt = StatsView::MEAN)
        : UserDefinedMeasurementTemplate(opt)
        , m_name(std::move(name)) {}

    virtual std::string getName() const override { return m_name; }
    virtual bool reportMean() const override { return true; }

private:
    std::string m_name;
};

class UDMCount : public UserDefinedMeasurementTemplate<int> {
public:
    UDMCount(const char* name = "count", int opt = StatsView::MEAN)
        : UserDefinedMeasurementTemplate(opt)
        , m_name(std::move(name)) {}
    virtual std::string getName() const override { return m_name; }

    virtual bool reportMean() const override { return true; }

private:
    std::string m_name;
};

class UDMUseage : public UserDefinedMeasurementTemplate<int> {
public:
    UDMUseage(std::string name = "useage", int opt = StatsView::MEAN)
        : UserDefinedMeasurementTemplate(opt)
        , m_name(std::move(name)) {}
    virtual std::string getName() const override { return m_name; }

    virtual bool reportMean() const override { return true; }

private:
    std::string m_name;
};

class UDMRatio : public UserDefinedMeasurementTemplate<double> {
public:
    UDMRatio(std::string name = "ratio", int opt = StatsView::MEAN)
        : UserDefinedMeasurementTemplate(opt)
        , m_name(std::move(name)) {}
    virtual std::string getName() const override { return m_name; }

    virtual bool reportMean() const override { return true; }

private:
    std::string m_name;
};

#endif