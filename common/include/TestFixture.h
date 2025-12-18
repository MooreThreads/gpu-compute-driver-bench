#ifndef COMMON_TEST_FIXTURE_H
#define COMMON_TEST_FIXTURE_H
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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "./Timer.h"

class Benchmark;
class UserDefinedMeasurement;

class TestFixture {
public:
    TestFixture();
    virtual ~TestFixture();

    enum class Constants : int64_t { NoProblemSpaceValue = std::numeric_limits<int64_t>::min() };

    /// \class ExperimentValue
    /// You can derive from this type to add your own information to the experiment value set.
    class ExperimentValue {
    public:
        ExperimentValue() = default;
        ExperimentValue(int64_t v)
            : Value(v){};
        ExperimentValue(int64_t v, int64_t i)
            : Value(v)
            , Iterations(i){};
        ExperimentValue(int64_t v, int64_t i, const std::vector<uint64_t>& data)
            : Value(v)
            , Iterations(i)
            , User_Data(data){};

        virtual ~ExperimentValue() = default;
        std::vector<uint64_t>&& getUserData() { return std::move(User_Data); }
        int64_t Value{0};
        int64_t Iterations{0};
        std::vector<uint64_t> User_Data;
    };

    /// Allows a test fixture to supply values to use for experiments.
    /// This is used to create multiple runs of the same experiment
    /// and varrying the data set size, for example.  The second value
    /// of the pair is an optional override for the number of iterations
    /// to be used.  If zero is specified, then the default number of
    /// iterations is used.
    /// It is only guaranteed that the constructor is called prior to this function being called.
    virtual std::vector<TestFixture::ExperimentValue> getExperimentValues() const {
        return std::vector<TestFixture::ExperimentValue>();
    };

    /// Provide a units result scale of each experiment value.
    /// If the value is greater than 0 then additional statistic value will be printed
    /// in output - [ xxxx units/sec ]. For example for measure speed of
    /// file IO operations method might return 1024 * 1024 to get megabytes
    /// per second.
    /// It is only guaranteed that the constructor is called prior to this function being called.
    virtual double getExperimentValueResultScale() const { return 1.0; };

    /// Allows the text fixture to run code that will be executed once immediately before the
    /// benchmark.
    /// Unlike setUp, the evaluation of this function IS included in the total experiment execution
    /// time.
    /// \param x The value for the experiment.  This can be ignored if the test does not utilize
    /// experiment values.
    virtual void onExperimentStart(const TestFixture::ExperimentValue& x);

    /// Allows the text fixture to run code that will be executed once immediately after the
    /// benchmark. Unlike tearDown, the evaluation of this function IS included in the total
    /// experiment execution time.
    virtual void onExperimentEnd();

    /// Set up the test fixture before benchmark execution.
    /// This code is NOT included in the benchmark timing.
    /// It is executed once before all iterations are executed and between each Sample.
    /// Your experiment should NOT rely on "setUp" methods to be called before EACH experiment run,
    /// only between each sample.
    /// \param x The TestFixture::ExperimentValue for the experiment.  This can be ignored if the
    /// test does not utilize experiment values.
    virtual void setUp(const TestFixture::ExperimentValue& x);

    /// Internal to Celero
    void setExperimentTime(uint64_t x);

    /// Valid inside tearDown().
    uint64_t getExperimentTime() const;

    /// Internal to Celero
    void setExperimentIterations(uint64_t x);

    /// Valid inside tearDown().
    uint64_t getExperimentIterations() const;

    /// Called after test completion to destroy the fixture.
    /// This code is NOT included in the benchmark timing.
    /// It is executed once after all iterations are executed and between each Sample.
    /// Your experiment should NOT rely on "tearDown" methods to be called after EACH experiment
    /// run, only between each sample.
    virtual void tearDown();

    /// \param threads The number of working threads.
    /// \param iterations The number of times to loop over the UserBenchmark function.
    /// \param experimentValue The experiment value to pass in setUp function.
    /// \return Returns the number of microseconds the run took.
    virtual uint64_t run(uint64_t threads, uint64_t iterations, const TestFixture::ExperimentValue& experimentValue);

    /// \brief If you want to use user-defined measurements, override this method to return them
    /// This method must return a vector of pointers, one per type of user-defined measurement that
    /// you want to measure.
    virtual std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const;

    /// \brief Returns the names of all user-defined measurements in this fixture.
    std::vector<std::string> getUserDefinedMeasurementNames() const;

protected:
    /// Executed for each operation the benchmarking test is run.
    virtual void UserBenchmark();

    /// Only used for baseline cases.  Used to define a hard-coded execution time vs. actually
    /// making a measurement.
    virtual uint64_t HardCodedMeasurement() const;

private:
    uint64_t experimentIterations{0};
    uint64_t experimentTime{0};
};

#endif