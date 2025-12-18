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

#include "Celero.h"
#include "UserDefinedMeasurements.h"

#include "musa.h"
#include "musa_runtime.h"
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <errno.h>
#include <string>
#include <vector>
#include <string>
#include <fstream>

#include "timer_his.h"
#include "helper_musa.h"
#include "helper_musa_drvapi.h"

static const int SamplesCount    = 20;
static const int IterationsCount = 2;
#ifdef TEST_ON_NVIDIA
const std::string fileSuffix = ".ptx";
#else
const std::string fileSuffix = ".elf";
#endif

std::vector<std::string> elfFileNames = {"EmptyKernel", "basic_funcs", "basic_funcs_template"};

std::vector<std::vector<std::string>> kernelNamess = {{"_Z11EmptyKernelv"},

    {"_Z20histogramCalculationPKiPiii", "_Z17parallelReductionPKfPfi", "_Z14vectorAdditionPKfS0_Pfi",
        "_Z15matrixTransposePKfPfii", "_Z12parallelScanPKfPfi", "_Z25elementWiseMultiplicationPKfS0_Pfi",
        "_Z20matrixMultiplicationPKfS0_Pfiii"},

    {"_Z15matrixTransposeIdEvPKT_PS0_ii", "_Z15matrixTransposeIfEvPKT_PS0_ii", "_Z15matrixTransposeIjEvPKT_PS0_ii",
        "_Z15matrixTransposeImEvPKT_PS0_ii", "_Z15matrixTransposeIlEvPKT_PS0_ii", "_Z15matrixTransposeIiEvPKT_PS0_ii",
        "_Z12parallelScanIhEvPKT_PS0_i", "_Z17parallelReductionIhEvPKT_PS0_i", "_Z17parallelReductionIaEvPKT_PS0_i",
        "_Z20matrixMultiplicationIdEvPKT_S2_PS0_iii", "_Z20matrixMultiplicationItEvPKT_S2_PS0_iii",
        "_Z14vectorAdditionImEvPKT_S2_PS0_i", "_Z20matrixMultiplicationIlEvPKT_S2_PS0_iii",
        "_Z17parallelReductionIlEvPKT_PS0_i", "_Z20matrixMultiplicationIjEvPKT_S2_PS0_iii",
        "_Z25elementWiseMultiplicationIaEvPKT_S2_PS0_i", "_Z17parallelReductionImEvPKT_PS0_i",
        "_Z15matrixTransposeIsEvPKT_PS0_ii", "_Z20histogramCalculationIsEvPKT_Piii",
        "_Z20matrixMultiplicationImEvPKT_S2_PS0_iii", "_Z25elementWiseMultiplicationIdEvPKT_S2_PS0_i",
        "_Z20histogramCalculationImEvPKT_Piii", "_Z20matrixMultiplicationIfEvPKT_S2_PS0_iii",
        "_Z20matrixMultiplicationIsEvPKT_S2_PS0_iii", "_Z14vectorAdditionIiEvPKT_S2_PS0_i",
        "_Z12parallelScanIiEvPKT_PS0_i", "_Z14vectorAdditionIlEvPKT_S2_PS0_i", "_Z15matrixTransposeIhEvPKT_PS0_ii",
        "_Z12parallelScanIdEvPKT_PS0_i", "_Z20matrixMultiplicationIhEvPKT_S2_PS0_iii",
        "_Z14vectorAdditionIjEvPKT_S2_PS0_i", "_Z14vectorAdditionIaEvPKT_S2_PS0_i",
        "_Z17parallelReductionItEvPKT_PS0_i", "_Z14vectorAdditionIhEvPKT_S2_PS0_i", "_Z14vectorAdditionIsEvPKT_S2_PS0_i",
        "_Z20matrixMultiplicationIiEvPKT_S2_PS0_iii", "_Z14vectorAdditionIfEvPKT_S2_PS0_i",
        "_Z20matrixMultiplicationIaEvPKT_S2_PS0_iii", "_Z25elementWiseMultiplicationItEvPKT_S2_PS0_i",
        "_Z20histogramCalculationIjEvPKT_Piii", "_Z20histogramCalculationIaEvPKT_Piii",
        "_Z17parallelReductionIfEvPKT_PS0_i", "_Z20histogramCalculationIlEvPKT_Piii",
        "_Z14vectorAdditionIdEvPKT_S2_PS0_i", "_Z14vectorAdditionItEvPKT_S2_PS0_i",
        "_Z17parallelReductionIdEvPKT_PS0_i", "_Z25elementWiseMultiplicationIfEvPKT_S2_PS0_i",
        "_Z25elementWiseMultiplicationIsEvPKT_S2_PS0_i", "_Z25elementWiseMultiplicationIiEvPKT_S2_PS0_i",
        "_Z17parallelReductionIjEvPKT_PS0_i", "_Z25elementWiseMultiplicationIlEvPKT_S2_PS0_i",
        "_Z20histogramCalculationIiEvPKT_Piii", "_Z25elementWiseMultiplicationIhEvPKT_S2_PS0_i",
        "_Z17parallelReductionIiEvPKT_PS0_i", "_Z25elementWiseMultiplicationIjEvPKT_S2_PS0_i",
        "_Z12parallelScanIaEvPKT_PS0_i", "_Z12parallelScanIsEvPKT_PS0_i", "_Z12parallelScanIlEvPKT_PS0_i",
        "_Z12parallelScanItEvPKT_PS0_i", "_Z12parallelScanIjEvPKT_PS0_i", "_Z12parallelScanImEvPKT_PS0_i",
        "_Z15matrixTransposeItEvPKT_PS0_ii", "_Z12parallelScanIfEvPKT_PS0_i", "_Z17parallelReductionIsEvPKT_PS0_i",
        "_Z25elementWiseMultiplicationImEvPKT_S2_PS0_i", "_Z20histogramCalculationIhEvPKT_Piii",
        "_Z20histogramCalculationItEvPKT_Piii", "_Z15matrixTransposeIaEvPKT_PS0_ii"}};

int main(int argc, char** argv) {
    int deviceCount;
    checkMusaErrors(musaGetDeviceCount(&deviceCount));
    musaDeviceProp prop;
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    Printer::get().TableSetPbName("funcCount(B)");
    Run(argc, argv);
    return 0;
}

std::size_t getFileSize(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);

    if (file.is_open()) {
        std::size_t size = static_cast<std::size_t>(file.tellg());
        file.seekg(0, std::ios::beg);
        file.close();
        return size;
    } else {
        std::cerr << "Unable to open file: " << file_path << std::endl;
        return 0;
    }
}

class LoadModuleFixture : public TestFixture {
public:
    LoadModuleFixture() {}
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        std::vector<TestFixture::ExperimentValue> problemSpace;
        for (int64_t i = 0; i < elfFileNames.size(); ++i) {
            ExperimentValue ev;
            ev.User_Data.push_back(i);
            ev.Value = kernelNamess[i].size();
            problemSpace.push_back(ev);
        }
        return problemSpace;
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        codeObjectFile = std::string(elfFileNames[experimentValue.User_Data[0]]) + fileSuffix;
        kernelNames    = kernelNamess[experimentValue.User_Data[0]];
    }

    void tearDown() override {}

    void onExperimentStart(const TestFixture::ExperimentValue& x) override {}
    void onExperimentEnd() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {this->ufilesize, this->utime, this->uband};
    }

    double testModuleLoad();
    double testGetFunction();
    std::string codeObjectFile;
    std::vector<std::string> kernelNames;
    std::shared_ptr<UDMCount> ufilesize{new UDMCount("Size(B)")};
    std::shared_ptr<UDMGPUTime> utime{new UDMGPUTime("tus")};
    std::shared_ptr<UDMThroughPut> uband{new UDMThroughPut("*TP")};
};

class funcionGetFixture : public LoadModuleFixture {};

double LoadModuleFixture::testModuleLoad() {
    CPerfCounter timer;
    checkMuErrors(muInit(0));
    MUdevice device;
    checkMuErrors(muDeviceGet(&device, 0));
    MUcontext context;
    checkMuErrors(muCtxCreate(&context, 0, device));
    timer.Reset();
    timer.Start();
    MUmodule module;
    checkMuErrors(muModuleLoad(&module, codeObjectFile.c_str()));
    timer.Stop();
    checkMuErrors(muModuleUnload(module));
    double time_ms = timer.GetElapsedSeconds() * 1000.f;
    checkMuErrors(muCtxDestroy(context));
    return time_ms;
}

double LoadModuleFixture::testGetFunction() {
    CPerfCounter timer;
    checkMuErrors(muInit(0));
    MUdevice device;
    checkMuErrors(muDeviceGet(&device, 0));
    MUcontext context;
    checkMuErrors(muCtxCreate(&context, 0, device));

    MUmodule module;
    checkMuErrors(muModuleLoad(&module, codeObjectFile.c_str()));
    timer.Reset();
    timer.Start();
    for (auto& kernelName : kernelNames) {
        MUfunction function;
        checkMuErrors(muModuleGetFunction(&function, module, kernelName.c_str()));
    }
    timer.Stop();
    double time_ms = timer.GetElapsedSeconds() * 1000.f;
    checkMuErrors(muModuleUnload(module));
    checkMuErrors(muCtxDestroy(context));
    return time_ms;
}

BASELINE_F(module, module_load, LoadModuleFixture, SamplesCount, IterationsCount) {
    auto time_ms = testModuleLoad();
    ufilesize->addValue(getFileSize(codeObjectFile));
    utime->addValue(time_ms * 1000.f);
    uband->addValue(kernelNames.size() / (time_ms / 1000.f));
}

BASELINE_F(function, getFunc, funcionGetFixture, SamplesCount, IterationsCount) {
    auto time_ms = testGetFunction();
    ufilesize->addValue(getFileSize(codeObjectFile));
    utime->addValue(time_ms * 1000.f);
    uband->addValue(kernelNames.size() / (time_ms / 1000.f));
}
