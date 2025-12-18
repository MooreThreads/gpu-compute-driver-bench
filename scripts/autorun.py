#!/usr/bin/env python3
# Copyright 2025 Moore Threads Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse

scriptPath  = os.path.dirname(os.path.abspath(__file__))
projectPath = os.path.dirname(scriptPath)

memoryOpsCases        = ["Copy1DAlignedRate", "Copy1DUnAlignedRate", "Copy2DRate", "Copy3DRate", "Copy3DArrayRate", "CopyPinnedRate",
                         "CopyRegisteredRate", "SetRate", "HostRegisterAndUnRegister", "MallocRateFrom1Bto4GB",
                         "CpuReadAndWriteRate", "GpuReadAndWriteRate", "IpcOpenMemHandle"]

mulStreamsCases       = ["streamConcurrencyCompute", "streamConcurrencyMemcpy","parallelismOfDifferentCommands",
                         "kernelLaunchMulStreamTimerByCpu", "kernelLaunchMulStreamTimerByEvent"]

graphAndScheduleCases = ["efficiencyOfGraph", "efficiencyOfSync",
                         "kernelLaunchThroughputMode", "kernelLaunchLatencyMode",
                         "moduleLoadAndGetFunction",
                         "efficiencyOfGraphLaunch","graphLaunchThroughputMode"
                         ]

resourceCases         = ["eventManage", "streamManage"]

multicardsCases       = ["CopyP2PRate", "CopySetiPj2PkRate", "kernelLaunchMulCards"]

exeDirectories        = [os.path.join(projectPath, "build", "memory"),
                         os.path.join(projectPath, "build", "schedule"),
                         os.path.join(projectPath, "build", "resource"),
                         os.path.join(projectPath, "build", "multicards")]


def main(args):
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    projectPath = os.path.dirname(scriptPath)
    resultPath = os.path.join(projectPath, "result") if not args.result else os.path.join(projectPath, args.result)

    test_suits = {
        "memoryOp": {
            "path": os.path.join(resultPath, "memoryOps"),
            "cases": memoryOpsCases,
        },
        "mulStreams": {
            "path": os.path.join(resultPath, "mulStreams"),
            "cases": mulStreamsCases,
        },
        "graphAndSchedule": {
            "path": os.path.join(resultPath, "graphAndSchedule"),
            "cases": graphAndScheduleCases,
        },
        "resourceManage": {
            "path": os.path.join(resultPath, "resource"),
            "cases": resourceCases,
        },
        "multicards": {
            "path": os.path.join(resultPath, "multicards"),
            "cases": multicardsCases,
        }
    }

    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    for suit_name in args.suits:
        if suit_name in test_suits:
            suit = test_suits[suit_name]
            if not os.path.exists(suit["path"]):
                os.makedirs(suit["path"])
            runOneTestSuit(suit["cases"], suit["path"])
        else:
            print(f"Warning: Unknown test suit '{suit_name}', skipping.")

def runOneTestSuit(cases, csvsPath):
    for case in cases:
        path = getCasePath(exeDirectories, case)
        if path is None:
            print(f"Can't find {case} executable")
            continue
        Gocommand = f"cd {os.path.dirname(path)}"
        Runcommand = f"{path} -t {csvsPath}/{case}.csv"
        # print(Gocommand)
        # print(Runcommand)
        os.chdir(os.path.dirname(path))
        if os.system(Runcommand) != 0:
            print("run fail")
            exit(-1)
    return

def getCasePath(exeDirectories, case):
    for exeDir in exeDirectories:
        for exe in os.listdir(exeDir):
            if exe==case:
                return os.path.join(exeDir, exe)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run executables and save results.")
    parser.add_argument('--result', type=str, help='The result directory,  default is projectPath/result')
    available_suits = ["memoryOp", "mulStreams", "graphAndSchedule", "resourceManage", "multicards"]

    parser.add_argument(
        '--suits',
        type=str,
        nargs='+',
        choices=available_suits,
        help=f'The test suits to run, choose from {available_suits}. Default is all.'
    )

    args = parser.parse_args()
    if args.suits is None:
        args.suits = available_suits

    main(args)
