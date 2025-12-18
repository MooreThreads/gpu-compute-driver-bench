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


import csv
import os
import argparse
import math
import json
import matplotlib.pyplot as plt

def calculateScoreOfSuit(testSuitConfig: dict, basicResutTableFold, testResultTableFlod, scoreFlod, scoreTable, histogramData):
    # 01. find how many result tables in the basic result table fold
    basicFiles = os.listdir(basicResutTableFold)
    testFiles  = os.listdir(testResultTableFlod)

    # 02. check the file in test suit has test on the basic suit
    for testFile in testFiles:
        if testFile not in basicFiles:
            print("[Warning] the test file %s is not in the basic suit" % testFile)
            testFiles.remove(testFile)
        # elif testFile not in testSuitConfig:
        elif not any(case['caseName'] == remove_file_suffix(testFile) for case in testSuitConfig['testCases']):
            print("[Warning] the test file %s is not in the config" % testFile)
            testFiles.remove(testFile)
        else :
            basicFilePath = os.path.join(basicResutTableFold, testFile)
            testFilePath = os.path.join(testResultTableFlod, testFile)

            basicTable = read_csv_to_list(basicFilePath)
            testTable = read_csv_to_list(testFilePath)

            if len(basicTable) == 0 or len(basicTable) != len(testTable) or any(len(basicRow) != len(testRow) for basicRow, testRow in zip(basicTable, testTable)):
                print("[Warning] the test file %s has mismatched rows or columns" % testFile)
                print("[Warning] the basic table has %d rows and %d columns" % (len(basicTable), len(basicTable[0])))
                print("[Warning] the test table has %d rows and %d columns" % (len(testTable), len(testTable[0])))
                testFiles.remove(testFile)

    # 03. calculate the score of per test case
    scoreTable.append(["testName", "testScore"])
    scoreOfCurrentSuit = 0
    for testFile in testFiles:
        score = [0]
        # resCsv = calculateScoreOfExecutableTestCases(testSuitConfig[testFile]['baseScore'], basicResutTableFold + testFile, testResultTableFlod + testFile, score)
        resCsv = calculateScoreOfExecutableTestCases(
            next((case for case in testSuitConfig['testCases'] if case['caseName'] == remove_file_suffix(testFile)))['baseScore'],
            basicResutTableFold + testFile,
            testResultTableFlod + testFile,
            score,
            histogramData)
        write_list_to_csv(scoreFlod + testFile, resCsv)
        scoreOfCurrentSuit += score[0]
        scoreTable.append([testFile, score[0]])
        print(f"    [cases:] The score of case {testFile} is {score[0]}")
    scoreTable.append(["sum", scoreOfCurrentSuit])
    return scoreOfCurrentSuit

# function: calculateScoreOfExecutableTestCases
def calculateScoreOfExecutableTestCases(basicScoreOfTest: float, basicResutTableFile, testResultTableFile, scoreContainer, histogramData):
    # 01. read the basic result table and test result table
    basicResutTable = read_csv_to_list(basicResutTableFile)
    testResultTable = read_csv_to_list(testResultTableFile)

    # 02. find the focus columns and calculate the score of per element
    focusColumn = find_focus_columns(basicResutTable[0])
    scoreOfPerElement = float(basicScoreOfTest) / (len(focusColumn) * (len(basicResutTable) - 1))


    # 03. copy the list of result table and calculate the score of each test case
    # 0301. add the title of the new columns
    testResultTableCopy = testResultTable.copy()
    title = testResultTableCopy[0]
    for colIndex in focusColumn:
        title.append(title[colIndex] + 'basic')
        title.append(title[colIndex] + 'score')

    # 0302. calculate the score of each test case
    socreOfExecutableTestCase = float(0)
    for rowIndex in range(1, len(testResultTableCopy)):
        row = testResultTableCopy[rowIndex]
        for colIndex in focusColumn:
            basicValue = float(basicResutTable[rowIndex][colIndex])
            testValue  = float(testResultTable[rowIndex][colIndex])
            scoreRadio = calculateRadio(basicValue, testValue)
            histogramData.append(scoreRadio)
            ##print("basicValue: %f, testValue: %f, scoreRadio: %f" % (basicValue, testValue, scoreRadio))
            socreOfExecutableTestCase += (scoreRadio * scoreOfPerElement)
            row.append(scoreOfPerElement)
            row.append(float(scoreRadio * scoreOfPerElement))
    scoreContainer[0] = socreOfExecutableTestCase
    return testResultTableCopy

# function: printfDim2List for debug
def printfDim2List(dim2List):
    for row in dim2List:
        for element in row:
            print(element, end=' ')
        print()

def remove_file_suffix(filename):
    return os.path.splitext(filename)[0]

#function: read_csv_to_list
def read_csv_to_list(file_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        return [row for row in reader]

# function: write_list_to_csv
def write_list_to_csv(file_path, data):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# function: find_focus_columns
def find_focus_columns(header_row):
    focus_columns = []
    for index, column_name in enumerate(header_row):
        if column_name.startswith('*'):
            focus_columns.append(index)
    return focus_columns

def calculateRadio(baseLine, test):
    #return test / baseLine
    return 1 + math.log10(test / baseLine)

def parseWeightConfig(configPath, testResultTableFold):
    with open(configPath, 'r') as configFile:
        configDict = json.load(configFile)
    testSuitConfigs = configDict['testSuitConfigs']
    for testSuitConfig in testSuitConfigs:
        baseScore = testSuitConfig['baseScore']
        testCases = testSuitConfig['testCases']
        # diff
        for case in testCases:
            casePath = testResultTableFold + '/' + testSuitConfig['suitName'] + '/' + case['caseName'] + '.csv'
            if not os.path.exists(casePath):
                print("[Warning] The test suit %s case %s does not exists" % (testSuitConfig['suitName'], case['caseName']))
                case['scoreWeight'] = 0
        totalWeight = sum(case['scoreWeight'] for case in testCases)
        if (totalWeight == 0) : continue
        for case in testCases:
            case['baseScore'] = (case['scoreWeight'] / totalWeight) * baseScore
    return testSuitConfigs

def drawHistogram(histogramData):
    plt.figure(figsize=(9,6), dpi=100)
    plt.hist(histogramData, 25, color='w', edgecolor='k', hatch=r'ooo',label='frequency')
    plt.savefig("./Histogram.svg")

def main(args):
    basicResutTableFold = args.base if args.base else '../baseline/'
    testResultTableFold = args.test if args.test else '../result/'
    scoreFlod           = args.score if args.score else '../score/'
    configPath          = args.config if args.config else '../TestSuitConfig.json'

    scoreSuitTable      = [] # the score table of the test suit, it will record the score of each test suit
    totalScoreOfAllSuit = 0
    totalBaseOfAllSuit  = 0
    histogramData       = []

    testSuitConfigs = parseWeightConfig(configPath, testResultTableFold)
    if not os.path.exists(scoreFlod):
        os.makedirs(scoreFlod)

    TestSubdirectories  = [d for d in os.listdir(testResultTableFold) if os.path.isdir(os.path.join(testResultTableFold, d))]

    scoreSuitTable.append(["testSuit", "baseScore", "totalScore"])
    for testSuit in TestSubdirectories:
        testSuitPath = os.path.join(scoreFlod, testSuit)
        if not os.path.exists(testSuitPath):
            os.makedirs(testSuitPath)
        scoreTestTable = []
        testSuitConfig = next((config for config in testSuitConfigs if config['suitName'] == testSuit), None)
        if testSuitConfig is None:
            print("[Warning] The test suit %s is not in the configs" % testSuit)
            continue
        baseLineDir = basicResutTableFold + '/' + testSuitConfig['baseline'] + '/' + testSuit + '/'
        if not os.path.isdir(baseLineDir):
            print("[Warning] The test suit %s is not in the basic suit, baseline: %s" % (testSuit, testSuitConfig['baseline']))
            continue
        score = calculateScoreOfSuit(testSuitConfig,
                                    baseLineDir,
                                    testResultTableFold + '/' + testSuit + '/',
                                    testSuitPath + '/', scoreTestTable, histogramData)
        write_list_to_csv(scoreFlod + '/' + testSuit + '.csv', scoreTestTable)
        baseScoreOfSuit = next((suite['baseScore'] for suite in testSuitConfigs if suite['suitName'] == testSuit), None)
        totalBaseOfAllSuit += baseScoreOfSuit
        scoreSuitTable.append([testSuit, baseScoreOfSuit, score])
        totalScoreOfAllSuit += score
        print("[suits] The score of %s is %f\n" % (testSuit, score))
    scoreSuitTable.append(["sum", totalBaseOfAllSuit ,totalScoreOfAllSuit])
    write_list_to_csv(scoreFlod + '/' + 'totalScore.csv', scoreSuitTable)
    print("[Summary] The total score of all test suit is %f" % totalScoreOfAllSuit)
    drawHistogram(histogramData)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the score of test cases.")
    parser.add_argument('--base',   type=str, help='The basic result table fold, default is ../baseline/')
    parser.add_argument('--test',   type=str, help='The test result table fold,  default is ../result/')
    parser.add_argument('--score',  type=str, help='The score path, default is projectPath/score')
    parser.add_argument('--config', type=str, help='The config path, default is projectPath/TestSuitConfig.json')
    args = parser.parse_args()
    main(args)
