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
import visualize_results_demo as vd

script_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.dirname(script_directory)
result_dir = os.path.join(project_directory, "result", "csvs")
file_list = os.listdir(result_dir)
csv_files = [file for file in file_list if file.endswith(".csv")]

for csv_file in csv_files:
    csv_file_path = os.path.join(result_dir, csv_file)
    if("Copy1DRate" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "4,5,6,7,8", True)
    elif("Copy3DRate" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "3,4,5", True)
    elif("Copy3DArrayRate" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "3,4,5,6,7", True)
    elif("Copy2DRate" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "3,4,5", True)
    elif("SetDRate" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "3,4,5", True)
    elif("MallocRate" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "3,4", True)
    elif("CopyHostRate" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "3,4", True)
    elif("CopyRegisteredRate" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "3,4", True)
    elif("CopyPinnedRate" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "3,4", True)
    elif("efficiencyOfDependency" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "3,6", False)
    elif("cdm_parallelism" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "3,4,5", False)
    elif("kernel_launch" in csv_file_path):
        vd.plot_csv_data(csv_file_path, 2, "3,4,5", False)