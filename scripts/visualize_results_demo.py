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
import pandas as pd
import matplotlib.pyplot as plt
import socket
from datetime import datetime

def main(args):
    # get csv data
    result_path = args.filename
    n = args.n
    m = args.m
    log = args.log
    output = args.output
    plot_csv_data(result_path, n, m, log, output)

def plot_csv_data(result_path, n, m, log=False, output=None):
    df = pd.read_csv(result_path)
    m_indices = [int(i) for i in m.split(",")]

    column_n = df.columns[n]
    column_m_labels = [df.columns[m] for m in m_indices]

    num_plots = len(m_indices)
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots), sharex=True)

    for i, m in enumerate(m_indices):
        ax = axes[i]
        column_m = df.columns[m]
        data_n = df[column_n]
        data_m = df[column_m]
        ax.plot(data_n, data_m, label=column_m)
        ax.set_ylabel(column_m)
        ax.legend()

    axes[-1].set_xlabel(column_n)
    plt.suptitle(f"Relationship between {column_n} and\n{', '.join(column_m_labels)}")

    if log:
        plt.xscale("log", base=2)

    if output:
        output_file = output
    else:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        project_directory = os.path.dirname(script_directory)
        result_dir = os.path.join(project_directory, "result","pngs")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        csv_file_name = os.path.basename(result_path)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hostname = socket.gethostname()
        output_file = f"{result_dir}/{csv_file_name}_{timestamp}_{hostname}.png"
    output_file_path = os.path.join(output_file)
    plt.savefig(output_file_path)
    print(f"[figure has save in file]: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot curves from a CSV file.")
    parser.add_argument("filename", type=str, help="Path to the CSV file")
    parser.add_argument("n", type=int, help="Index of the first column to plot")
    parser.add_argument("m", type=str, help="Indices of the second columns to plot, separated by commas")
    parser.add_argument("-log", action="store_true", help="Plot x-axis on a log2 scale")
    parser.add_argument("-o", "--output", type=str, help="Output file name (optional)")
    args = parser.parse_args()
    main(args)