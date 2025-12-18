## 🚀 GPU Compute Driver Bench

GPU Compute Driver Bench is a comprehensive performance evaluation suite focused on assessing GPU compute driver performance from practical development and deployment perspectives.
It supports both Moore Threads MUSA drivers and CUDA-compatible GPU drivers, enabling fair and repeatable cross-platform evaluation.

### ✨ Key Features

🧠 **Realistic Workloads**
Covers diverse compute and memory scenarios closely aligned with real-world usage patterns.

⚙️ **Multi-Dimensional Driver Evaluation**
Evaluates driver performance, resource management, and execution efficiency across multiple dimensions.

📊 **Standardized Metrics & Baselines**
Provides standardized metrics and baselines for comparing hardware and software optimizations.

🔬 **Granular & Holistic Analysis**
Enables both fine-grained subsystem testing and holistic, end-to-end performance assessment.

🔁 **Cross-Platform Compatibility**
Supports cross-platform evaluation with MUSA and CUDA-compatible GPU drivers.

📈 **Automated Performance Scoring**
Provides a scoring system for automated performance regression tracking across driver and hardware versions.

## File structure

- common

  This directory contains the core library based on the Celero framework, providing benchmark infrastructure including test fixtures, timing utilities, result collection, and printing facilities. Both header and source files are included to expose interfaces and implementation for convenient user access.

- schedule

  Benchmarks focused on kernel execution and scheduling performance:
  1. Kernel execution time measurements for different types of kernels.
  2. Evaluation of memcpy and kernel co-execution scenarios to analyze dependency resolution efficiency.
  3. Event synchronization efficiency between operations in different streams.
  4. First kernel launch latency to assess module loading performance.
  5. Graph optimization benefits for small kernel launch performance.
  6. Kernel gap analysis and multi-stream execution concurrency.
  7. Module loading and function retrieval performance.

- memory

  Comprehensive memory operation benchmarks:
  1. Memory allocation/deallocation performance across sizes from 1B to 4GB.
  2. Memory reuse efficiency after releasing and reapplying same-sized allocations.
  3. Bandwidth measurements for 1D aligned/unaligned, 2D, 3D, and array memory copies.
  4. Host and device memory read/write performance.
  5. Pinned memory and registered host memory copy performance.
  6. Inter-process memory handle operations.
  7. Memory set/clear performance benchmarks.

- multicards

  Multi-GPU benchmarks for evaluating cross-device performance:
  1. P2P memory copy bandwidth and latency.
  2. Cross-device memory set operations.
  3. Complex multi-card kernel launching scenarios.

- resource

  Resource management benchmarks:
  1. Event management performance.
  2. Stream management and concurrency.

- scripts

  Utility scripts for automation and result processing:
  1. Automated test execution (autorun.py).
  2. Performance scoring calculation (calculateScoreOfSuit.py).
  3. Result visualization tools (csv2pngs.py, visualize_results_demo.py).
  4. Code porting utilities between CUDA and MUSA (porting2cuda.sh, porting2musa.sh).

## How to build
- Run `sudo apt install libcpuid-dev` to install libcpuid for getting some cpu infos.

- Run `sudo apt-get install libeigen3-dev` to install Eigen.

- Run `sudo apt-get install hwloc` to intall topology graph tool.

- Run `sudo apt install libblas-dev libopenblas-base libopenblas-dev` to install blas lib.

- Run the  script `install.sh` in  gpu-compute-driver-bench. add `-m`(musa) / `-n`(cuda)

  > [!IMPORTANT]
  >
  > 1. Make sure that the MUSA Toolkits or CUDA Toolkits are properly installed.
  >
  > 2. We provide a script under `scripts/` for one-step conversion between CUDA and MUSA code. It depends on the `musify` tool and is ready to use.

- You can get the usage with -h eg. `./install -h`, the the message below will be showed.

  > [!TIP]
  >
  > Some device functions have been precompiled. If you want to run on other architectures, please modify `gpu-compute-driver-bench/schedule/elf/gen.sh`, rerun `./gen.sh`, and specify the `-R` option when running `./install.sh`.

  ```shell
  Usage: ./install.sh [OPTIONS]
  Options:
    -R       : Rebuild (clean build directory and rebuild)
    -j N     : Number of parallel jobs for make (default: j12)
    -h       : Display this help message
    -m       : Enable MCC Compiler (default: OFF)
    -n       : Enable NVCC Compiler (default: OFF)
    -d       : Enable Debug Mode (default: OFF)
  ```

## How to run a case

- You can run each case by the command like ./programeName [-t tableName.csv] [b]
`[opt]` means you can choose it or not
`[-t tableName.csv]` means save the result to table if you need.
`[b]` means show the basic info of your environment if you need.
`[-h]` is also provided for printing the help messages.

## How to get score

- Run the cases and collect the results.

  ```sh
  $ cd scripts/
  $ python3 autorun.py -h
  usage: autorun.py [-h] [--result RESULT] [--suits {memoryOp,mulStreams,graphAndSchedule} [{memoryOp,mulStreams,graphAndSchedule} ...]]

  Run executables and save results.

  options:
    -h, --help            show this help message and exit
    --result RESULT       The result directory, default is projectPath/result
    --suits {memoryOp,mulStreams,graphAndSchedule} [{memoryOp,mulStreams,graphAndSchedule} ...]
                          The test suits to run, choose from ['memoryOp', 'mulStreams', 'graphAndSchedule']. Default is all.
  ```

- Calculate the score.

  ```shell
  $ cd scripts/
  $ python3 calculateScoreOfSuit.py -h
  usage: calculateScoreOfSuit.py [-h] [--base BASE] [--test TEST] [--score SCORE] [--config CONFIG]

  Calculate the score of test cases.

  options:
    -h, --help       show this help message and exit
    --base BASE      The basic result table fold, default is ../baseline/
    --test TEST      The test result table fold, default is ../result/
    --score SCORE    The score path, default is projectPath/score

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

This project includes code from the Celero project (https://github.com/DigitalInBlue/Celero), which is also licensed under the Apache License 2.0.
Copyright 2015-2023 John Farrier

Additional third-party libraries used in this project:
- libcpuid: Copyright 2008-2013 Veselin Georgiev, licensed under BSD License
- Eigen: Copyright (C) 2008 Gael Guennebaud, licensed under MPL2
- hwloc: Copyright 2006-2021 The University of Tennessee, licensed under BSD License
- OpenBLAS: Copyright 2015-2021 OpenBLAS project, licensed under BSD License