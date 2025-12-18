#!/bin/bash

#
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

BUILD_DIR="build"
DEFAULT_JOBS=12
ENABLE_MCC=OFF
ENABLE_NVCC=OFF
ENABLE_DEBUG=OFF

# Function to print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -R       : Rebuild (clean build directory and rebuild)"
    echo "  -j N     : Number of parallel jobs for make (default: j12)"
    echo "  -h       : Display this help message"
    echo "  -m       : Enable MCC Compiler (default: OFF)"
    echo "  -n       : Enable NVCC Compiler (default: OFF), Cannot enable both MCC and NVCC at the same time"
    echo "  -d       : Enable Debug Mode (default: OFF)"
    exit 1
}

# Process command line arguments
while getopts "Rj:nmdh" opt; do
    case $opt in
        R)
            REBUILD=true
            ;;
        j)
            JOBS=$OPTARG
            ;;
        m)
            ENABLE_MCC=ON
            ;;
        n)
            ENABLE_NVCC=ON
            ;;
        d)
            ENABLE_DEBUG=ON
            ;;
        h)
            print_usage
            ;;
        *)
            print_usage
            ;;
    esac
done

if [ -z "$JOBS" ]; then
    JOBS=$DEFAULT_JOBS
fi

# Check if the build directory exists, if not, create it
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating '$BUILD_DIR' directory..."
    mkdir "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. "-DENABLE_MCC=$ENABLE_MCC" "-DENABLE_DEBUG=$ENABLE_DEBUG" "-DENABLE_NVCC=$ENABLE_NVCC"
    cd ..
fi

# Navigate to build directory
cd "$BUILD_DIR"

# If REBUILD is true, clean build directory and rebuild
if [ "$REBUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf *
    cmake .. "-DENABLE_MCC=$ENABLE_MCC" "-DENABLE_DEBUG=$ENABLE_DEBUG" "-DENABLE_NVCC=$ENABLE_NVCC"
    make "-j$JOBS"
else
    # Build without cleaning
    echo "building with $JOBS JOBS"
    make "-j$JOBS"
fi

echo "Build completed."