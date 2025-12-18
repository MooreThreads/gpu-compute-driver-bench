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
test_path=$(dirname $(pwd))
# echo ${MUSA_PORTING_PATH}
PORTING_PATH=${MUSA_PORTING_PATH}
# echo "your porting-tool path = ${PORTING_PATH}/build/musify-text"

if [[ "x${MUSA_PORTING_PATH}" == "x" ]]; then
    PORTING_PATH=/usr/local/musa/tools
    echo your porting-tool path = ${PORTING_PATH}/musify-text
    ${PORTING_PATH}/musify-text -m ${PORTING_PATH}/general.json ${PORTING_PATH}/include.json ${PORTING_PATH}/extra.json --inplace -- \
`find ${test_path}/memory/ ${test_path}/common/ ${test_path}/resource/ ${test_path}/schedule/ ${test_path}/multicards \
-name '*.cu' -o -name '*.h' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.c' | grep -v "mtcc/mtcc_test"`
else
    PORTING_PATH=${MUSA_PORTING_PATH}
    last_char=${PORTING_PATH: -1}
    if [[ ${last_char} == "/" ]];then
    echo your porting-tool path = ${PORTING_PATH}build/musify-text
    ${PORTING_PATH}build/musify-text -m ${PORTING_PATH}mapping/general.json ${PORTING_PATH}mapping/include.json  ${PORTING_PATH}mapping/extra.json --inplace -- \
`find ${test_path}/memory/ ${test_path}/common/ ${test_path}/resource/ ${test_path}/schedule/ ${test_path}/multicards \
-name '*.cu' -o -name '*.h' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.c' | grep -v "mtcc/mtcc_test"`
    else
    echo your porting-tool path = ${PORTING_PATH}/build/musify-text
    ${PORTING_PATH}/build/musify-text -m ${PORTING_PATH}/mapping/general.json ${PORTING_PATH}/mapping/include.json ${PORTING_PATH}/mapping/extra.json --inplace -- \
`find ${test_path}/memory/ ${test_path}/common/ ${test_path}/resource/ ${test_path}/schedule/ ${test_path}/multicards \
-name '*.cu' -o -name '*.h' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.c' | grep -v "mtcc/mtcc_test"`
    fi
fi
echo "PORTING CUDA => MUSA"
echo "PORTING SUCCESS"
