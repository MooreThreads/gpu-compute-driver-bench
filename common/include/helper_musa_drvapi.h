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

#ifndef COMMON_HELPER_MUSA_DRVAPI_H_
#define COMMON_HELPER_MUSA_DRVAPI_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstring>
#include <iostream>
#include <sstream>

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#ifndef checkMuErrors
#define checkMuErrors(err) __checkMusaErrors(err, __FILE__, __LINE__)

inline void __checkMusaErrors(MUresult err, const char* file, const int line) {
    if (MUSA_SUCCESS != err) {
        const char* errorStr = NULL;
        muGetErrorString(err, &errorStr);
        fprintf(stderr,
            "checkMusaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}
#endif

#endif // COMMON_HELPER_MUSA_DRVAPI_H_
