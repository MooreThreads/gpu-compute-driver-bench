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

#ifndef COMMON_HELPER_MUSA_H_
#define COMMON_HELPER_MUSA_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <musa.h>
#include <musa_runtime.h>

static const char* _musaGetErrorEnum(musaError_t error) {
    return musaGetErrorName(error);
}

template<typename T> void check(T result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "MUSA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result),
            _musaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkMusaErrors(val) check((val), #val, __FILE__, __LINE__)

#define MUSACHECK(expected, cmd)                                                                                 \
    do {                                                                                                         \
        musaError_t e = cmd;                                                                                     \
        if (e != expected) {                                                                                     \
            printf("Failed: Musa error %s:%d '%s', expected: '%s'\n", __FILE__, __LINE__, musaGetErrorString(e), \
                musaGetErrorString(expected));                                                                   \
            exit(EXIT_FAILURE);                                                                                  \
        }                                                                                                        \
    } while (0)

#endif // COMMON_HELPER_MUSA_H_
