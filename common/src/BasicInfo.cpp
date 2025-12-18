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

#include "../include/BasicInfo.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>

#include "../include/Print.h"

std::string BasicInfos::getSysInfo() {
    struct utsname sname;
    int ret = uname(&sname);
    if (ret == -1) {
        return "failed to get SysInfo";
    }
    std::stringstream sysInfoStream;
    sysInfoStream << std::left << std::setw(33) << "System Name:" << sname.sysname << "\n";
    sysInfoStream << std::left << std::setw(33) << "Node Name:" << sname.nodename << "\n";
    sysInfoStream << std::left << std::setw(33) << "Release:" << sname.release << "\n";
    sysInfoStream << std::left << std::setw(33) << "Version:" << sname.version << "\n";
    sysInfoStream << std::left << std::setw(33) << "Machine:" << sname.machine << "\n";
    return sysInfoStream.str();
}

std::string BasicInfos::getMemInfo() {
    struct sysinfo memInfo;
    if (sysinfo(&memInfo) == -1) {
        return "failed to get Memory Info";
    }

    std::stringstream memoryInfoStream;
    memoryInfoStream << std::left << std::setw(33) << "Total RAM (KB):" << memInfo.totalram / 1024 << " MB\n";
    memoryInfoStream << std::left << std::setw(33) << "Free RAM (KB):" << memInfo.freeram / 1024 << " MB\n";
    memoryInfoStream << std::left << std::setw(33) << "Shared RAM (KB):" << memInfo.sharedram / 1024 << " MB\n";
    memoryInfoStream << std::left << std::setw(33) << "Buffer RAM (KB):" << memInfo.bufferram / 1024 << " MB\n";

    return memoryInfoStream.str();
}

std::string BasicInfos::getCpuInfo() {
    std::stringstream cpuInfoStream;
    if (!cpuid_present()) {
        return "Sorry, your CPU doesn't support CPUID!\n";
    }
    struct cpu_raw_data_t raw;
    struct cpu_id_t cpuInfo;

    if (cpuid_get_raw_data(&raw) < 0) {
        return ("Sorry, cannot get the CPUID raw data.\n");
    }
    if (cpu_identify(&raw, &cpuInfo) < 0) {
        return ("Sorrry, CPU identification failed.\n");
    }
    cpuInfoStream << std::left << std::setw(33) << "Vendor:" << cpuInfo.vendor_str << "\n";
    cpuInfoStream << std::left << std::setw(33) << "Brand:" << cpuInfo.brand_str << "\n";
    cpuInfoStream << std::left << std::setw(33) << "Family:" << cpuInfo.family << "\n";
    cpuInfoStream << std::left << std::setw(33) << "Model:" << cpuInfo.model << "\n";
    cpuInfoStream << std::left << std::setw(33) << "Stepping:" << cpuInfo.stepping << "\n";
    cpuInfoStream << std::left << std::setw(33) << "Ext. Family:" << cpuInfo.ext_family << "\n";
    cpuInfoStream << std::left << std::setw(33) << "Ext. Model:" << cpuInfo.ext_model << "\n";
    cpuInfoStream << std::left << std::setw(33) << "Num Cores:" << cpuInfo.num_cores << "\n";
    cpuInfoStream << std::left << std::setw(33) << "Num Logical CPUs:" << cpuInfo.num_logical_cpus << "\n";
    cpuInfoStream << std::left << std::setw(33) << "Total Logical CPUs:" << cpuInfo.total_logical_cpus << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L1 Data Cache (KB):" << cpuInfo.l1_data_cache << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L1 Instruction Cache (KB):" << cpuInfo.l1_instruction_cache << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L2 Cache (KB):" << cpuInfo.l2_cache << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L3 Cache (KB):" << cpuInfo.l3_cache << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L4 Cache (KB):" << cpuInfo.l4_cache << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L1 Assoc:" << cpuInfo.l1_assoc << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L2 Assoc:" << cpuInfo.l2_assoc << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L3 Assoc:" << cpuInfo.l3_assoc << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L4 Assoc:" << cpuInfo.l4_assoc << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L1 Cacheline:" << cpuInfo.l1_cacheline << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L2 Cacheline:" << cpuInfo.l2_cacheline << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L3 Cacheline:" << cpuInfo.l3_cacheline << "\n";
    cpuInfoStream << std::left << std::setw(33) << "L4 Cacheline:" << cpuInfo.l4_cacheline << "\n";
    cpuInfoStream << std::left << std::setw(33) << "CPU Codename:" << cpuInfo.cpu_codename << "\n";
    cpuInfoStream << std::left << std::setw(33) << "SSE Size:" << cpuInfo.sse_size << "\n";

    return cpuInfoStream.str();
}

std::string BasicInfos::getCpuInfoFromFile() {
    std::string lscpuOutput;
    std::string lscpuCommand = "lscpu";
    FILE* pipe               = popen(lscpuCommand.c_str(), "r");
    if (!pipe) {
        return "Error: Unable to execute lscpu command.\n";
    }
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        lscpuOutput += buffer;
    }
    pclose(pipe);
    std::istringstream iss(lscpuOutput);
    std::stringstream cpuInfoStream;
    std::string line;
    while (std::getline(iss, line)) {
        if (line.find("Vulnerability") != std::string::npos) {
            break;
        }
        cpuInfoStream << line << std::endl;
    }
    return cpuInfoStream.str();
}

std::string BasicInfos::getMusaVersion() {
    std::string getVersionCommand = "musa_version_query";
    FILE* pipe                    = popen(getVersionCommand.c_str(), "r");
    if (!pipe) {
        return "Error: Unable to execute musa_version_query command.\n";
    }
    std::string result;
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    return result;
}
std::string BasicInfos::getTopology() {
    std::string curPath = __FILE__;
    std::string proPath = getParentPath(curPath, 2);
    std::string figPath = proPath + "/result/pngs";
    if (proPath == "FAIL") {
        return ("FAIL to get path\n");
    }
    struct stat info;
    mode_t mode = 0777;
    if (stat(figPath.c_str(), &info) != 0 || (info.st_mode & S_IFDIR) == 0) {
        if (mkdir(figPath.c_str(), mode) != 0) {
            return ("FAIL to creat folder\n");
        }
    }
    std::string command = "lstopo -v -f --whole-io --children-order plain " + figPath + "/topologyGraph.svg";
    return !std::system(command.c_str()) ? ("### topologyGraph saved in:" + figPath + "\n")
                                         : ("### failed getTopology:" + command + "\n");
}

std::string BasicInfos::getParentPath(const std::string& curPath, int n) {
    if (n < 0) {
        return ("FAIL");
    }
    size_t lastSeparatorPos = curPath.find_last_of("/\\");
    while (n > 0 && lastSeparatorPos != std::string::npos) {
        lastSeparatorPos = curPath.find_last_of("/\\", lastSeparatorPos - 1);
        n--;
    }
    if (lastSeparatorPos != std::string::npos) {
        return curPath.substr(0, lastSeparatorPos);
    } else {
        return "FAIL";
    }
}

void BasicInfos::showBasicInfos() {
    printf("### |getSysInfo()|\n");
    std::cout << this->getSysInfo();
    printf("\n### |getMemInfo()|\n");
    std::cout << this->getMemInfo();
    printf("\n### |getCpuInfo()|\n");
    std::cout << this->getCpuInfo();
    printf("\n### |getCpuInfoFromFile()|\n");
    std::cout << this->getCpuInfoFromFile();
    printf("\n### |getMusaVersion()|\n");
    std::cout << this->getMusaVersion();
    std::cout << getTopology();
}