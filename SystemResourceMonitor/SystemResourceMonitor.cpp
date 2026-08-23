#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <ctime>
#include <tlhelp32.h>
#include <iomanip>
#include <sstream>
#include <vector>
#include <conio.h>
#include <algorithm>  

std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    struct tm timeInfo;
    localtime_s(&timeInfo, &t);
    std::ostringstream oss;
    oss << std::put_time(&timeInfo, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}   
double getCPULoad() {
    static PDH_HQUERY cpuQuery = NULL;
    static PDH_HCOUNTER cpuTotal;
    static bool initialized = false;

    if (!initialized) {
        PdhOpenQuery(NULL, 0, &cpuQuery);
        PdhAddEnglishCounter(cpuQuery, L"\\Processor(_Total)\\% Processor Time", 0, &cpuTotal);
        PdhCollectQueryData(cpuQuery);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        initialized = true;
    }

    PDH_FMT_COUNTERVALUE counterVal;
    PdhCollectQueryData(cpuQuery);
    PdhGetFormattedCounterValue(cpuTotal, PDH_FMT_DOUBLE, NULL, &counterVal);
    return counterVal.doubleValue;
}

double getMemoryUsagePercent() {
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memStatus);
    DWORDLONG totalPhys = memStatus.ullTotalPhys;
    DWORDLONG availPhys = memStatus.ullAvailPhys;
    DWORDLONG usedPhys = totalPhys - availPhys;
    return (double)usedPhys / (double)totalPhys * 100.0;
}

struct BatteryInfo {
    int percent;
    bool isCharging;
    bool hasBattery;
};

BatteryInfo getBatteryInfo() {
    SYSTEM_POWER_STATUS sps;
    BatteryInfo info = { -1, false, false };
    if (GetSystemPowerStatus(&sps)) {
        info.hasBattery = (sps.BatteryFlag != 128);
        info.percent = (int)sps.BatteryLifePercent;
        if (info.percent > 100) info.percent = -1;
        info.isCharging = (sps.ACLineStatus == 1);
    }
    return info;
}

void logToCSV(const std::string& filename, double cpu, double memory, const BatteryInfo& battery) {
    std::ofstream file;
    std::ifstream checkFile(filename);
    bool fileExists = checkFile.good();
    checkFile.close();
    file.open(filename, std::ios::app);
    if (!fileExists) {
        file << "Timestamp,CPU_Load(%),Memory_Usage(%),Battery(%),Charging\n";
    }
    file << getTimestamp() << ","
        << std::fixed << std::setprecision(2) << cpu << ","
        << std::fixed << std::setprecision(2) << memory << ","
        << (battery.hasBattery ? std::to_string(battery.percent) : "N/A") << ","
        << (battery.isCharging ? "Yes" : "No") << "\n";
    file.close();
}
struct ProcessInfo {
    std::string name;
    DWORD pid;
    double cpuPercent;
    SIZE_T memoryMB;
};

std::vector<ProcessInfo> getRunningProcesses() {
    std::vector<ProcessInfo> processes;

    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return processes;

    PROCESSENTRY32 pe;
    pe.dwSize = sizeof(PROCESSENTRY32);

    if (!Process32First(snapshot, &pe)) {
        CloseHandle(snapshot);
        return processes;
    }

    do {
        ProcessInfo info;

        // Convert process name from wide to narrow string
        char nameBuf[256];
        wcstombs_s(nullptr, nameBuf, pe.szExeFile, 255);
        info.name = nameBuf;
        info.pid = pe.th32ProcessID;

        // Get memory usage
        HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pe.th32ProcessID);
        if (hProcess) {
            PROCESS_MEMORY_COUNTERS pmc;
            if (GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
                info.memoryMB = pmc.WorkingSetSize / (1024 * 1024); // bytes → MB
            }
            else {
                info.memoryMB = 0;
            }
            CloseHandle(hProcess);
        }
        else {
            info.memoryMB = 0;
        }

        info.cpuPercent = 0.0; // updated below
        processes.push_back(info);

    } while (Process32Next(snapshot, &pe));

    CloseHandle(snapshot);

    // Sort by memory usage descending
    std::sort(processes.begin(), processes.end(), [](const ProcessInfo& a, const ProcessInfo& b) {
        return a.memoryMB > b.memoryMB;
        });

    return processes;
}
void printProcesses(const std::vector<ProcessInfo>& processes, int topN = 15) {
    std::cout << "\n----------- TOP " << topN << " PROCESSES BY MEMORY -----------\n";
    std::cout << std::left
        << std::setw(8) << "PID"
        << std::setw(35) << "Process Name"
        << std::setw(15) << "Memory (MB)"
        << "\n";
    std::cout << std::string(58, '-') << "\n";

    int count = 0;
    for (const auto& p : processes) {
        if (count >= topN) break;
        std::cout << std::left
            << std::setw(8) << p.pid
            << std::setw(35) << p.name
            << std::setw(15) << p.memoryMB
            << " MB\n";
        count++;
    }

    std::cout << "\nTotal processes running: " << processes.size() << "\n";
    std::cout << std::string(58, '-') << "\n\n";
}
void logProcessesToCSV(const std::string& filename, const std::vector<ProcessInfo>& processes) {
    std::ofstream file;
    std::ifstream checkFile(filename);
    bool fileExists = checkFile.good();
    checkFile.close();

    file.open(filename, std::ios::app);

    if (!fileExists) {
        file << "Timestamp,PID,ProcessName,Memory_MB\n";
    }

    std::string ts = getTimestamp();
    int count = 0;
    for (const auto& p : processes) {
        if (count >= 30) break; // log top 30 processes only
        file << ts << ","
            << p.pid << ","
            << p.name << ","
            << p.memoryMB << "\n";
        count++;
    }

    file.close();
}
void checkPythonAlerts(const std::string& alertFile) {
    std::ifstream file(alertFile);
    if (!file.is_open()) return;

    std::string line;
    std::vector<std::string> criticals;
    std::vector<std::string> warnings;
    std::string timestamp;

    while (std::getline(file, line)) {
        if (line.find("TIMESTAMP:") == 0) {
            timestamp = line.substr(10);
        }
        else if (line.find("CRITICAL:") == 0) {
            criticals.push_back(line.substr(9));
        }
        else if (line.find("WARNING:") == 0) {
            warnings.push_back(line.substr(8));
        }
    }
    file.close();

    if (criticals.empty() && warnings.empty()) return;

    std::cout << "\n";
    std::cout << "!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=\n";

    if (!criticals.empty()) {
        std::cout << "  ** CRITICAL ALERTS **\n";
        for (const auto& a : criticals) {
            // format: proc:mem:avg:tier:z
            std::istringstream ss(a);
            std::string proc, mem, avg, tier, z;
            std::getline(ss, proc, ':');
            std::getline(ss, mem, ':');
            std::getline(ss, avg, ':');
            std::getline(ss, tier, ':');
            std::getline(ss, z, ':');
            std::cout << "  [CRITICAL] " << proc
                << " | Using: " << mem << " MB"
                << " | Normal: " << avg << " MB"
                << " | Tier: " << tier << "\n";
        }
    }

    if (!warnings.empty()) {
        std::cout << "  ** WARNINGS **\n";
        for (const auto& a : warnings) {
            std::istringstream ss(a);
            std::string proc, mem, avg, tier, z;
            std::getline(ss, proc, ':');
            std::getline(ss, mem, ':');
            std::getline(ss, avg, ':');
            std::getline(ss, tier, ':');
            std::getline(ss, z, ':');
            std::cout << "  [WARNING]  " << proc
                << " | Using: " << mem << " MB"
                << " | Normal: " << avg << " MB"
                << " | Tier: " << tier << "\n";
        }
    }

    std::cout << "!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=\n\n";
}


int main() {
    const std::string csvFile = "system_log.csv";
    const int intervalSeconds = 3;

    std::cout << "========================================\n";
    std::cout << "   Predictive System Resource Monitor   \n";
    std::cout << "========================================\n";
    std::cout << "\n  [P] Show Processes\n";
    std::cout << "  [S] Show System Stats Now\n";
    std::cout << "  [Q] Quit\n";
    std::cout << "\nMonitoring started... Press a key anytime.\n\n";

    bool running = true;

    while (running) {
        double cpu = getCPULoad();
        double mem = getMemoryUsagePercent();
        BatteryInfo bat = getBatteryInfo();
        logToCSV(csvFile, cpu, mem, bat);
w        checkPythonAlerts("D:\\PROJECTS\\SystemResourceMonitor\\x64\\Debug\\alerts.txt");

        // Also silently log processes every cycle
        auto processes = getRunningProcesses();
        logProcessesToCSV("process_log.csv", processes);

        // Check if user pressed a key
        if (_kbhit()) {
            char key = _getch();
            key = toupper(key);

            if (key == 'P') {
                auto processes = getRunningProcesses();
                printProcesses(processes, 15);

            }
            else if (key == 'S') {
                std::cout << "\n========== SYSTEM STATS ==========\n";
                std::cout << "Time    : " << getTimestamp() << "\n";
                std::cout << "CPU     : " << std::fixed << std::setprecision(1) << cpu << "%\n";
                std::cout << "RAM     : " << std::fixed << std::setprecision(1) << mem << "%\n";
                std::cout << "Battery : "
                    << (bat.hasBattery ? std::to_string(bat.percent) + "%" : "N/A")
                    << (bat.isCharging ? " (Charging)" : " (Not Charging)") << "\n";
                std::cout << "==================================\n\n";

            }
            else if (key == 'Q') {
                std::cout << "\nExiting... Data saved to " << csvFile << "\n";
                running = false;
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(intervalSeconds));
    }

    return 0;
}