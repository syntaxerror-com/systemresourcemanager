// Predictive System Resource Monitor
// - Collects system + per-process telemetry using Win32 / Windows system APIs
//   (PDH, ToolHelp32, PSAPI, GetProcessTimes, GetSystemPowerStatus, Get/SetPriorityClass)
// - Logs it to CSV for the Python anomaly detector (z-score + KNN)
// - Reads scheduler_actions.txt written by Python and applies automatic task scheduling
//   (lower the priority of CRITICAL processes, restore it when they calm down)
//
// All files are read/written in the current working directory (run the exe and
// `python anomaly_detector.py` from the same folder, or set SRM_DATA_DIR for Python).

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#include <tlhelp32.h>
#include <conio.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "psapi.lib")

// ------------------------------------------------------------------ utils --
std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    struct tm timeInfo;
    localtime_s(&timeInfo, &t);
    std::ostringstream oss;
    oss << std::put_time(&timeInfo, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

static ULONGLONG fileTimeToU64(const FILETIME& ft) {
    ULARGE_INTEGER u;
    u.LowPart = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;
    return u.QuadPart;
}

static bool fileExists(const std::string& name) {
    std::ifstream f(name);
    return f.good();
}

// ---------------------------------------------------------- system metrics --
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
    MEMORYSTATUSEX m;
    m.dwLength = sizeof(m);
    GlobalMemoryStatusEx(&m);
    return (double)(m.ullTotalPhys - m.ullAvailPhys) / (double)m.ullTotalPhys * 100.0;
}

struct BatteryInfo {
    int percent;
    bool isCharging;
    bool hasBattery;
    double drainPerMin;  // % per minute while on battery; 0 when charging / unknown
};

// Battery drain rate: % lost per minute since we last started discharging.
BatteryInfo getBatteryInfo() {
    static int startPct = -1;
    static std::chrono::steady_clock::time_point startTime;

    SYSTEM_POWER_STATUS sps;
    BatteryInfo info = {-1, false, false, 0.0};
    if (!GetSystemPowerStatus(&sps)) return info;

    info.hasBattery = (sps.BatteryFlag != 128);
    info.percent = (int)sps.BatteryLifePercent;
    if (info.percent > 100) info.percent = -1;
    info.isCharging = (sps.ACLineStatus == 1);

    if (!info.hasBattery || info.percent < 0 || info.isCharging) {
        startPct = -1;  // reset window whenever we are on AC
        return info;
    }
    auto now = std::chrono::steady_clock::now();
    if (startPct < 0 || info.percent > startPct) {
        startPct = info.percent;
        startTime = now;
    }
    double minutes = std::chrono::duration<double>(now - startTime).count() / 60.0;
    if (minutes >= 1.0) info.drainPerMin = (startPct - info.percent) / minutes;
    if (info.drainPerMin < 0) info.drainPerMin = 0;
    return info;
}

void logToCSV(const std::string& filename, double cpu, double memory, const BatteryInfo& b) {
    bool exists = fileExists(filename);
    std::ofstream file(filename, std::ios::app);
    if (!exists)
        file << "Timestamp,CPU_Load(%),Memory_Usage(%),Battery(%),Charging,Battery_Drain(%/min)\n";
    file << getTimestamp() << "," << std::fixed << std::setprecision(2) << cpu << "," << memory << ","
         << (b.hasBattery ? std::to_string(b.percent) : "N/A") << "," << (b.isCharging ? "Yes" : "No")
         << "," << b.drainPerMin << "\n";
}

// ----------------------------------------------------------- process metrics --
struct ProcessInfo {
    std::string name;
    DWORD pid = 0;
    double cpuPercent = 0.0;
    SIZE_T memoryMB = 0;
};

// Per-process CPU% from GetProcessTimes deltas between two snapshots.
static std::map<DWORD, ULONGLONG> g_lastProcTime;
static ULONGLONG g_lastWall = 0;

std::vector<ProcessInfo> getRunningProcesses() {
    std::vector<ProcessInfo> processes;

    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const double numCPU = si.dwNumberOfProcessors ? si.dwNumberOfProcessors : 1;

    FILETIME nowFt;
    GetSystemTimeAsFileTime(&nowFt);
    ULONGLONG wall = fileTimeToU64(nowFt);
    double wallDelta = g_lastWall ? (double)(wall - g_lastWall) : 0.0;

    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return processes;

    PROCESSENTRY32W pe;
    pe.dwSize = sizeof(pe);
    if (!Process32FirstW(snapshot, &pe)) {
        CloseHandle(snapshot);
        return processes;
    }

    std::map<DWORD, ULONGLONG> currentTimes;
    do {
        ProcessInfo info;
        char nameBuf[260];
        wcstombs_s(nullptr, nameBuf, pe.szExeFile, 259);
        info.name = nameBuf;
        info.pid = pe.th32ProcessID;

        HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, FALSE, pe.th32ProcessID);
        if (h) {
            PROCESS_MEMORY_COUNTERS pmc;
            if (GetProcessMemoryInfo(h, &pmc, sizeof(pmc))) info.memoryMB = pmc.WorkingSetSize / (1024 * 1024);

            FILETIME c, e, k, u;
            if (GetProcessTimes(h, &c, &e, &k, &u)) {
                ULONGLONG total = fileTimeToU64(k) + fileTimeToU64(u);
                currentTimes[info.pid] = total;
                auto it = g_lastProcTime.find(info.pid);
                if (it != g_lastProcTime.end() && wallDelta > 0 && total >= it->second)
                    info.cpuPercent = (double)(total - it->second) / wallDelta / numCPU * 100.0;
            }
            CloseHandle(h);
        }
        processes.push_back(info);
    } while (Process32NextW(snapshot, &pe));
    CloseHandle(snapshot);

    g_lastProcTime.swap(currentTimes);
    g_lastWall = wall;

    std::sort(processes.begin(), processes.end(),
              [](const ProcessInfo& a, const ProcessInfo& b) { return a.memoryMB > b.memoryMB; });
    return processes;
}

void printProcesses(const std::vector<ProcessInfo>& processes, int topN = 15) {
    std::cout << "\n----------- TOP " << topN << " PROCESSES BY MEMORY -----------\n";
    std::cout << std::left << std::setw(8) << "PID" << std::setw(32) << "Process Name"
              << std::setw(14) << "Memory (MB)" << "CPU (%)\n";
    std::cout << std::string(64, '-') << "\n";
    int count = 0;
    for (const auto& p : processes) {
        if (count++ >= topN) break;
        std::cout << std::left << std::setw(8) << p.pid << std::setw(32) << p.name << std::setw(14)
                  << p.memoryMB << std::fixed << std::setprecision(1) << p.cpuPercent << "\n";
    }
    std::cout << "\nTotal processes running: " << processes.size() << "\n" << std::string(64, '-') << "\n\n";
}

// Logs top 30 by memory plus any process using >5% CPU. Each row carries the
// system-wide metrics so the Python KNN model sees them as features.
void logProcessesToCSV(const std::string& filename, const std::vector<ProcessInfo>& processes,
                       double sysCpu, double sysMem, double drain) {
    bool exists = fileExists(filename);
    std::ofstream file(filename, std::ios::app);
    if (!exists) file << "Timestamp,PID,ProcessName,Memory_MB,CPU_Percent,Sys_CPU,Sys_Mem,Battery_Drain\n";

    std::string ts = getTimestamp();
    int count = 0;
    for (const auto& p : processes) {
        if (count >= 30 && p.cpuPercent <= 5.0) continue;
        file << ts << "," << p.pid << "," << p.name << "," << p.memoryMB << "," << std::fixed
             << std::setprecision(2) << p.cpuPercent << "," << sysCpu << "," << sysMem << "," << drain << "\n";
        count++;
    }
}

// ---------------------------------------------------- alerts from Python ----
void checkPythonAlerts(const std::string& alertFile) {
    std::ifstream file(alertFile);
    if (!file.is_open()) return;

    std::string line;
    std::vector<std::string> criticals, warnings;
    while (std::getline(file, line)) {
        if (line.find("CRITICAL:") == 0) criticals.push_back(line.substr(9));
        else if (line.find("WARNING:") == 0) warnings.push_back(line.substr(8));
    }
    if (criticals.empty() && warnings.empty()) return;

    auto show = [](const char* tag, const std::string& a) {
        // format: proc:mem:avg:tier:z:knn
        std::istringstream ss(a);
        std::string proc, mem, avg, tier, z, knn;
        std::getline(ss, proc, ':'); std::getline(ss, mem, ':'); std::getline(ss, avg, ':');
        std::getline(ss, tier, ':'); std::getline(ss, z, ':');   std::getline(ss, knn, ':');
        std::cout << "  [" << tag << "] " << proc << " | Using: " << mem << " MB | Normal: " << avg
                  << " MB | Tier: " << tier << " | z=" << z << " | knn=" << knn << "\n";
    };
    std::cout << "\n!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=\n";
    for (const auto& a : criticals) show("CRITICAL", a);
    for (const auto& a : warnings) show("WARNING ", a);
    std::cout << "!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=\n\n";
}

// ------------------------------------------- automatic task scheduling -------
// Python writes lines like  THROTTLE:<pid>:<name>  /  RESTORE:<pid>:<name>.
// THROTTLE lowers the process to BELOW_NORMAL priority so the CPU scheduler gives
// foreground work more time; RESTORE puts back the original priority class.
static std::map<DWORD, DWORD> g_originalPriority;

static bool isSafeToTouch(DWORD pid, const std::string& name) {
    static const char* protectedNames[] = {"system", "csrss.exe", "wininit.exe", "services.exe",
                                           "lsass.exe", "winlogon.exe", "svchost.exe", "dwm.exe",
                                           "explorer.exe", "systemresourcemonitor.exe"};
    if (pid == 0 || pid == 4 || pid == GetCurrentProcessId()) return false;
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (auto* p : protectedNames) if (lower == p) return false;
    return true;
}

void logSchedulerEvent(const std::string& action, DWORD pid, const std::string& name, bool ok) {
    bool exists = fileExists("scheduler_log.csv");
    std::ofstream f("scheduler_log.csv", std::ios::app);
    if (!exists) f << "Timestamp,Action,PID,ProcessName,Success\n";
    f << getTimestamp() << "," << action << "," << pid << "," << name << "," << (ok ? "Yes" : "No") << "\n";
}

void applySchedulerActions(const std::string& actionFile) {
    std::ifstream in(actionFile);
    if (!in.is_open()) return;
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
    in.close();
    std::remove(actionFile.c_str());  // consume once

    for (const auto& l : lines) {
        std::istringstream ss(l);
        std::string action, pidStr, name;
        std::getline(ss, action, ':'); std::getline(ss, pidStr, ':'); std::getline(ss, name);
        if (pidStr.empty()) continue;
        DWORD pid = (DWORD)std::stoul(pidStr);
        if (!isSafeToTouch(pid, name)) continue;

        HANDLE h = OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
        bool ok = false;
        if (h) {
            if (action == "THROTTLE") {
                DWORD cur = GetPriorityClass(h);
                if (cur && g_originalPriority.find(pid) == g_originalPriority.end()) g_originalPriority[pid] = cur;
                ok = SetPriorityClass(h, BELOW_NORMAL_PRIORITY_CLASS) != 0;
                if (ok) std::cout << "  [SCHEDULER] Lowered priority of " << name << " (PID " << pid << ")\n";
            } else if (action == "RESTORE") {
                auto it = g_originalPriority.find(pid);
                DWORD prio = (it != g_originalPriority.end()) ? it->second : NORMAL_PRIORITY_CLASS;
                ok = SetPriorityClass(h, prio) != 0;
                g_originalPriority.erase(pid);
                if (ok) std::cout << "  [SCHEDULER] Restored priority of " << name << " (PID " << pid << ")\n";
            }
            CloseHandle(h);
        }
        logSchedulerEvent(action, pid, name, ok);
    }
}

// ------------------------------------------------------------------ main -----
int main() {
    const std::string csvFile = "system_log.csv";
    const std::string alertFile = "alerts.txt";
    const std::string actionFile = "scheduler_actions.txt";
    const int intervalSeconds = 3;
    bool autoSchedule = true;

    std::cout << "========================================\n";
    std::cout << "   Predictive System Resource Monitor   \n";
    std::cout << "========================================\n";
    std::cout << "\n  [P] Show Processes\n  [S] Show System Stats Now\n"
                 "  [T] Toggle Auto Task Scheduling (currently ON)\n  [Q] Quit\n";
    std::cout << "\nMonitoring started... (run: python anomaly_detector.py in this folder)\n\n";

    getRunningProcesses();  // prime per-process CPU baseline
    bool running = true;
    while (running) {
        double cpu = getCPULoad();  // blocks ~1s on first call
        double mem = getMemoryUsagePercent();
        BatteryInfo bat = getBatteryInfo();
        logToCSV(csvFile, cpu, mem, bat);

        auto processes = getRunningProcesses();
        logProcessesToCSV("process_log.csv", processes, cpu, mem, bat.drainPerMin);

        checkPythonAlerts(alertFile);
        if (autoSchedule) applySchedulerActions(actionFile);
        else std::remove(actionFile.c_str());

        if (_kbhit()) {
            char key = (char)toupper(_getch());
            if (key == 'P') {
                printProcesses(processes, 15);
            } else if (key == 'S') {
                std::cout << "\n========== SYSTEM STATS ==========\n"
                          << "Time    : " << getTimestamp() << "\n"
                          << "CPU     : " << std::fixed << std::setprecision(1) << cpu << "%\n"
                          << "RAM     : " << mem << "%\n"
                          << "Battery : " << (bat.hasBattery ? std::to_string(bat.percent) + "%" : "N/A")
                          << (bat.isCharging ? " (Charging)" : " (Not Charging)")
                          << " | drain " << std::setprecision(2) << bat.drainPerMin << " %/min\n"
                          << "==================================\n\n";
            } else if (key == 'T') {
                autoSchedule = !autoSchedule;
                std::cout << "\nAuto task scheduling: " << (autoSchedule ? "ON" : "OFF") << "\n\n";
            } else if (key == 'Q') {
                std::cout << "\nExiting... Data saved to " << csvFile << "\n";
                running = false;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(intervalSeconds));
    }
    return 0;
}
