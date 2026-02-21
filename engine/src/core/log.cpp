#include <vex/core/log.h>
#include <iostream>
#include <chrono>
#include <cstdio>

namespace vex
{
namespace Log
{

static std::vector<LogEntry> s_entries;
static const auto s_startTime = std::chrono::steady_clock::now();

static double elapsed()
{
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - s_startTime).count();
}

static void print(const char* tag, std::ostream& out, double ts, std::string_view msg)
{
    char tsBuf[16];
    std::snprintf(tsBuf, sizeof(tsBuf), "[%7.3fs]", ts);
    out << tsBuf << " " << tag << " " << msg << "\n";
}

void info(std::string_view msg)
{
    double ts = elapsed();
    print("[VEX INFO]",  std::cout, ts, msg);
    s_entries.push_back({ Level::Info, ts, std::string(msg) });
}

void warn(std::string_view msg)
{
    double ts = elapsed();
    print("[VEX WARN]",  std::cerr, ts, msg);
    s_entries.push_back({ Level::Warn, ts, std::string(msg) });
}

void error(std::string_view msg)
{
    double ts = elapsed();
    print("[VEX ERROR]", std::cerr, ts, msg);
    s_entries.push_back({ Level::Error, ts, std::string(msg) });
}

const std::vector<LogEntry>& getEntries()
{
    return s_entries;
}

void clear()
{
    s_entries.clear();
}

} // namespace Log
} // namespace vex
