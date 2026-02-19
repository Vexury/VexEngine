#include <vex/core/log.h>
#include <iostream>

namespace vex
{
namespace Log
{

static std::vector<LogEntry> s_entries;

void info(std::string_view msg)
{
    std::cout << "[VEX INFO] " << msg << std::endl;
    s_entries.push_back({ Level::Info, std::string(msg) });
}

void warn(std::string_view msg)
{
    std::cerr << "[VEX WARN] " << msg << std::endl;
    s_entries.push_back({ Level::Warn, std::string(msg) });
}

void error(std::string_view msg)
{
    std::cerr << "[VEX ERROR] " << msg << std::endl;
    s_entries.push_back({ Level::Error, std::string(msg) });
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
