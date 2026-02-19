#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace vex
{
namespace Log
{

enum class Level { Info, Warn, Error };

struct LogEntry
{
    Level level;
    std::string message;
};

void info(std::string_view msg);
void warn(std::string_view msg);
void error(std::string_view msg);

const std::vector<LogEntry>& getEntries();
void clear();

} // namespace Log
} // namespace vex
