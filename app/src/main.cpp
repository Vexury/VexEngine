#include "app.h"

#include <vex/core/engine.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static AppConfig parseArgs(int argc, char* argv[])
{
    AppConfig config;

    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i)
    {
        if (args[i] == "--headless")
            config.engine.headless = true;
        else if (args[i] == "--width" && i + 1 < args.size())
            config.engine.windowWidth = static_cast<uint32_t>(std::stoi(args[++i]));
        else if (args[i] == "--height" && i + 1 < args.size())
            config.engine.windowHeight = static_cast<uint32_t>(std::stoi(args[++i]));
        else if (args[i] == "--bench" && i + 1 < args.size())
            config.benchConfigPath = args[++i];
        else if (args[i] == "--bench-out" && i + 1 < args.size())
            config.benchOutDir = args[++i];
        else if (args[i] == "--help")
        {
            std::cout << "Usage: vex_app [options]\n"
                      << "  --headless          Run without a window\n"
                      << "  --width <W>         Window width (default 1280)\n"
                      << "  --height <H>        Window height (default 720)\n"
                      << "  --bench <file>      Run a benchmark from a JSON config and exit\n"
                      << "  --bench-out <dir>   Override the benchmark output directory\n";
            std::exit(0);
        }
    }

    return config;
}

int main(int argc, char* argv[])
{
    App app;

    // init can fail after the window and device already exist, so it still has
    // to be torn down before returning.
    if (!app.init(parseArgs(argc, argv)))
    {
        app.shutdown();
        return app.exitCode() != 0 ? app.exitCode() : EXIT_FAILURE;
    }

    app.run();
    app.shutdown();

    return app.exitCode();
}
