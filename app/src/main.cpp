#include "app.h"

#include <vex/core/engine.h>

#include <iostream>
#include <string>
#include <vector>

static vex::EngineConfig parseArgs(int argc, char* argv[])
{
    vex::EngineConfig config;

    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i)
    {
        if (args[i] == "--headless")
            config.headless = true;
        else if (args[i] == "--width" && i + 1 < args.size())
            config.windowWidth = static_cast<uint32_t>(std::stoi(args[++i]));
        else if (args[i] == "--height" && i + 1 < args.size())
            config.windowHeight = static_cast<uint32_t>(std::stoi(args[++i]));
        else if (args[i] == "--help")
        {
            std::cout << "Usage: vex_app [options]\n"
                      << "  --headless       Run without a window\n"
                      << "  --width <W>      Window width (default 1280)\n"
                      << "  --height <H>     Window height (default 720)\n";
            std::exit(0);
        }
    }

    return config;
}

int main(int argc, char* argv[])
{
    App app;

    if (!app.init(parseArgs(argc, argv)))
        return EXIT_FAILURE;

    app.run();
    app.shutdown();

    return EXIT_SUCCESS;
}
