cmake_minimum_required(VERSION 3.16)
project(stb C)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${STB_SOURCE_DIR})
