cmake_minimum_required(VERSION 3.7)
project(atari_uct)

set(CMAKE_CXX_STANDARD 11)

set(ALE_PATH /home/mist/Projects/Dissertation/xitari)
set(CMAKE_CXX_FLAGS  -I${ALE_PATH})
set(CMAKE_EXE_LINKER_FLAGS "-lxitari -lgflags -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_imgcodecs")

set(SOURCE_FILES_UCT_PLANNER
    uct.cpp
    uct.hpp)

set(SOURCE_ATARI_PLANNER
        atari.hpp
        atari_images.hpp)

add_executable(uct_planner uctPlanner.cpp ${SOURCE_FILES_UCT_PLANNER})
add_executable(atari_uct atariUCTPlanner.cpp ${SOURCE_FILES_UCT_PLANNER} ${SOURCE_ATARI_PLANNER})
