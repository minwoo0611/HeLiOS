cmake_minimum_required(VERSION 3.10)

project(PointCloudCombiner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Find PCL
find_package(PCL 1.8.1 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Find Threads
find_package(Threads REQUIRED)

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# Add executable
add_executable(overlap 
    src/pointcloud_listener.cpp
)

# Link libraries
target_link_libraries(overlap 
    ${PCL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    yaml-cpp
    stdc++fs
)
