cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# OpenCV
FIND_PACKAGE( OpenCV REQUIRED )                              

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

# Specify the paths where Lua is installed (adjust if necessary)
set(LUA_INCLUDE_DIR "C:/msys64/mingw64/include")
set(LUA_LIBRARIES "C:/msys64/mingw64/lib")

# Include Lua headers
include_directories(${LUA_INCLUDE_DIR})

PROJECT(inkpath)

file(GLOB SOURCES src/cv/*.cpp src/ipcv_obj/*.cpp)
ADD_LIBRARY(ipcvobj SHARED ${SOURCES})

# Link Lua library
target_link_libraries(ipcvobj ${LUA_LIBRARIES}/liblua.a)
target_link_libraries(ipcvobj ${OpenCV_LIBRARIES})
