cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# OpenCV
FIND_PACKAGE( OpenCV REQUIRED )                              

# For some reason Lua FIND_PACKAGE isn't working.
# FIND_PACKAGE( Lua 5.3 REQUIRED )                              

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

IF (WIN32)
		# Specify the paths where Lua is installed (adjust if necessary)
		set(LUA_INCLUDE_DIR "C:/msys64/mingw64/include")
		set(LUA_LIBRARIES "C:/msys64/mingw64/lib")
ELSE()
		set(LUA_INCLUDE_DIR "/usr/include/lua5.3")
		set(LUA_LIBRARIES "/usr/lib/lua5.3")
ENDIF()

# Include Lua headers
include_directories(${LUA_INCLUDE_DIR})

PROJECT(inkpath)

file(GLOB SOURCES src/cv/*.cpp src/ipcv_obj/*.cpp)
IF (WIN32)
		ADD_LIBRARY(ipcvobj SHARED ${SOURCES})
ELSE()
		ADD_LIBRARY(ipcvobj STATIC ${SOURCES})
ENDIF()

# Link Lua library
target_link_libraries(ipcvobj ${LUA_LIBRARIES}/liblua.a)
target_link_libraries(ipcvobj ${OpenCV_LIBRARIES})

# Copy the script and manifest into the build artifact
file(GLOB PLUGIN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/plugin/*")
foreach(FILE ${PLUGIN_FILES})
    file(COPY ${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
