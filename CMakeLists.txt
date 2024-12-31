cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Send artifacts to /build/ImageTranscription. That will be the final artifact.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

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
		set(LUA_LIBRARIES "/usr/lib/x86_64-linux-gnu")
ENDIF()

# Include Lua headers
include_directories(${LUA_INCLUDE_DIR})

PROJECT(inkpath)

file(GLOB SOURCES src/cv/*.cpp src/ipcv_obj/*.cpp)
IF (WIN32)
		ADD_LIBRARY(inkpath SHARED ${SOURCES})
		target_link_libraries(inkpath ${LUA_LIBRARIES}/liblua.a)
ELSE()
		ADD_LIBRARY(inkpath STATIC ${SOURCES})
		target_link_libraries(inkpath ${LUA_LIBRARIES}/liblua5.3.a)
ENDIF()

# Link OpenCV
target_link_libraries(inkpath ${OpenCV_LIBRARIES})

# Copy the script and manifest into the build artifact
file(GLOB PLUGIN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/plugin/*")
foreach(FILE ${PLUGIN_FILES})
		file(COPY ${FILE} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endforeach()

# Finally, an install target.
IF (WIN32)
		set(INSTALL_DESTINATION "C:/Program Files/Xournal++/share/xournalpp/plugins")
ELSE()
		set(INSTALL_DESTINATION /usr/share/xournalpp/plugins)
ENDIF()

install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} DESTINATION ${INSTALL_DESTINATION})

