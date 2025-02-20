cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
PROJECT(inkpath)

set(PLUGIN_NAME "ImageTranscription")
set(PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/plugin")
set(CMAKE_INSTALL_PREFIX_PLUGIN "${CMAKE_INSTALL_PREFIX}")

# Options
set(CMAKE_INSTALL_PREFIX_ICONS "" CACHE PATH "Installation path for icons")

# Send artifacts to /build/ImageTranscription. That will be the final artifact.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)

# Things work a little differently on Windows vs Linux.
if(UNIX AND NOT APPLE)
    # For dynamic libraries enabling a -fPIC (Position-Independent Code) flag is
    # required when compiling code into shared libraries
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Define our sources
file(GLOB CV_SOURCES src/cv/*.cpp)
file(GLOB PLUGIN_SOURCES src/plugin/*.cpp)
file(GLOB DEBUG_SOURCES src/cv/debug/*.cpp)

# Locate dependent packages
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(Lua 5.4 REQUIRED)
INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})

# We link OpenCV statically in Linux,
# but link totally dynamically and redistribute dlls in Windows
IF(WIN32)
	# FIXME (wdn): Fix FIND_PACKAGE behavior for Lua on Windows
	set(LUA_INCLUDE_DIR "C:/msys64/mingw64/include")
	set(LUA_LIBRARIES "C:/msys64/mingw64/lib")

	ADD_LIBRARY(inkpath SHARED ${CV_SOURCES} ${PLUGIN_SOURCES})
	target_link_libraries(inkpath ${LUA_LIBRARIES}/liblua.a)
ELSE()
	# Compile the CV component of Inkpath separately. This is mostly so that we
	# can build our debug program and such.
	ADD_LIBRARY(ipcv STATIC ${CV_SOURCES})
	target_link_libraries(ipcv ${OpenCV_LIBRARIES})

	# Compile plugin component of Inkpath.
	ADD_LIBRARY(inkpath SHARED ${PLUGIN_SOURCES})
	target_compile_options(inkpath PRIVATE)
	target_link_libraries(inkpath ipcv)
ENDIF()

# OpenCV Gets linked the same way for both platforms
target_link_libraries(inkpath ${OpenCV_LIBRARIES})


# Finally, set an install target.
install(
    DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    DESTINATION ${CMAKE_INSTALL_PREFIX_PLUGIN}
)
# Copy plugin directory files to the install prefix
install(
    DIRECTORY ${PLUGIN_DIR}/
    DESTINATION ${CMAKE_INSTALL_PREFIX_PLUGIN}/${PLUGIN_NAME}
    FILES_MATCHING
    PATTERN "*"
)
# Copy icon files to the install prefix for icons
if(CMAKE_INSTALL_PREFIX_ICONS)
    install(
        DIRECTORY ${PLUGIN_DIR}/
        DESTINATION ${CMAKE_INSTALL_PREFIX_ICONS}
        FILES_MATCHING
        PATTERN "*.svg"
    )
endif()

# Also set up debugging target
add_executable(inkpath-debug EXCLUDE_FROM_ALL ${CV_SOURCES} ${DEBUG_SOURCES})
target_compile_definitions(inkpath-debug PRIVATE INKPATH_DEBUG)

target_link_libraries(inkpath-debug ${OpenCV_LIBRARIES})
set_target_properties(inkpath-debug PROPERTIES OUTPUT_NAME "inkpath-debug")
set_target_properties(inkpath-debug PROPERTIES RUNTIME_OUTPUT_DIRECTORY "debug")

# If cross compiling on Windows copy dependent DLLs
if(CMAKE_GENERATOR STREQUAL "MinGW Makefiles")
    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND
            ldd "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/lib${PROJECT_NAME}.dll" | grep
            mingw64 | awk "{ print \$3 }" | xargs -I {} cp {}
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Copying ${PROJECT_NAME} dependent MinGW DLLs"
        VERBATIM
    )
endif()
