cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
PROJECT(inkpath)

# Send artifacts to /build/ImageTranscription. That will be the final artifact.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)

# Things work a little differently on Windows vs Linux.
IF(WIN32)
	message("Building for Windows")
	set(INSTALL_DESTINATION "C:/Program Files/Xournal++/share/xournalpp/plugins")
ELSE()
	# Need position-independent code flag enabled to make Lua work
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)
	set(INSTALL_DESTINATION /usr/share/xournalpp/plugins)
ENDIF()

# Define our sources
file(GLOB CV_SOURCES src/cv/*.cpp)
file(GLOB PLUGIN_SOURCES src/plugin/*.cpp)
file(GLOB DEBUG_SOURCES src/cv/debug/*.cpp)

# Locate dependent packages
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
                            
FIND_PACKAGE(Lua 5.4 REQUIRED)
message(STATUS "LUA_INCLUDE_DIR: ${LUA_INCLUDE_DIR}")
message(STATUS "LUA_LIBRARIES: ${LUA_LIBRARIES}")
INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})

# We link OpenCV statically in Linux,
# but link totally dynamically and redistribute dlls in Windows
IF(WIN32)
	ADD_LIBRARY(inkpath SHARED ${CV_SOURCES} ${PLUGIN_SOURCES})
	target_link_libraries(inkpath ${LUA_LIBRARIES})
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


# Copy the script and manifest into the build artifact
file(GLOB PLUGIN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/plugin/*")
FOREACH(FILE ${PLUGIN_FILES})
	file(COPY ${FILE} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
ENDFOREACH()

# Finally, set an install target.
install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} DESTINATION ${INSTALL_DESTINATION})

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
