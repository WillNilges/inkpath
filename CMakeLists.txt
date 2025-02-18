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
option(USE_EXTERNAL_OPENCV_STATIC "Link OpenCV statically (from subdirectory)" OFF)
IF(USE_EXTERNAL_OPENCV_STATIC)
	add_subdirectory(external/opencv_static)
	IF(EXISTS "${OpenCV_DIR}")
		# Require that the package is only being detected using the provided path from external project
		FIND_PACKAGE(OpenCV 4.5.1 REQUIRED COMPONENTS core imgproc imgcodecs calib3d NO_DEFAULT_PATH)
	ENDIF()
ELSE()
	FIND_PACKAGE(OpenCV 4.5.1 REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
ENDIF()
IF(OpenCV_FOUND)
	message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
	message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
ENDIF()

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
	target_include_directories(ipcv PRIVATE ${OpenCV_INCLUDE_DIRS})

	# Compile plugin component of Inkpath.
	ADD_LIBRARY(inkpath SHARED ${PLUGIN_SOURCES})
	target_compile_options(inkpath PRIVATE)
	target_link_libraries(inkpath ipcv)

	# Add dependencies if required by given options
	IF(USE_EXTERNAL_OPENCV_STATIC)
		add_dependencies(ipcv ExternalOpenCVStatic)
	ENDIF()
ENDIF()

# OpenCV Gets linked the same way for both platforms
target_link_libraries(inkpath ${OpenCV_LIBRARIES})
target_include_directories(inkpath PRIVATE ${OpenCV_INCLUDE_DIRS})


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
target_include_directories(inkpath-debug PRIVATE ${OpenCV_INCLUDE_DIRS})
set_target_properties(inkpath-debug PROPERTIES OUTPUT_NAME "inkpath-debug")
set_target_properties(inkpath-debug PROPERTIES RUNTIME_OUTPUT_DIRECTORY "debug")

# Add dependencies if required by given options
IF(USE_EXTERNAL_OPENCV_STATIC)
	add_dependencies(${PROJECT_NAME} ExternalOpenCVStatic)
	add_dependencies(${PROJECT_NAME}-debug ExternalOpenCVStatic)
ENDIF()

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
