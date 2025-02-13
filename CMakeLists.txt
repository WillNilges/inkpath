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
IF(NOT WIN32 AND USE_EXTERNAL_OPENCV_STATIC)
	add_subdirectory(external/opencv_static)
	IF(EXISTS "${OpenCV_DIR}")
		# Require that the package is only being detected using the provided path from external project
		FIND_PACKAGE(OpenCV 4.5.1 REQUIRED COMPONENTS core imgproc imgcodecs calib3d NO_DEFAULT_PATH)
	ENDIF()
ELSE()
	FIND_PACKAGE(OpenCV 4.5.1 REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
ENDIF()
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
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

	# Add project dependency if it should be linked against the static OpenCV libraries
	IF(USE_EXTERNAL_OPENCV_STATIC)
		add_dependencies(ipcv ExternalOpenCVStatic)
		add_dependencies(${PROJECT_NAME} ExternalOpenCVStatic)
	ENDIF()
ENDIF()

# OpenCV Gets linked the same way for both platforms
target_link_libraries(inkpath ${OpenCV_LIBRARIES})


# Copy the script and manifest into the build artifact
file(GLOB PLUGIN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/plugin/*")
FOREACH(FILE ${PLUGIN_FILES})
	file(COPY ${FILE} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
ENDFOREACH()

IF(WIN32)
	install(CODE "execute_process(COMMAND bash \"${CMAKE_SOURCE_DIR}/scripts/copy_dlls.sh\")")
ENDIF()

# Finally, set an install target.
install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} DESTINATION ${INSTALL_DESTINATION})

# Also set up debugging target
add_executable(inkpath-debug EXCLUDE_FROM_ALL ${CV_SOURCES} ${DEBUG_SOURCES})
target_compile_definitions(inkpath-debug PRIVATE INKPATH_DEBUG)

target_link_libraries(inkpath-debug ${OpenCV_LIBRARIES})
set_target_properties(inkpath-debug PROPERTIES OUTPUT_NAME "inkpath-debug")
set_target_properties(inkpath-debug PROPERTIES RUNTIME_OUTPUT_DIRECTORY "debug")
