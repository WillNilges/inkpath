cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
PROJECT(inkpath)

# Send artifacts to /build/ImageTranscription. That will be the final artifact.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ImageTranscription)

# Things work a little differently on Windows vs Linux.
if (WIN32)
	# Need to specify specific generator b/c building on MSYS2 MINGW64
	set (CMAKE_GENERATOR "MinGW Makefiles" CACHE INTERNAL "" FORCE)
	set(INSTALL_DESTINATION "C:/Program Files/Xournal++/share/xournalpp/plugins")
ELSE()
	# Need position-independent code flag enabled to make Lua work
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)
	set(INSTALL_DESTINATION /usr/share/xournalpp/plugins)
ENDIF()

# Define our sources
file(GLOB CV_SOURCES src/cv/*.cpp)
file(GLOB PLUGIN_SOURCES src/ipcv_obj/*.cpp)

# Locate dependent packages
FIND_PACKAGE(OpenCV REQUIRED)                              
FIND_PACKAGE(Lua 5.3 REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})

#IF (WIN32)
#		# Specify the paths where Lua is installed (adjust if necessary)
#		set(LUA_INCLUDE_DIR "C:/msys64/mingw64/include")
#		set(LUA_LIBRARIES "C:/msys64/mingw64/lib")
#ENDIF()
	#set(LUA_INCLUDE_DIR "/usr/include/lua5.3")
	#set(LUA_LIBRARIES "/usr/lib/x86_64-linux-gnu")

# Compile the CV component of Inkpath separately. This is mostly so that we
# can build our debug program and such.
ADD_LIBRARY(ipcv STATIC ${CV_SOURCES})
target_link_libraries(ipcv ${OpenCV_LIBRARIES})

# Compile plugin component of Inkpath.
ADD_LIBRARY(inkpath SHARED ${PLUGIN_SOURCES})
target_compile_options(inkpath PRIVATE)
target_link_libraries(inkpath ipcv)
target_link_libraries(inkpath ${OpenCV_LIBRARIES})

# Copy the script and manifest into the build artifact
file(GLOB PLUGIN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/plugin/*")
foreach(FILE ${PLUGIN_FILES})
		file(COPY ${FILE} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endforeach()

# Finally, set an install target.
install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} DESTINATION ${INSTALL_DESTINATION})

