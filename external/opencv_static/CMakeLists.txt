cmake_minimum_required(VERSION 3.22.1)
project(ExternalOpenCVStatic)

include(ExternalProject)

# Add option to disable since this crashes on some CPUs because of SSE3 problems
option(DISABLE_WEBP_EXTERNAL_OPENCV "Disable OpenCV library 3rd party dependency for WEBP (from subdirectory)" OFF)

# Set where to build and install OpenCV
set(OPENCV_BUILD_DIR "${CMAKE_BINARY_DIR}/${PROJECT_NAME}")
set(OPENCV_INSTALL_DIR "${CMAKE_BINARY_DIR}/${PROJECT_NAME}_install")
# Set the URL to clone OpenCV from GitHub
set(OPENCV_TAG "4.11.0")
set(OPENCV_URL "https://github.com/opencv/opencv.git")

# Expose OpenCV_DIR so the main project can find the installed OpenCV
set(OpenCV_DIR "${OPENCV_INSTALL_DIR}/lib/cmake/opencv4" PARENT_SCOPE)

# Clone and build OpenCV
ExternalProject_Add(
    ${PROJECT_NAME}
    PREFIX ${OPENCV_BUILD_DIR}
    GIT_REPOSITORY ${OPENCV_URL}
    GIT_TAG ${OPENCV_TAG}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_BUILD_TYPE=Release
        -DOPENCV_GENERATE_PKGCONFIG=ON
        # Require image support
        -DWITH_AVIF=ON
        -DWITH_JPEG=ON
        -DWITH_PNG=ON
        -DWITH_TIFF=ON
        -DWITH_WEBP=$<IF:$<BOOL:${DISABLE_WEBP_EXTERNAL_OPENCV}>,OFF,ON>
        # Disable tests, docs, ...
        -DBUILD_TESTS=OFF
        -DBUILD_DOCS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_PERF_TESTS=OFF
        # Disable unnecessary functionality
        -DWITH_FFMPEG=OFF
        -DWITH_GSTREAMER=OFF
        -DWITH_GTK=OFF
        -DWITH_OPENEXR=OFF
        -DWITH_QT=OFF
        -DBUILD_opencv_java=OFF
        -DBUILD_opencv_python3=OFF
        # Disable unnecessary modules
        -DBUILD_opencv_highgui=OFF
        -DBUILD_opencv_ml=OFF
        -DBUILD_opencv_objdetect=OFF
        -DBUILD_opencv_stitching=OFF
        -DBUILD_opencv_video=OFF
        -DBUILD_opencv_videoio=OFF
    # Declare expected output files that will be generated after the external project is built
    BUILD_BYPRODUCTS "${OpenCV_DIR}/OpenCVConfig.cmake"
)

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR} -B ${CMAKE_BINARY_DIR}
    COMMENT "Re-running CMake to configure ${PROJECT_NAME}"
)
