IF (WIN32)
	# Need to specify specific generator b/c building on MSYS2 MINGW64
	set (CMAKE_GENERATOR "MinGW Makefiles" CACHE INTERNAL "" FORCE)
    message("generator is set to ${CMAKE_GENERATOR}")
ENDIF()