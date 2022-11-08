CC=gcc

# Warnings
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wformat=2 -Winit-self \
	-Wmissing-include-dirs -Wformat-nonliteral -Wnested-externs \
	-Wno-unused-parameter -Wold-style-definition -Wredundant-decls -Wshadow \
	-Wstrict-prototypes -Wwrite-strings
LIGHT_WARNINGS = -Wall
CV_PKG_CONFIG=`pkg-config --cflags --libs opencv4`

.PHONY: clean install uninstall dev-install dev-uninstall

build: ipcv-cuda-debug

*.o: *.cpp
		$(CC) $(CFLAGS) $(CV_PKG_CONFIG) -g -c $?

# For generating a CV debugging binary
ipcv-cuda-debug: *.o 
	g++ $< $(CV_PKG_CONFIG) -g -o $@

clean:
	rm -rf *.o
	rm -rf ipcv-cuda-debug
