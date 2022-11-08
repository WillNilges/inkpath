CC=gcc

# Warnings
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wformat=2 -Winit-self \
	-Wmissing-include-dirs -Wformat-nonliteral -Wnested-externs \
	-Wno-unused-parameter -Wold-style-definition -Wredundant-decls -Wshadow \
	-Wstrict-prototypes -Wwrite-strings
LIGHT_WARNINGS = -Wall

.PHONY: clean install uninstall dev-install dev-uninstall

build: ipcv-cuda-debug

*.o: *.cpp
		$(CC) $(CFLAGS) -c $?

# For generating a CV debugging binary
ipcv-cuda-debug: *.o 
	mkdir -p build
	g++ $< `pkg-config --cflags --libs opencv4` -o $@

clean:
	rm -rf build
