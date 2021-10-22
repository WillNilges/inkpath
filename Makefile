CC=gcc

# Warnings
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wformat=2 -Winit-self \
	-Wmissing-include-dirs -Wformat-nonliteral -Wnested-externs \
	-Wno-unused-parameter -Wold-style-definition -Wredundant-decls -Wshadow \
	-Wstrict-prototypes -Wwrite-strings

LIGHT_WARNINGS = -Wall

# CFLAGS += -std=gnu99

build/inkpath: src/main.c src/util.c src/util.h
	mkdir -p build
	$(CC) $(LIGHT_WARNINGS) $(CFLAGS) src/main.c src/util.c -g `pkg-config --libs autotrace glib-2.0` `pkg-config --cflags autotrace glib-2.0` -o build/inkpath

lua-module: build/inkpath
	$(CC) $(LIGHT_WARNINGS) $(CFLAGS) src/lua_util.c -g `pkg-config --libs lua autotrace glib-2.0` `pkg-config --cflags lua autotrace glib-2.0` -fPIC -shared -o build/inkpath.so

.PHONY: clean
clean:
	rm -rf build
