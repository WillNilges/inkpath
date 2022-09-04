CC=gcc

# Warnings
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wformat=2 -Winit-self \
	-Wmissing-include-dirs -Wformat-nonliteral -Wnested-externs \
	-Wno-unused-parameter -Wold-style-definition -Wredundant-decls -Wshadow \
	-Wstrict-prototypes -Wwrite-strings
#LIGHT_WARNINGS = -Wall
PLUGIN_NAME=ImageTranscription
SO_INSTALL_PATH=/usr/local/lib/lua/5.3# Just one of many possible destinations :)

# CFLAGS += -std=gnu99

.PHONY: clean install uninstall dev-install dev-uninstall

# TODO: `-g` is for debugging. Make a target that supports debugging separately from primary compilation

at_source := $(wildcard src/autotrace/*.c src/autotrace/*.h)
build/inkpath: src/main.c src/util.c src/util.h $(at_source)
	mkdir -p build
	$(CC) $(LIGHT_WARNINGS) $(CFLAGS) $(at_source) src/main.c src/util.c -g `pkg-config --libs glib-2.0` `pkg-config --cflags glib-2.0` -o build/inkpath

lua-plugin: src/lua_util.c $(at_source)
	mkdir -p build
	$(CC) $(LIGHT_WARNINGS) $(CFLAGS) src/lua_util.c $(at_source) -g `pkg-config --libs lua5.3 glib-2.0` `pkg-config --cflags lua5.3 glib-2.0` -fPIC -shared -o $(PLUGIN_NAME)/inkpath.so

install: lua-plugin
	cp -r $(PLUGIN_NAME) /usr/share/xournalpp/plugins/
	mkdir -p $(SO_INSTALL_PATH)
	cp -r $(PLUGIN_NAME)/inkpath.so $(SO_INSTALL_PATH)/inkpath.so

uninstall:
	rm -rf /usr/share/xournalpp/plugins/$(PLUGIN_NAME)
	rm $(SO_INSTALL_PATH)/inkpath.so

# Used to install the plugin into a source code repository of xournalpp
dev-install:
	cp -r $(PLUGIN_NAME) ../xournalpp/plugins
	cp -r HACKING/StrokeTest ../xournalpp/plugins
	cp $(PLUGIN_NAME)/inkpath.so ../xournalpp/build/

dev-uninstall:
	rm -rf ../xournalpp/plugins/$(PLUGIN_NAME)
	rm -rf HACKING/StrokeTest ../xournalpp/plugins
	rm ../xournalpp/build/inkpath.so

stroketest:
	cp -r HACKING/StrokeTest ../xournalpp/plugins

stroketest-uninstall:
	rm -rf HACKING/StrokeTest ../xournalpp/plugins

clean:
	rm -rf build
	rm $(PLUGIN_NAME)/inkpath.so
