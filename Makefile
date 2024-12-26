CC=gcc

# Warnings
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wformat=2 -Winit-self \
	-Wmissing-include-dirs -Wformat-nonliteral -Wnested-externs \
	-Wno-unused-parameter -Wold-style-definition -Wredundant-decls -Wshadow \
	-Wstrict-prototypes -Wwrite-strings
LIGHT_WARNINGS = -Wall
PLUGIN_NAME=ImageTranscription
SO_NAME=ipcvobj.so
XOPP_DEV_INSTALL_PATH=/xournalpp
LUA_VERSION=lua53

.PHONY: clean install uninstall dev-install dev-uninstall

ip_source := $(wildcard src/ipcv_obj/*.cpp)
cv_source := $(wildcard src/cv/*.cpp)

lua_deps=`pkg-config --cflags --libs --static $(LUA_VERSION)`
cv_deps=`pkg-config --cflags --libs --static opencv4`

.PHONY: build_dir
build_dir:
	@mkdir -p build

# Compiles and statically links Inkpath's OpenCV code to the necessary OpenCV libraries
ipcv: $(cv_source)
	@mkdir -p build
	g++ -c $(cv_source) $(lua_deps) $(cv_deps) -fPIC -static
	@mv *.o build
	ar -crsT build/libipcv.a build/*.o

# Compiles Inkpath's shared object library
lua-plugin: $(ip_source) ipcv
	g++ $(LIGHT_WARNINGS) $(ip_source) -L./build -lipcv $(cv_deps) $(lua_deps) -g -fPIC -shared -o $(PLUGIN_NAME)/$(SO_NAME)

# Installs the plugin into your Xournalpp installation
# FIXME: Not smart enough to avoid re-building the app every time :(
install: lua-plugin
	cp -r $(PLUGIN_NAME) /usr/share/xournalpp/plugins/

# Remove the plugin files from the xournalpp install dir
uninstall:
	rm -rf /usr/share/xournalpp/plugins/$(PLUGIN_NAME)

# Used to install the plugin into a source code repository of xournalpp
dev-install: lua-plugin
	cp -r $(PLUGIN_NAME) $(XOPP_DEV_INSTALL_PATH)/plugins
	cp -r HACKING/StrokeTest $(XOPP_DEV_INSTALL_PATH)/plugins

# Remove the plugin from the development environment
dev-uninstall:
	rm -rf $(XOPP_DEV_INSTALL_PATH)/plugins/$(PLUGIN_NAME)
	rm -rf HACKING/StrokeTest $(XOPP_DEV_INSTALL_PATH)/plugins

# For generating a CV debugging binary
ipcv-debug: $(cv_source) 
	mkdir -p build
	g++ src/cv/debug/debug.cpp $(cv_source) $(cv_deps) -static -o build/ipcv-debug

help:
	@echo ipcv lua-plugin install uninstall dev-install dev-uninstall ipcv-debug

clean:
	rm -rf build
	rm $(PLUGIN_NAME)/$(SO_NAME)
