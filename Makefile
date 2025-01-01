# DEPRECATED DO NOT USE

CC=gcc
CXX=g++

# Warnings
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wformat=2 -Winit-self \
	-Wmissing-include-dirs -Wformat-nonliteral -Wnested-externs \
	-Wno-unused-parameter -Wold-style-definition -Wredundant-decls -Wshadow \
	-Wstrict-prototypes -Wwrite-strings
LIGHT_WARNINGS = -Wall
PLUGIN_NAME=ImageTranscription
LIB_NAME=inkpath.a
ARTIFACT=build/$(PLUGIN_NAME)
INSTALL_PATH=/usr/share/xournalpp/plugins/
XOPP_DEV_INSTALL_PATH=/xournalpp
LUA_VERSION=lua54

.PHONY: clean install uninstall dev-install dev-uninstall

ip_source := $(wildcard src/ipcv_obj/*.cpp)
cv_source := $(wildcard src/cv/*.cpp)

lua_deps=`pkg-config --cflags --libs --static $(LUA_VERSION)`
cv_deps=`pkg-config --cflags --libs --static opencv4`

.PHONY: build_dir
build_dir:
	@mkdir -p build

# Remove the plugin files from the xournalpp install dir
uninstall:
	rm -rf $(INSTALL_PATH)$(PLUGIN_NAME)

# Used to install the plugin into a source code repository of xournalpp
# TODO: Port to CMake
#dev-install: plugin
#	cp -r $(ARTIFACT) $(XOPP_DEV_INSTALL_PATH)/plugins
#	cp -r HACKING/StrokeTest $(XOPP_DEV_INSTALL_PATH)/plugins

# Remove the plugin from the development environment
dev-uninstall:
	rm -rf $(XOPP_DEV_INSTALL_PATH)/plugins/$(PLUGIN_NAME)
	rm -rf HACKING/StrokeTest $(XOPP_DEV_INSTALL_PATH)/plugins

help:
	@echo uninstall dev-uninstall 

clean:
	rm -rf build
