CC=gcc

# Warnings
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wformat=2 -Winit-self \
	-Wmissing-include-dirs -Wformat-nonliteral -Wnested-externs \
	-Wno-unused-parameter -Wold-style-definition -Wredundant-decls -Wshadow \
	-Wstrict-prototypes -Wwrite-strings
#LIGHT_WARNINGS = -Wall
PLUGIN_NAME=ImageTranscription
SO_NAME=ipcvobj.so
LUA_VERSION=5.4
SO_INSTALL_PATH=/usr/lib64/lua/$(LUA_VERSION)# Just one of many possible destinations :)
XOPP_DEV_INSTALL_PATH=/xournalpp

.PHONY: clean install uninstall dev-install dev-uninstall

# TODO: `-g` is for debugging. Make a target that supports debugging separately from primary compilation

ip_source := $(wildcard src/ipcv_obj/*.cpp)
cv_source := $(wildcard src/cv/*.cpp)

luashit=`pkg-config --cflags --libs lua`
cvshit=`pkg-config --cflags --libs --static opencv4`

.PHONY: build_dir
build_dir:
	@mkdir -p build

# Compiles and statically links Inkpath's OpenCV code to the necessary OpenCV libraries
ipcv: $(ip_source) $(cv_source)
	@mkdir -p build
	g++ -c $(cv_source) $(luashit) $(cvshit) -fPIC -static
	@mv *.o build
	ar -crsT build/libipcv.a build/*.o

# Compiles Inkpath's shared object library
lua-plugin: $(ip_source) ipcv
	g++ $(LIGHT_WARNINGS) $(ip_source) -L/xopp-dev/inkpath/build -lipcv $(cvshit) $(luashit) -g -fPIC -shared -o $(PLUGIN_NAME)/$(SO_NAME)

# Installs the plugin into your Xournalpp installation
install: lua-plugin
	cp -r $(PLUGIN_NAME) /usr/share/xournalpp/plugins/
	mkdir -p $(SO_INSTALL_PATH)
	cp -r $(PLUGIN_NAME)/$(SO_NAME) $(SO_INSTALL_PATH)/$(SO_NAME)

uninstall:
	rm -rf /usr/share/xournalpp/plugins/$(PLUGIN_NAME)
	rm $(SO_INSTALL_PATH)/$(SO_NAME)

# Used to install the plugin into a source code repository of xournalpp
dev-install: lua-plugin
	cp -r $(PLUGIN_NAME) $(XOPP_DEV_INSTALL_PATH)/plugins
	cp -r HACKING/StrokeTest $(XOPP_DEV_INSTALL_PATH)/plugins
	cp $(PLUGIN_NAME)/$(SO_NAME) $(XOPP_DEV_INSTALL_PATH)/build/

dev-uninstall:
	rm -rf $(XOPP_DEV_INSTALL_PATH)/plugins/$(PLUGIN_NAME)
	rm -rf HACKING/StrokeTest $(XOPP_DEV_INSTALL_PATH)/plugins
	rm $(XOPP_DEV_INSTALL_PATH)/$(SO_NAME)

# For generating a CV debugging binary
ipcv-debug: $(cv_source) 
	mkdir -p build
	g++ src/cv/debug/debug.cpp $(cv_source) `pkg-config --cflags --libs --static opencv4` -static -o build/ipcv

clean:
	rm -rf build
	rm $(PLUGIN_NAME)/$(SO_NAME)
