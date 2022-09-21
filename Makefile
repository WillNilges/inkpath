CC=gcc

# Warnings
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wformat=2 -Winit-self \
	-Wmissing-include-dirs -Wformat-nonliteral -Wnested-externs \
	-Wno-unused-parameter -Wold-style-definition -Wredundant-decls -Wshadow \
	-Wstrict-prototypes -Wwrite-strings
#LIGHT_WARNINGS = -Wall
PLUGIN_NAME=ImageTranscription
LUA_VERSION=5.4
SO_INSTALL_PATH=/usr/lib64/lua/$(LUA_VERSION)# Just one of many possible destinations :)

.PHONY: clean install uninstall dev-install dev-uninstall

# TODO: `-g` is for debugging. Make a target that supports debugging separately from primary compilation

ip_source := $(wildcard src/ipcv_obj/*.cpp)
cv_source := $(wildcard src/cv/*.cpp)

luashit=`pkg-config --cflags --libs lua`
cvshit=`pkg-config --cflags --libs --static opencv4`

.PHONY: build_dir
build_dir:
	@mkdir -p build

#ipcvobj.so: src/ipcv_obj/*.cpp
#	@mkdir -p build/ipcv_obj
#	$(CC) -c $< $(luashit) $(cvshit) -fPIC -static
#	@mv *.o build/ipcv_obj
#	ar -crsT build/ipcv_obj/libipcv.a build/*.o #/scratch/opencv/build/lib/*.a
##@mv $@ build/ipcv_obj

ipcv: $(ip_source) $(cv_source)
	@mkdir -p build
	g++ -c $(cv_source) src/cv/debug/debug.cpp $(luashit) $(cvshit) -fPIC -static
	@mv *.o build
	ar -crsT build/libipcv.a build/*.o #/scratch/opencv/build/lib/*.a

lua-plugin: $(ip_source) ipcv
	g++ $(LIGHT_WARNINGS) $(ip_source) -L/xopp-dev/inkpath/build -lipcv $(cvshit) $(luashit) -g -fPIC -shared -o $(PLUGIN_NAME)/ipcvobj.so

install: lua-plugin
	cp -r $(PLUGIN_NAME) /usr/share/xournalpp/plugins/
	mkdir -p $(SO_INSTALL_PATH)
	cp -r $(PLUGIN_NAME)/ipcvobj.so $(SO_INSTALL_PATH)/ipcvobj.so

uninstall:
	rm -rf /usr/share/xournalpp/plugins/$(PLUGIN_NAME)
	rm $(SO_INSTALL_PATH)/ipcvobj.so

# Used to install the plugin into a source code repository of xournalpp
dev-install: lua-plugin
	cp -r $(PLUGIN_NAME) ../xournalpp/plugins
	cp -r HACKING/StrokeTest ../xournalpp/plugins
	cp $(PLUGIN_NAME)/ipcvobj.so ../xournalpp/build/
#	cp build/ipcv.so ../xournalpp/build/

dev-uninstall:
	rm -rf ../xournalpp/plugins/$(PLUGIN_NAME)
	rm -rf HACKING/StrokeTest ../xournalpp/plugins
	rm ../xournalpp/build/ipcvobj.so

# For generating a CV debugging binary
ipcv-debug: $(cv_source) 
	mkdir -p build
	g++ src/cv/debug/debug.cpp $(cv_source) `pkg-config --cflags --libs --static opencv4` -static -o build/ipcv

clean:
	rm -rf build
	rm $(PLUGIN_NAME)/ipcvobj.so
