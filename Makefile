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

# CFLAGS += -std=gnu99

.PHONY: clean install uninstall dev-install dev-uninstall

# TODO: `-g` is for debugging. Make a target that supports debugging separately from primary compilation

ip_source := $(wildcard src/*.cpp src/*.h)
at_source := $(wildcard src/autotrace/*.c src/autotrace/*.h)
cv_source := $(wildcard src/cv/*.cpp src/cv/*.h)
cv_deps=-I/usr/local/include/opencv4/opencv -I/usr/local/include/opencv4 -L/usr/local/lib/opencv4

#-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio

ipcv: $(cv_source) 
	mkdir -p build
	g++ $(cv_source) `pkg-config --cflags --libs --static opencv4` -static -o build/ipcv

ipcv-lua: $(ip_source) $(cv_source)
	@mkdir -p build
	g++ $(cv_source) $(ip_source) `pkg-config --cflags --libs --static opencv4` `pkg-config --cflags --libs lua` -fPIC -shared -o $(PLUGIN_NAME)/inkpath.lib

lua-plugin: $(ip_source) $(at_source) $(cv_source)
	@mkdir -p build
	g++ -c $(cv_source) `pkg-config --cflags --libs --static opencv4` -static  -fPIC # This works, but I need it to not go into the base dir of the project...
	@mv *.o build # God I'm such a fucking asshole
	ar -crs build/libipcv.a build/*.o  
	$(CC) $(LIGHT_WARNINGS) $(CFLAGS) $(ip_source) $(at_source) -static /xopp-dev/inkpath/build/libipcv.a -g `pkg-config --cflags --libs lua glib-2.0` -fPIC -shared -o $(PLUGIN_NAME)/inkpath.so

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
	#cp $(PLUGIN_NAME)/inkpath.so ../xournalpp/build/
	cp $(PLUGIN_NAME)/inkpath.lib ../xournalpp/build/
#	cp build/ipcv.so ../xournalpp/build/

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
