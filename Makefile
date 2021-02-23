cc=gcc

inkpath:
	gcc main.c -g `pkg-config --libs autotrace glib-2.0` `pkg-config --cflags autotrace glib-2.0` -o build/inkpath

clean:
	rm build/inkpath
