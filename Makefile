cc=gcc

inkpath:
	gcc main.c `pkg-config --libs autotrace` `pkg-config --cflags autotrace` -o build/inkpath

clean:
	rm build/inkpath
