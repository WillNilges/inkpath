cc=gcc

all:
	gcc main.c `pkg-config --libs autotrace` `pkg-config --cflags autotrace`
