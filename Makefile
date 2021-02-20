cc=gcc

all:
    gcc sample.c `pkg-config --libs autotrace` `pkg-config --cflags autotrace`