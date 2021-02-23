cc=gcc

whitetrace:
	gcc main.c `pkg-config --libs autotrace` `pkg-config --cflags autotrace` -o whitetrace

clean:
	rm whitetrace
