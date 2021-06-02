#include "util.h"

int main(int argc, char *argv[])
{
    char* temp_xoj = "temp.xoj";
    int color_count = 2;

    invoke_autotrace(argv[1], temp_xoj, color_count, "FFFFFF");
    xoj_compress(temp_xoj, argv[2]);
    remove(temp_xoj);

    return 0;
}